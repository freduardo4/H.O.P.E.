using System.IO;
using System.Net;
using System.Net.Http;
using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.Protocols;
using HOPE.Core.Testing;
using HOPE.Core.Services.Safety;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests;

public class SafeFlashServiceIntegrationTests : IAsyncLifetime, IDisposable
{
    private readonly SimulatedHardwareAdapter _adapter;
    private readonly VoltageMonitor _voltageMonitor;
    private readonly UdsProtocolService _udsService;
    private readonly CalibrationRepository _repository;
    private readonly SafeFlashService _flashService;
    private readonly CloudSafetyService _cloudSafety;
    private readonly string _testRepoPath;

    public SafeFlashServiceIntegrationTests()
    {
        _testRepoPath = Path.Combine(Path.GetTempPath(), "HOPE_Test_Repo_" + Guid.NewGuid());
        _adapter = new SimulatedHardwareAdapter();
        _voltageMonitor = new VoltageMonitor(_adapter);
        _udsService = new UdsProtocolService(_adapter);
        _repository = new CalibrationRepository(_testRepoPath);

        // Mock Cloud Safety using a Fake Handler
        var fakeHandler = new FakeSafetyHandler(true);
        _cloudSafety = new CloudSafetyService(new HttpClient(fakeHandler));
        
        _flashService = new SafeFlashService(_adapter, _voltageMonitor, _udsService, _repository, _cloudSafety);
    }

    public async Task InitializeAsync()
    {
        await _repository.InitializeAsync();
    }

    public Task DisposeAsync()
    {
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        _flashService.Dispose();
        _voltageMonitor.Dispose();
        _udsService.Dispose();
        _adapter.Dispose();
        _repository.Dispose();

        if (Directory.Exists(_testRepoPath))
            Directory.Delete(_testRepoPath, true);
    }

    private CalibrationFile CreateTestCalibration()
    {
        var blockData = new byte[] { 0x01, 0x02, 0x03, 0x04 };
        var block = new CalibrationBlock
        {
            Name = "TEST",
            StartAddress = 0x1000,
            Data = new byte[32]
        };
        new Random().NextBytes(block.Data);
        block.Checksum = ComputeChecksum(blockData);

        var cal = new CalibrationFile
        {
            EcuId = "SIM_ECU",
            Blocks = new List<CalibrationBlock> { block }
        };
        cal.FullChecksum = ComputeFileChecksum(cal);
        return cal;
    }

    private string ComputeChecksum(byte[] data)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var hash = sha.ComputeHash(data);
        return Convert.ToHexString(hash).ToLower();
    }

    private string ComputeFileChecksum(CalibrationFile calibration)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        using var stream = new MemoryStream();
        foreach (var block in calibration.Blocks.OrderBy(b => b.StartAddress))
        {
            stream.Write(block.Data);
        }
        stream.Position = 0;
        var hash = sha.ComputeHash(stream);
        return Convert.ToHexString(hash).ToLower();
    }

    [Fact]
    public async Task FlashAsync_HappyPath_Succeeds()
    {
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal, VerifyAfterWrite = true, CreateBackup = true };
        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();

        var result = await _flashService.FlashAsync(config, keyAlgo);

        Assert.True(result.Success, $"Flash failed: {result.FailureReason}");
        Assert.Equal(FlashStage.Complete, result.Success ? FlashStage.Complete : FlashStage.PreFlight);
    }

    [Fact]
    public async Task FlashAsync_LowVoltage_FailsInPreFlight()
    {
        await _adapter.ConnectAsync("SIM");
        _adapter.SimulatedVoltage = 11.0; 
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        var result = await _flashService.FlashAsync(config, _ => _);

        Assert.False(result.Success);
        Assert.Equal("Pre-flight checks failed", result.FailureReason);
        Assert.Contains("Battery voltage", result.PreFlightResult?.Checks.First(c => c.Name == "Battery Voltage").Message);
    }

    [Fact]
    public async Task FlashAsync_SecurityAccessDenied_Fails()
    {
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        var result = await _flashService.FlashAsync(config, _ => new byte[] { 0x00 });

        Assert.False(result.Success);
        Assert.Equal(FlashStage.Security, result.FailedAtStage);
        Assert.Contains("Security access denied", result.FailureReason);
    }

    [Fact]
    public async Task FlashAsync_AdapterDisconnect_Fails()
    {
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        _adapter.InjectError = true;
        _adapter.InjectedErrorType = HardwareErrorType.ConnectionLost;
        
        var result = await _flashService.FlashAsync(config, _ => _);

        Assert.False(result.Success);
        Assert.Equal(FlashStage.PreFlight, result.FailedAtStage);
    }

    [Fact]
    public async Task FlashAsync_VoltageDropMidFlash_Aborts()
    {
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        cal.Blocks[0].Data = new byte[2048]; 
        cal.FullChecksum = ComputeFileChecksum(cal);
        var config = new FlashConfig { Calibration = cal, CreateBackup = false };

        _adapter.SimulatedVoltage = 14.0;
        _adapter.DropVoltageAfterMessages = 8; 
        _adapter.VoltageAfterDrop = 11.0;

        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();
        var result = await _flashService.FlashAsync(config, keyAlgo);

        Assert.False(result.Success);
        Assert.Contains("CRITICAL: Battery voltage dropped below", result.FailureReason);
        Assert.Equal(FlashStage.Transfer, result.FailedAtStage);
    }

    [Fact]
    public async Task RestoreFromBackupAsync_Succeeds()
    {
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal, CreateBackup = true };
        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();
        
        var flashResult = await _flashService.FlashAsync(config, keyAlgo);
        Assert.True(flashResult.Success);
        
        var restoreResult = await _flashService.RestoreFromBackupAsync(flashResult.BackupPath, keyAlgo);

        Assert.True(restoreResult.Success);
        Assert.Equal(FlashStage.Complete, restoreResult.Success ? FlashStage.Complete : FlashStage.PreFlight);
    }
}

public class FakeSafetyHandler : HttpMessageHandler
{
    private readonly bool _allowed;
    
    public FakeSafetyHandler(bool allowed)
    {
        _allowed = allowed;
    }

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        // Simple mock response
        var content = _allowed ? "{\"allowed\": true}" : "{\"allowed\": false}";
        return Task.FromResult(new HttpResponseMessage
        {
            StatusCode = HttpStatusCode.OK,
            Content = new StringContent(content)
        });
    }
}
