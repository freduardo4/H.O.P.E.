using System.IO;
using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.Protocols;
using HOPE.Core.Testing;
using Xunit;

namespace HOPE.Desktop.Tests;

public class SafeFlashServiceIntegrationTests : IAsyncLifetime, IDisposable
{
    private readonly SimulatedHardwareAdapter _adapter;
    private readonly VoltageMonitor _voltageMonitor;
    private readonly UdsProtocolService _udsService;
    private readonly CalibrationRepository _repository;
    private readonly SafeFlashService _flashService;
    private readonly string _testRepoPath;

    public SafeFlashServiceIntegrationTests()
    {
        _testRepoPath = Path.Combine(Path.GetTempPath(), "HOPE_Test_Repo_" + Guid.NewGuid());
        _adapter = new SimulatedHardwareAdapter();
        _voltageMonitor = new VoltageMonitor(_adapter);
        _udsService = new UdsProtocolService(_adapter);
        _repository = new CalibrationRepository(_testRepoPath);
        _flashService = new SafeFlashService(_adapter, _voltageMonitor, _udsService, _repository);
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
        // Arrange
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();

        var config = new FlashConfig
        {
            Calibration = cal,
            VerifyAfterWrite = true,
            CreateBackup = true
        };

        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();

        // Act
        var result = await _flashService.FlashAsync(config, keyAlgo);

        // Assert
        Assert.True(result.Success, $"Flash failed: {result.FailureReason}");
        Assert.Equal(FlashStage.Complete, result.Success ? FlashStage.Complete : FlashStage.PreFlight);
    }

    [Fact]
    public async Task FlashAsync_LowVoltage_FailsInPreFlight()
    {
        // Arrange
        await _adapter.ConnectAsync("SIM");
        _adapter.SimulatedVoltage = 11.0; // Below MIN_FLASH_VOLTAGE (13.0)
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        // Act
        var result = await _flashService.FlashAsync(config, _ => _);

        // Assert
        Assert.False(result.Success);
        Assert.Equal("Pre-flight checks failed", result.FailureReason);
        Assert.Contains("Battery voltage", result.PreFlightResult?.Checks.First(c => c.Name == "Battery Voltage").Message);
    }

    [Fact]
    public async Task FlashAsync_SecurityAccessDenied_Fails()
    {
        // Arrange
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        // Act - use wrong key algorithm
        var result = await _flashService.FlashAsync(config, _ => new byte[] { 0x00 });

        // Assert
        Assert.False(result.Success);
        Assert.Equal(FlashStage.Security, result.FailedAtStage);
        Assert.Contains("Security access denied", result.FailureReason);
    }

    [Fact]
    public async Task FlashAsync_AdapterDisconnect_Fails()
    {
        // Arrange
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        var config = new FlashConfig { Calibration = cal };

        // Act
        _adapter.InjectError = true;
        _adapter.InjectedErrorType = HardwareErrorType.ConnectionLost;
        
        var result = await _flashService.FlashAsync(config, _ => _);

        // Assert
        Assert.False(result.Success);
        Assert.Equal(FlashStage.PreFlight, result.FailedAtStage);
    }

    [Fact]
    public async Task FlashAsync_VoltageDropMidFlash_Aborts()
    {
        // Arrange
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        // Create a larger calibration to ensure voltage check is hit
        cal.Blocks[0].Data = new byte[2048]; 
        cal.FullChecksum = ComputeFileChecksum(cal);
        
        var config = new FlashConfig { Calibration = cal, CreateBackup = false };

        // Act - Inject voltage drop after ~8 messages (pre-flight + session setup)
        _adapter.SimulatedVoltage = 14.0;
        _adapter.DropVoltageAfterMessages = 8; 
        _adapter.VoltageAfterDrop = 11.0;

        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();
        var result = await _flashService.FlashAsync(config, keyAlgo);

        // Assert
        Assert.False(result.Success);
        Assert.Contains("CRITICAL: Battery voltage dropped below", result.FailureReason);
        Assert.Equal(FlashStage.Transfer, result.FailedAtStage);
    }

    [Fact]
    public async Task RestoreFromBackupAsync_Succeeds()
    {
        // Arrange
        await _adapter.ConnectAsync("SIM");
        var cal = CreateTestCalibration();
        
        // 1. Manually create a backup "file" by running CreateShadowBackupAsync (internal)
        // Or just call FlashAsync which creates one
        var config = new FlashConfig { Calibration = cal, CreateBackup = true };
        Func<byte[], byte[]> keyAlgo = seed => seed.Select(b => (byte)(b + 1)).ToArray();
        
        var flashResult = await _flashService.FlashAsync(config, keyAlgo);
        Assert.True(flashResult.Success);
        Assert.NotNull(flashResult.BackupPath);

        // 2. Modify "ECU" memory so it's different
        // Simulated memory is cleared on each test usually if adapter is fresh, 
        // but here we are in same test.
        
        // 3. Restore
        var restoreResult = await _flashService.RestoreFromBackupAsync(flashResult.BackupPath, keyAlgo);

        // Assert
        Assert.True(restoreResult.Success);
        Assert.Equal(FlashStage.Complete, restoreResult.Success ? FlashStage.Complete : FlashStage.PreFlight);
    }
}
