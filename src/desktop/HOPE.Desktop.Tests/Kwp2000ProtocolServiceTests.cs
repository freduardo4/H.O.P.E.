using HOPE.Core.Interfaces;
using HOPE.Core.Services.Protocols;
using HOPE.Core.Testing;
using Xunit;

namespace HOPE.Desktop.Tests;

public class Kwp2000ProtocolServiceTests
{
    private readonly SimulatedHardwareAdapter _adapter;
    private readonly Kwp2000ProtocolService _service;

    public Kwp2000ProtocolServiceTests()
    {
        _adapter = new SimulatedHardwareAdapter();
        _adapter.ConnectAsync("test").Wait();
        _service = new Kwp2000ProtocolService(_adapter);
    }

    [Fact]
    public async Task InitializeFast_SendsCorrectCommands()
    {
        // Act
        var result = await _service.InitializeFastAsync();

        // Assert
        Assert.True(result);
        Assert.True(_service.IsInitialized);
    }

    [Fact]
    public async Task StartDiagnosticSession_ReturnsPositiveResponse()
    {
        // Arrange
        await _service.InitializeFastAsync();

        // Act
        var response = await _service.StartDiagnosticSessionAsync(Kwp2000Session.Extended);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(Kwp2000Session.Extended, _service.CurrentSession);
    }

    [Fact]
    public async Task RequestSecuritySeed_ReturnsSeed()
    {
        // Arrange
        await _service.InitializeFastAsync();
        await _service.StartDiagnosticSessionAsync(Kwp2000Session.Extended);

        // Act
        var response = await _service.RequestSecuritySeedAsync(0x01);

        // Assert
        Assert.True(response.IsPositive);
        Assert.NotEmpty(response.Seed);
    }

    [Fact]
    public async Task SendSecurityKey_UnlocksSecurity()
    {
        // Arrange
        await _service.InitializeFastAsync();
        await _service.StartDiagnosticSessionAsync(Kwp2000Session.Extended);
        var seedResponse = await _service.RequestSecuritySeedAsync(0x01);
        
        // Simple key for simulator: seed[i] + 1
        var key = seedResponse.Seed.Select(b => (byte)(b + 1)).ToArray();

        // Act
        var response = await _service.SendSecurityKeyAsync(0x01, key);

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task ReadMemoryByAddress_ReturnsData()
    {
        // Arrange
        await _service.InitializeFastAsync();
        
        // Act
        var response = await _service.ReadMemoryByAddressAsync(0x1234, 4);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(4, response.Data.Length);
    }

    [Fact]
    public async Task ReadDataByLocalIdentifier_ReturnsData()
    {
        // Arrange
        await _service.InitializeFastAsync();

        // Act
        var response = await _service.ReadDataByLocalIdentifierAsync(0x01);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(0x01, response.LocalIdentifier);
        Assert.NotEmpty(response.Data);
    }

    [Fact]
    public async Task RequestDownload_PreparesForTransfer()
    {
        // Arrange
        await _service.InitializeFastAsync();
        // Unlock security (mock logic handles download if unlocked)
        var seed = await _service.RequestSecuritySeedAsync(0x01);
        await _service.SendSecurityKeyAsync(0x01, seed.Seed.Select(b => (byte)(b + 1)).ToArray());

        // Act
        var response = await _service.RequestDownloadAsync(0x1000, 0x100);

        // Assert
        Assert.True(response.IsPositive);
        Assert.True(response.MaxBlockSize > 0);
    }

    [Fact]
    public async Task TransferData_SendsLargeBlock()
    {
        // Arrange
        await _service.InitializeFastAsync();
        var seed = await _service.RequestSecuritySeedAsync(0x01);
        await _service.SendSecurityKeyAsync(0x01, seed.Seed.Select(b => (byte)(b + 1)).ToArray());
        await _service.RequestDownloadAsync(0x1000, 0x100);

        var data = new byte[128];
        new Random().NextBytes(data);

        // Act
        var response = await _service.TransferDataAsync(data);

        // Assert
        Assert.True(response.IsPositive);
    }
}
