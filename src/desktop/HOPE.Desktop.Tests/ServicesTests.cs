using HOPE.Core.Models;
using HOPE.Core.Protocols;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.OBD;

namespace HOPE.Desktop.Tests;

public class ServicesTests
{
    [Fact]
    public async Task MockOBD2Service_Connect_Success()
    {
        // Arrange
        var service = new MockOBD2Service();

        // Act
        var result = await service.ConnectAsync("MOCK");

        // Assert
        Assert.True(result);
        Assert.True(service.IsConnected);
    }

    [Fact]
    public async Task MockOBD2Service_ReadRPM_ReturnsValue()
    {
        // Arrange
        var service = new MockOBD2Service();
        await service.ConnectAsync("MOCK");

        // Act
        var reading = await service.ReadPIDAsync(OBD2PIDs.EngineRPM);

        // Assert
        Assert.Equal(OBD2PIDs.EngineRPM, reading.PID);
        Assert.True(reading.Value > 0);
        Assert.Equal("RPM", reading.Unit);
    }

    [Fact]
    public async Task ECUService_ReadMap_ReturnsGeneratedData()
    {
        // Arrange
        // We need a dummy protocol for ECU service
        var mockProtocol = new DummyProtocol();
        var service = new ECUService(mockProtocol);

        // Act
        var map = await service.ReadMapAsync("FuelMap");

        // Assert
        Assert.NotNull(map);
        Assert.Equal(8, map.GetLength(0));
        Assert.Equal(8, map.GetLength(1));
    }

    private class DummyProtocol : IDiagnosticProtocol
    {
        public string Name => "Dummy";
        public Task<bool> StartSessionAsync(byte sessionType) => Task.FromResult(true);
        public Task<bool> SecurityAccessAsync(byte accessType, byte[]? keyOverride = null) => Task.FromResult(true);
        public Task<byte[]> ReadMemoryAsync(long address, int length) => Task.FromResult(new byte[length]);
        public Task<byte[]> SendRequestAsync(byte serviceId, byte[] data) => Task.FromResult(Array.Empty<byte>());
    }
}
