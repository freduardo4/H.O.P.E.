using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.Hardware;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Services;

public class BenchControlServiceTests
{
    private readonly Mock<IHardwareAdapter> _hardwareMock;
    private readonly Mock<IBenchPowerSupply> _benchSupplyMock;
    private readonly Mock<ILogger<BenchControlService>> _loggerMock;
    private readonly BenchControlService _service;

    public BenchControlServiceTests()
    {
        _hardwareMock = new Mock<IHardwareAdapter>();
        _benchSupplyMock = _hardwareMock.As<IBenchPowerSupply>(); // Multi-interface mock
        _loggerMock = new Mock<ILogger<BenchControlService>>();
        _service = new BenchControlService(_hardwareMock.Object, _loggerMock.Object);
    }

    [Fact]
    public async Task SetIgnitionAsync_UsesIBenchPowerSupply_WhenAvailable()
    {
        // Arrange
        _benchSupplyMock.Setup(b => b.SetIgnitionAsync(true, It.IsAny<CancellationToken>()))
            .ReturnsAsync(true);

        // Act
        var result = await _service.SetIgnitionAsync(true);

        // Assert
        Assert.True(result);
        _benchSupplyMock.Verify(b => b.SetIgnitionAsync(true, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task SetIgnitionAsync_FallbacksToPin12_WhenNotIBenchPowerSupply()
    {
        // Arrange
        var basicHardwareMock = new Mock<IHardwareAdapter>(); // Does NOT implement IBenchPowerSupply
        var service = new BenchControlService(basicHardwareMock.Object, _loggerMock.Object);
        
        basicHardwareMock.Setup(h => h.SetProgrammingVoltageAsync(12, 12.0, It.IsAny<CancellationToken>()))
            .ReturnsAsync(true);

        // Act
        var result = await service.SetIgnitionAsync(true);

        // Assert
        Assert.True(result);
        basicHardwareMock.Verify(h => h.SetProgrammingVoltageAsync(12, 12.0, It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task SetIgnitionAsync_RespectsSafetyLock_WhenTurningOff()
    {
        // Arrange
        _service.SetSafetyLock(true);

        // Act
        var result = await _service.SetIgnitionAsync(false);

        // Assert
        Assert.False(result); // Should be blocked
        _benchSupplyMock.Verify(b => b.SetIgnitionAsync(false, It.IsAny<CancellationToken>()), Times.Never);
    }

    [Fact]
    public async Task SetIgnitionAsync_AllowsTurningOn_EvenWithSafetyLock()
    {
        // Arrange
        _service.SetSafetyLock(true);
        _benchSupplyMock.Setup(b => b.SetIgnitionAsync(true, It.IsAny<CancellationToken>())).ReturnsAsync(true);

        // Act
        var result = await _service.SetIgnitionAsync(true);

        // Assert
        Assert.True(result);
        _benchSupplyMock.Verify(b => b.SetIgnitionAsync(true, It.IsAny<CancellationToken>()), Times.Once);
    }
}
