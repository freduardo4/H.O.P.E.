using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Models.Simulation;
using HOPE.Core.Services.Simulation;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Simulation;

public class SimulationOrchestratorTests
{
    private readonly Mock<IBeamNgService> _beamNgMock = new();
    private readonly Mock<ILogger<SimulationOrchestrator>> _loggerMock = new();

    [Fact]
    public async Task ValidateInSimulationAsync_ShouldReturnSuccess_WhenTelemetryIsStable()
    {
        // Arrange
        var telemetrySubject = new Subject<BeamNgTelemetry>();
        _beamNgMock.Setup(b => b.IsConnected).Returns(true);
        _beamNgMock.Setup(b => b.TelemetryStream).Returns(telemetrySubject.AsObservable());
        
        var orchestrator = new SimulationOrchestrator(_beamNgMock.Object, _loggerMock.Object);

        // Act
        var validationTask = orchestrator.ValidateInSimulationAsync(new byte[] { 0x01 });
        
        // Push a telemetry packet
        telemetrySubject.OnNext(new BeamNgTelemetry { Car = "SIM\0".ToCharArray(), Rpm = 800 });

        var (success, message) = await validationTask;

        // Assert
        Assert.True(success);
        Assert.Contains("Virtual validation successful", message);
    }

    [Fact]
    public async Task ValidateInSimulationAsync_ShouldReturnFailure_WhenSimulationNotRespondingSegmented()
    {
        // Arrange
        _beamNgMock.Setup(b => b.IsConnected).Returns(false);
        _beamNgMock.Setup(b => b.StartAsync(It.IsAny<int>(), It.IsAny<CancellationToken>()))
                   .ThrowsAsync(new Exception("Socket error"));

        var orchestrator = new SimulationOrchestrator(_beamNgMock.Object, _loggerMock.Object);

        // Act
        var (success, message) = await orchestrator.ValidateInSimulationAsync(new byte[] { 0x01 });

        // Assert
        Assert.False(success);
        Assert.Contains("simulation is not responding", message);
    }

    [Fact]
    public async Task ValidateInSimulationAsync_ShouldReturnFailure_WhenTelemetrySyncTimeOut()
    {
        // Arrange
        var telemetrySubject = new Subject<BeamNgTelemetry>();
        _beamNgMock.Setup(b => b.IsConnected).Returns(true);
        _beamNgMock.Setup(b => b.TelemetryStream).Returns(telemetrySubject.AsObservable());

        var orchestrator = new SimulationOrchestrator(_beamNgMock.Object, _loggerMock.Object);

        // Act: Don't push any telemetry, wait for timeout
        var (success, message) = await orchestrator.ValidateInSimulationAsync(new byte[] { 0x01 });

        // Assert
        Assert.False(success);
        Assert.Contains("Digital Twin sync timed out", message);
    }
}
