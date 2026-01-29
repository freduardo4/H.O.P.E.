using HOPE.Core.Models.Simulation;
using HOPE.Core.Services.Simulation;
using Xunit;

namespace HOPE.Desktop.Tests.Services.Simulation;

public class HiLServiceTests
{
    [Fact]
    public void ProcessTelemetry_ApplySensorNoise_ModifiesRpm()
    {
        // Arrange
        var service = new HiLService();
        var raw = new BeamNgTelemetry { Rpm = 2000 };
        var fault = new HiLFault 
        { 
            Type = FaultType.SensorNoise, 
            TargetParameter = "Rpm", 
            Intensity = 1.0, 
            OccurredAt = DateTime.UtcNow 
        };

        service.InjectFault(fault);

        // Act
        var processed = service.ProcessTelemetry(raw);

        // Assert
        Assert.NotEqual(2000f, processed.Rpm);
        Assert.InRange(processed.Rpm, 2000f, 3000f);
    }

    [Fact]
    public void ProcessTelemetry_ApplyHighTemp_OverridesTemp()
    {
        // Arrange
        var service = new HiLService();
        var raw = new BeamNgTelemetry { EngineTemp = 90 };
        var fault = new HiLFault 
        { 
            Type = FaultType.HighTemp, 
            OccurredAt = DateTime.UtcNow 
        };

        service.InjectFault(fault);

        // Act
        var processed = service.ProcessTelemetry(raw);

        // Assert
        Assert.Equal(130.0f, processed.EngineTemp);
    }

    [Fact]
    public void ClearFaults_RemovesActiveFaults()
    {
        // Arrange
        var service = new HiLService();
        service.InjectFault(new HiLFault { Type = FaultType.VoltageDrop });

        // Act
        service.ClearFaults();

        // Assert
        Assert.False(service.IsActive);
        Assert.Empty(service.ActiveFaults);
    }
}
