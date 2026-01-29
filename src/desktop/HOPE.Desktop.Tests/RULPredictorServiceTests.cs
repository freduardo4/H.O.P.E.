using HOPE.Core.Services.AI;

namespace HOPE.Desktop.Tests;

public class RULPredictorServiceTests
{
    [Fact]
    public async Task PredictMaintenance_WithoutTelemetry_ReturnsFallbackPrediction()
    {
        // Arrange
        var service = new RULPredictorService();

        // Act
        var result = await service.PredictMaintenanceAsync(
            "TEST001",
            50000,
            Array.Empty<ComponentTelemetry>(),
            50.0);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("TEST001", result.VehicleId);
        Assert.Equal(50000, result.OdometerKm);
        Assert.True(result.Success);
        Assert.Equal(10, result.Components.Count); // All 10 component types
    }

    [Fact]
    public async Task PredictMaintenance_WithTelemetry_CalculatesHealth()
    {
        // Arrange
        var service = new RULPredictorService();
        var telemetry = new[]
        {
            new ComponentTelemetry
            {
                Component = VehicleComponentType.Battery,
                SensorData = GenerateDegradationData(0.95, 0.75, 30)
            },
            new ComponentTelemetry
            {
                Component = VehicleComponentType.BrakePads,
                SensorData = GenerateDegradationData(1.0, 0.4, 30)
            }
        };

        // Act
        var result = await service.PredictMaintenanceAsync(
            "TEST002",
            75000,
            telemetry,
            60.0);

        // Assert
        Assert.True(result.Success);
        Assert.Equal(10, result.Components.Count);

        var battery = result.Components.First(c => c.Component == VehicleComponentType.Battery);
        Assert.True(battery.HealthScore > 0 && battery.HealthScore <= 1);
        Assert.True(battery.Confidence >= 0.5);

        var brakes = result.Components.First(c => c.Component == VehicleComponentType.BrakePads);
        Assert.True(brakes.HealthScore >= 0 && brakes.HealthScore <= 1);
    }

    [Fact]
    public async Task PredictMaintenance_IdentifiesUrgentItems()
    {
        // Arrange
        var service = new RULPredictorService();
        var telemetry = new[]
        {
            new ComponentTelemetry
            {
                Component = VehicleComponentType.BrakePads,
                SensorData = new double[] { 0.25, 0.22, 0.20, 0.18, 0.15 } // Below threshold
            }
        };

        // Act
        var result = await service.PredictMaintenanceAsync(
            "TEST003",
            100000,
            telemetry,
            50.0);

        // Assert
        Assert.True(result.Success);

        var brakes = result.Components.First(c => c.Component == VehicleComponentType.BrakePads);
        Assert.Equal(WarningLevel.Critical, brakes.WarningLevel);
        Assert.True(result.UrgentItems.Count > 0);
        Assert.True(result.EstimatedMaintenanceCost > 0);
    }

    [Fact]
    public async Task PredictComponentRUL_SingleComponent_ReturnsHealth()
    {
        // Arrange
        var service = new RULPredictorService();
        var sensorData = GenerateDegradationData(0.9, 0.7, 20);

        // Act
        var result = await service.PredictComponentRULAsync(
            VehicleComponentType.O2Sensor,
            sensorData,
            80000,
            40.0);

        // Assert
        Assert.Equal(VehicleComponentType.O2Sensor, result.Component);
        Assert.True(result.HealthScore >= 0 && result.HealthScore <= 1);
        Assert.True(result.EstimatedRulKm >= 0);
        Assert.True(result.EstimatedRulDays >= 0);
    }

    [Fact]
    public async Task PredictMaintenance_CalculatesOverallHealth()
    {
        // Arrange
        var service = new RULPredictorService();

        // Act
        var result = await service.PredictMaintenanceAsync(
            "TEST004",
            30000,
            Array.Empty<ComponentTelemetry>(),
            50.0);

        // Assert
        Assert.True(result.OverallHealth >= 0 && result.OverallHealth <= 1);
        Assert.True(result.NextRecommendedService > DateTime.Now);
    }

    [Fact]
    public async Task PredictMaintenance_ReportsProgress()
    {
        // Arrange
        var service = new RULPredictorService();
        var progressUpdates = new List<RULPredictionProgress>();
        var progress = new Progress<RULPredictionProgress>(p => progressUpdates.Add(p));

        // Act
        await service.PredictMaintenanceAsync(
            "TEST005",
            50000,
            Array.Empty<ComponentTelemetry>(),
            50.0,
            progress);

        // Allow time for progress updates to be processed
        await Task.Delay(100);

        // Assert
        Assert.True(progressUpdates.Count > 0);
        Assert.Contains(progressUpdates, p => p.PercentComplete == 100);
    }

    [Fact]
    public async Task PredictMaintenance_SupportsCancellation()
    {
        // Arrange
        var service = new RULPredictorService();
        var cts = new CancellationTokenSource();

        // Start prediction, then cancel mid-way
        var telemetry = Enumerable.Range(0, 5).Select(i => new ComponentTelemetry
        {
            Component = (VehicleComponentType)i,
            SensorData = GenerateDegradationData(0.9, 0.7, 100)
        }).ToList();

        // Cancel immediately to ensure deterministic behavior
        cts.Cancel();

        // Act
        var result = await service.PredictMaintenanceAsync(
            "TEST006",
            50000,
            telemetry,
            50.0,
            null,
            cts.Token);

        // Assert - either throws or returns with error message
        // The service should either throw or return an unsuccessful result
        Assert.True(
            !result.Success || result.ErrorMessage?.Contains("cancelled") == true || result.Components.Count < 10,
            "Cancellation should either throw, return unsuccessful result, or return partial results");
    }

    [Fact]
    public void ComponentHealth_WarningLevel_SetCorrectly()
    {
        // Arrange & Act
        var critical = new ComponentHealth { HealthScore = 0.2, WarningLevel = WarningLevel.Critical };
        var warning = new ComponentHealth { HealthScore = 0.5, WarningLevel = WarningLevel.Warning };
        var normal = new ComponentHealth { HealthScore = 0.9, WarningLevel = WarningLevel.Normal };

        // Assert
        Assert.Equal(WarningLevel.Critical, critical.WarningLevel);
        Assert.Equal(WarningLevel.Warning, warning.WarningLevel);
        Assert.Equal(WarningLevel.Normal, normal.WarningLevel);
    }

    [Fact]
    public void VehicleComponentType_AllTypesAvailable()
    {
        // Arrange & Act
        var allTypes = Enum.GetValues<VehicleComponentType>();

        // Assert
        Assert.Equal(10, allTypes.Length);
        Assert.Contains(VehicleComponentType.CatalyticConverter, allTypes);
        Assert.Contains(VehicleComponentType.O2Sensor, allTypes);
        Assert.Contains(VehicleComponentType.SparkPlugs, allTypes);
        Assert.Contains(VehicleComponentType.Battery, allTypes);
        Assert.Contains(VehicleComponentType.BrakePads, allTypes);
        Assert.Contains(VehicleComponentType.AirFilter, allTypes);
        Assert.Contains(VehicleComponentType.FuelFilter, allTypes);
        Assert.Contains(VehicleComponentType.TimingBelt, allTypes);
        Assert.Contains(VehicleComponentType.Coolant, allTypes);
        Assert.Contains(VehicleComponentType.TransmissionFluid, allTypes);
    }

    [Fact]
    public void ForecasterScriptPath_ReturnsPath()
    {
        // Arrange
        var service = new RULPredictorService();

        // Act
        var path = service.ForecasterScriptPath;

        // Assert
        Assert.NotNull(path);
        Assert.EndsWith("rul_forecaster.py", path);
    }

    private static double[] GenerateDegradationData(double start, double end, int samples)
    {
        var data = new double[samples];
        var step = (end - start) / (samples - 1);

        for (int i = 0; i < samples; i++)
        {
            // Linear degradation with small noise
            data[i] = start + (step * i) + (Random.Shared.NextDouble() - 0.5) * 0.02;
        }

        return data;
    }
}
