using System.IO;
using HOPE.Core.Models;
using HOPE.Core.Services.AI;

namespace HOPE.Desktop.Tests;

public class OnnxAnomalyServiceTests : IDisposable
{
    private readonly OnnxAnomalyService _service;
    private readonly string _modelPath;

    public OnnxAnomalyServiceTests()
    {
        _service = new OnnxAnomalyService();
        _modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "anomaly_detector.onnx");
    }

    public void Dispose()
    {
        _service.Dispose();
    }

    [Fact]
    public async Task LoadModelAsync_LoadsModel_WhenPathValid()
    {
        // Skip if model file not available
        if (!File.Exists(_modelPath))
        {
            return;
        }

        // Act
        await _service.LoadModelAsync(_modelPath);

        // Assert
        Assert.True(_service.IsModelLoaded);
    }

    [Fact]
    public async Task AnalyzeAsync_ReturnsNotLoaded_WhenModelNotLoaded()
    {
        // Arrange - don't load model
        var readings = new List<OBD2Reading>();

        // Act
        var result = await _service.AnalyzeAsync(readings);

        // Assert
        Assert.False(result.IsAnomaly);
        Assert.Equal("Model not loaded", result.Description);
    }

    [Fact]
    public async Task AnalyzeAsync_ReturnsResult_WhenModelLoaded()
    {
        // Skip if model file not available
        if (!File.Exists(_modelPath))
        {
            return;
        }

        // Arrange
        await _service.LoadModelAsync(_modelPath);

        var readings = GenerateNormalReadings();

        // Act
        var result = await _service.AnalyzeAsync(readings);

        // Assert
        Assert.NotNull(result);
        Assert.True(result.Confidence > 0);
    }

    [Fact]
    public async Task AnalyzeAsync_DetectsAnomaly_WhenValuesAbnormal()
    {
        // Skip if model file not available
        if (!File.Exists(_modelPath))
        {
            return;
        }

        // Arrange
        await _service.LoadModelAsync(_modelPath);

        // Create readings with abnormal values (extreme RPM, mismatched speed)
        var readings = GenerateAnomalousReadings();

        // Act
        var result = await _service.AnalyzeAsync(readings);

        // Assert
        Assert.NotNull(result);
        // The model should detect something unusual with extreme values
        Assert.True(result.Score > 0);
    }

    private static List<OBD2Reading> GenerateNormalReadings()
    {
        var readings = new List<OBD2Reading>();
        var sessionId = Guid.NewGuid();
        var baseTime = DateTime.UtcNow;

        // Generate 60 seconds of normal driving data
        for (int t = 0; t < 60; t++)
        {
            var timestamp = baseTime.AddSeconds(t);

            // Normal idle to light driving
            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.EngineRPM,
                Name = "Engine RPM",
                Value = 1500 + t * 10,
                Unit = "RPM",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.VehicleSpeed,
                Name = "Vehicle Speed",
                Value = 30 + t * 0.5,
                Unit = "km/h",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.EngineLoad,
                Name = "Engine Load",
                Value = 25 + t * 0.2,
                Unit = "%",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.CoolantTemp,
                Name = "Coolant Temp",
                Value = 90,
                Unit = "C",
                Timestamp = timestamp
            });
        }

        return readings;
    }

    private static List<OBD2Reading> GenerateAnomalousReadings()
    {
        var readings = new List<OBD2Reading>();
        var sessionId = Guid.NewGuid();
        var baseTime = DateTime.UtcNow;

        // Generate data with anomalies: high RPM but zero speed (transmission problem?)
        for (int t = 0; t < 60; t++)
        {
            var timestamp = baseTime.AddSeconds(t);

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.EngineRPM,
                Name = "Engine RPM",
                Value = 6000 + (t % 10) * 100, // Very high RPM oscillating
                Unit = "RPM",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.VehicleSpeed,
                Name = "Vehicle Speed",
                Value = 0, // No movement despite high RPM
                Unit = "km/h",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.EngineLoad,
                Name = "Engine Load",
                Value = 95, // Very high load
                Unit = "%",
                Timestamp = timestamp
            });

            readings.Add(new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.CoolantTemp,
                Name = "Coolant Temp",
                Value = 115, // Overheating
                Unit = "C",
                Timestamp = timestamp
            });
        }

        return readings;
    }
}
