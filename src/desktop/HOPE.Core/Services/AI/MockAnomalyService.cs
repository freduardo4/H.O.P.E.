using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Mock implementation of anomaly detection for testing/development
/// </summary>
public class MockAnomalyService : IAnomalyService
{
    private readonly Random _random = new();
    private bool _modelLoaded = false;

    public bool IsModelLoaded => _modelLoaded;

    public Task LoadModelAsync(string modelPath)
    {
        // Simulate model loading
        _modelLoaded = true;
        return Task.CompletedTask;
    }

    public Task<AnomalyResult> AnalyzeAsync(IEnumerable<OBD2Reading> readings)
    {
        var readingsList = readings.ToList();
        
        if (readingsList.Count == 0)
        {
            return Task.FromResult(new AnomalyResult
            {
                Score = 0,
                IsAnomaly = false,
                Confidence = 0,
                Description = "Insufficient data for analysis"
            });
        }

        // Simulate anomaly detection based on reading patterns
        double anomalyScore = 0.0;
        var contributingParams = new List<string>();

        // Check for high RPM
        var rpmReadings = readingsList.Where(r => r.PID == OBD2PIDs.EngineRPM).ToList();
        if (rpmReadings.Any(r => r.Value > 6000))
        {
            anomalyScore += 0.3;
            contributingParams.Add("High RPM detected");
        }

        // Check for high coolant temp
        var tempReadings = readingsList.Where(r => r.PID == OBD2PIDs.CoolantTemp).ToList();
        if (tempReadings.Any(r => r.Value > 100))
        {
            anomalyScore += 0.4;
            contributingParams.Add("Elevated coolant temperature");
        }

        // Check for erratic engine load
        var loadReadings = readingsList.Where(r => r.PID == OBD2PIDs.EngineLoad).ToList();
        if (loadReadings.Count >= 2)
        {
            var variance = CalculateVariance(loadReadings.Select(r => r.Value));
            if (variance > 100)
            {
                anomalyScore += 0.2;
                contributingParams.Add("Erratic engine load fluctuations");
            }
        }

        // Add some randomness for demo purposes
        anomalyScore += _random.NextDouble() * 0.1;
        anomalyScore = Math.Min(anomalyScore, 1.0);

        bool isAnomaly = anomalyScore > 0.5;
        string description = isAnomaly
            ? "Potential issue detected - review contributing parameters"
            : "Vehicle parameters within normal range";

        return Task.FromResult(new AnomalyResult
        {
            Score = anomalyScore,
            IsAnomaly = isAnomaly,
            Confidence = 0.85 + (_random.NextDouble() * 0.1),
            Description = description,
            ContributingParameters = contributingParams
        });
    }

    private double CalculateVariance(IEnumerable<double> values)
    {
        var list = values.ToList();
        if (list.Count < 2) return 0;
        
        double mean = list.Average();
        double sumSquaredDiff = list.Sum(v => Math.Pow(v - mean, 2));
        return sumSquaredDiff / (list.Count - 1);
    }
}
