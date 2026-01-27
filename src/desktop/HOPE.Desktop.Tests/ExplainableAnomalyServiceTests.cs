using HOPE.Core.Models;
using HOPE.Core.Services.AI;
using Xunit;
using Moq;
using AnomalyDetectionResult = HOPE.Core.Services.AI.AnomalyDetectionResult;

namespace HOPE.Desktop.Tests;

public class ExplainableAnomalyServiceTests
{
    private readonly ExplainableAnomalyService _service;

    public ExplainableAnomalyServiceTests()
    {
        _service = new ExplainableAnomalyService();
    }

    [Fact]
    public async Task ExplainAnomaly_GeneratesValidExplanation()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.85, new[] { "10", "04" }); // MAF and Load
        var context = CreateTestVehicleContext();

        // Act
        var explanation = await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        Assert.NotNull(explanation);
        Assert.Equal(anomaly.Id, explanation.AnomalyId);
        Assert.NotEmpty(explanation.Narrative);
        Assert.NotNull(explanation.PrimaryDeviation);
        Assert.NotNull(explanation.GhostCurveData);
    }

    [Fact]
    public async Task ExplainAnomaly_ClassifiesSeverityCorrectly()
    {
        // Arrange & Act
        var criticalAnomaly = CreateTestAnomaly(0.96);
        var highAnomaly = CreateTestAnomaly(0.87);
        var mediumAnomaly = CreateTestAnomaly(0.72);
        var lowAnomaly = CreateTestAnomaly(0.55);
        var infoAnomaly = CreateTestAnomaly(0.35);

        var context = CreateTestVehicleContext();

        var criticalExplanation = await _service.ExplainAnomalyAsync(criticalAnomaly, context);
        var highExplanation = await _service.ExplainAnomalyAsync(highAnomaly, context);
        var mediumExplanation = await _service.ExplainAnomalyAsync(mediumAnomaly, context);
        var lowExplanation = await _service.ExplainAnomalyAsync(lowAnomaly, context);
        var infoExplanation = await _service.ExplainAnomalyAsync(infoAnomaly, context);

        // Assert
        Assert.Equal(AnomalySeverity.Critical, criticalExplanation.Severity);
        Assert.Equal(AnomalySeverity.High, highExplanation.Severity);
        Assert.Equal(AnomalySeverity.Medium, mediumExplanation.Severity);
        Assert.Equal(AnomalySeverity.Low, lowExplanation.Severity);
        Assert.Equal(AnomalySeverity.Info, infoExplanation.Severity);
    }

    [Fact]
    public async Task ExplainAnomaly_MatchesDiagnosticPatterns()
    {
        // Arrange - Create anomaly that should match vacuum leak pattern
        var anomaly = CreateTestAnomaly(0.80, new[] { "10", "04", "06" });
        var context = CreateTestVehicleContext();
        context.CurrentRPM = 800; // Idle
        context.EngineLoad = 5;

        // Act
        var explanation = await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        Assert.NotEmpty(explanation.MatchedPatterns);
    }

    [Fact]
    public async Task ExplainAnomaly_GeneratesRepairSuggestions()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.85, new[] { "10", "04" });
        var context = CreateTestVehicleContext();

        // Act
        var explanation = await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        if (explanation.MatchedPatterns.Any())
        {
            Assert.NotEmpty(explanation.RepairSuggestions);
            Assert.All(explanation.RepairSuggestions, s =>
            {
                Assert.NotEmpty(s.Title);
                Assert.NotEmpty(s.Description);
            });
        }
    }

    [Fact]
    public void GenerateGhostCurveData_CreatesValidComparison()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.75, new[] { "10" });
        anomaly.RecentReadings = new List<OBD2Reading>
        {
            new() { PID = "10", Value = 15.0, Timestamp = DateTime.UtcNow.AddSeconds(-5) },
            new() { PID = "10", Value = 16.5, Timestamp = DateTime.UtcNow.AddSeconds(-4) },
            new() { PID = "10", Value = 14.2, Timestamp = DateTime.UtcNow.AddSeconds(-3) },
            new() { PID = "10", Value = 18.0, Timestamp = DateTime.UtcNow.AddSeconds(-2) },
            new() { PID = "10", Value = 12.5, Timestamp = DateTime.UtcNow.AddSeconds(-1) }
        };

        var context = CreateTestVehicleContext();
        context.CurrentRPM = 2000;
        context.EngineLoad = 40;

        // Act
        var ghostData = _service.GenerateGhostCurveData(anomaly, context);

        // Assert
        Assert.NotNull(ghostData);
        Assert.Equal("10", ghostData.ParameterName);
        Assert.NotEmpty(ghostData.ExpectedCurve);
        Assert.NotEmpty(ghostData.ActualCurve);
        Assert.Equal(ghostData.ExpectedCurve.Count, ghostData.ActualCurve.Count);
    }

    [Fact]
    public void GenerateGhostCurveData_CalculatesDeviationStatistics()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.75, new[] { "10" });
        anomaly.RecentReadings = new List<OBD2Reading>
        {
            new() { PID = "10", Value = 20.0, Timestamp = DateTime.UtcNow.AddSeconds(-2) },
            new() { PID = "10", Value = 22.0, Timestamp = DateTime.UtcNow.AddSeconds(-1) },
            new() { PID = "10", Value = 25.0, Timestamp = DateTime.UtcNow }
        };

        var context = CreateTestVehicleContext();

        // Act
        var ghostData = _service.GenerateGhostCurveData(anomaly, context);

        // Assert
        Assert.True(ghostData.MeanDeviation >= 0);
        Assert.True(ghostData.MaxDeviation >= ghostData.MeanDeviation);
        Assert.True(ghostData.DeviationPercentage >= 0);
    }

    [Fact]
    public void GenerateGhostCurveData_IdentifiesDeviationZones()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.80, new[] { "10" });
        anomaly.RecentReadings = new List<OBD2Reading>
        {
            new() { PID = "10", Value = 15.0, Timestamp = DateTime.UtcNow.AddSeconds(-10) },
            new() { PID = "10", Value = 15.5, Timestamp = DateTime.UtcNow.AddSeconds(-8) },
            new() { PID = "10", Value = 25.0, Timestamp = DateTime.UtcNow.AddSeconds(-6) }, // Deviation
            new() { PID = "10", Value = 28.0, Timestamp = DateTime.UtcNow.AddSeconds(-4) }, // Deviation
            new() { PID = "10", Value = 15.2, Timestamp = DateTime.UtcNow.AddSeconds(-2) }  // Back to normal
        };

        var context = CreateTestVehicleContext();

        // Act
        var ghostData = _service.GenerateGhostCurveData(anomaly, context);

        // Assert
        Assert.NotEmpty(ghostData.DeviationZones);
    }

    [Fact]
    public void AnalyzeAnomalyCorrelation_GroupsAnomaliesByParameter()
    {
        // Arrange
        var anomalies = new List<AnomalyDetectionResult>
        {
            CreateTestAnomaly(0.75, new[] { "10", "04" }),
            CreateTestAnomaly(0.80, new[] { "10" }),
            CreateTestAnomaly(0.70, new[] { "04", "05" }),
            CreateTestAnomaly(0.85, new[] { "10", "05" })
        };

        // Act
        var analysis = _service.AnalyzeAnomalyCorrelation(anomalies);

        // Assert
        Assert.Equal(4, analysis.TotalAnomalies);
        Assert.NotEmpty(analysis.ParameterCorrelations);

        var mafCorrelation = analysis.ParameterCorrelations.FirstOrDefault(p => p.ParameterName == "10");
        Assert.NotNull(mafCorrelation);
        Assert.Equal(3, mafCorrelation.AnomalyCount); // MAF appears in 3 anomalies
    }

    [Fact]
    public void AnalyzeAnomalyCorrelation_CalculatesSystemHealthScore()
    {
        // Arrange
        var anomalies = new List<AnomalyDetectionResult>
        {
            CreateTestAnomaly(0.60),
            CreateTestAnomaly(0.65),
            CreateTestAnomaly(0.70)
        };

        // Act
        var analysis = _service.AnalyzeAnomalyCorrelation(anomalies);

        // Assert
        Assert.True(analysis.SystemHealthScore >= 0);
        Assert.True(analysis.SystemHealthScore <= 100);
    }

    [Fact]
    public void AnalyzeAnomalyCorrelation_IdentifiesRootCauseCandidates()
    {
        // Arrange
        var anomalies = new List<AnomalyDetectionResult>
        {
            CreateTestAnomaly(0.80, new[] { "10" }),
            CreateTestAnomaly(0.82, new[] { "10" }),
            CreateTestAnomaly(0.78, new[] { "10", "04" }),
            CreateTestAnomaly(0.85, new[] { "10" }),
            CreateTestAnomaly(0.75, new[] { "05" })
        };

        // Act
        var analysis = _service.AnalyzeAnomalyCorrelation(anomalies);

        // Assert
        Assert.NotEmpty(analysis.RootCauseCandidates);

        // MAF (PID 10) should be the top candidate
        var topCandidate = analysis.RootCauseCandidates.First();
        Assert.Equal("10", topCandidate.ParameterName);
    }

    [Fact]
    public async Task ExplainAnomaly_CalculatesConfidenceScore()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.90, new[] { "10", "04", "06" });
        var context = CreateTestVehicleContext();
        context.CurrentRPM = 800;
        context.EngineLoad = 5;

        // Act
        var explanation = await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        Assert.True(explanation.Confidence >= 0);
        Assert.True(explanation.Confidence <= 1);
    }

    [Fact]
    public async Task ExplainAnomaly_FiresExplanationGeneratedEvent()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.75);
        var context = CreateTestVehicleContext();

        AnomalyExplanation? receivedExplanation = null;
        _service.ExplanationGenerated += (sender, args) => receivedExplanation = args.Explanation;

        // Act
        await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        Assert.NotNull(receivedExplanation);
    }

    [Fact]
    public async Task ExplainAnomaly_IdentifiesDeviationDirection()
    {
        // Arrange
        var anomaly = CreateTestAnomaly(0.80, new[] { "10" });
        anomaly.RecentReadings = new List<OBD2Reading>
        {
            new() { PID = "10", Value = 50.0, Timestamp = DateTime.UtcNow } // High MAF
        };
        anomaly.ParameterContributions = new Dictionary<string, double> { ["10"] = 0.8 };

        var context = CreateTestVehicleContext();

        // Act
        var explanation = await _service.ExplainAnomalyAsync(anomaly, context);

        // Assert
        Assert.NotNull(explanation.PrimaryDeviation);
        Assert.Equal("10", explanation.PrimaryDeviation.ParameterName);
        Assert.True(explanation.PrimaryDeviation.Direction == DeviationDirection.High ||
                    explanation.PrimaryDeviation.Direction == DeviationDirection.Low);
    }

    #region Helper Methods

    private static AnomalyDetectionResult CreateTestAnomaly(
        double score,
        string[]? parameters = null)
    {
        parameters ??= new[] { "10" };

        var contributions = new Dictionary<string, double>();
        foreach (var param in parameters)
        {
            contributions[param] = 1.0 / parameters.Length;
        }

        return new AnomalyDetectionResult
        {
            Id = Guid.NewGuid(),
            Timestamp = DateTime.UtcNow,
            AnomalyScore = score,
            ContributingParameters = parameters.ToList(),
            ParameterContributions = contributions,
            RecentReadings = new List<OBD2Reading>
            {
                new() { PID = parameters[0], Value = 15.0, Timestamp = DateTime.UtcNow }
            }
        };
    }

    private static VehicleContext CreateTestVehicleContext()
    {
        return new VehicleContext
        {
            Make = "Honda",
            Model = "Civic",
            Year = 2020,
            CurrentRPM = 2500,
            EngineLoad = 45,
            VehicleSpeed = 60,
            EngineDisplacementL = 2.0,
            EngineRuntime = 600,
            AmbientTemp = 25
        };
    }

    #endregion
}
