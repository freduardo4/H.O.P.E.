using System.Text;
using System.Text.Json;
using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Explainable AI (XAI) service that provides human-readable diagnostic narratives
/// for detected anomalies, including ghost curve visualization data and repair suggestions.
/// </summary>
public class ExplainableAnomalyService
{
    private readonly Dictionary<string, ParameterProfile> _parameterProfiles;
    private readonly List<DiagnosticPattern> _knownPatterns;
    private readonly ILLMService? _llmService;

    /// <summary>
    /// Event raised when a new explanation is generated
    /// </summary>
    public event EventHandler<ExplanationGeneratedEventArgs>? ExplanationGenerated;

    public ExplainableAnomalyService(ILLMService? llmService = null)
    {
        _llmService = llmService;
        _parameterProfiles = InitializeParameterProfiles();
        _knownPatterns = InitializeKnownPatterns();
    }

    /// <summary>
    /// Analyze an anomaly and generate a human-readable explanation
    /// </summary>
    public async Task<AnomalyExplanation> ExplainAnomalyAsync(
        AnomalyDetectionResult anomaly,
        VehicleContext context,
        CancellationToken ct = default)
    {
        var explanation = new AnomalyExplanation
        {
            AnomalyId = anomaly.Id,
            Timestamp = anomaly.Timestamp,
            Severity = ClassifySeverity(anomaly),
            AffectedParameters = anomaly.ContributingParameters.ToList()
        };

        // 1. Identify the primary deviation
        var primaryDeviation = IdentifyPrimaryDeviation(anomaly);
        explanation.PrimaryDeviation = primaryDeviation;

        // 2. Calculate expected vs actual comparison (ghost curves)
        explanation.GhostCurveData = GenerateGhostCurveData(anomaly, context);

        // 3. Match against known diagnostic patterns
        var matchedPatterns = MatchDiagnosticPatterns(anomaly, context);
        explanation.MatchedPatterns = matchedPatterns;

        // 4. Generate the narrative
        explanation.Narrative = GenerateNarrative(anomaly, primaryDeviation, matchedPatterns, context);

        // 5. Generate repair suggestions
        explanation.RepairSuggestions = GenerateRepairSuggestions(matchedPatterns);

        // 6. Calculate confidence score
        explanation.Confidence = CalculateConfidence(anomaly, matchedPatterns);

        // 7. Optionally enhance with LLM
        if (_llmService != null && anomaly.AnomalyScore > 0.9)
        {
            explanation.EnhancedNarrative = await GenerateLLMEnhancedNarrativeAsync(
                explanation, context, ct);
        }

        ExplanationGenerated?.Invoke(this, new ExplanationGeneratedEventArgs(explanation));

        return explanation;
    }

    /// <summary>
    /// Generate ghost curve comparison data
    /// </summary>
    public GhostCurveData GenerateGhostCurveData(
        AnomalyDetectionResult anomaly,
        VehicleContext context)
    {
        var data = new GhostCurveData
        {
            ParameterName = anomaly.ContributingParameters.FirstOrDefault() ?? "Unknown",
            TimeWindow = TimeSpan.FromSeconds(30)
        };

        // Calculate expected values based on vehicle state
        var expectedValues = new List<GhostCurvePoint>();
        var actualValues = new List<GhostCurvePoint>();

        foreach (var reading in anomaly.RecentReadings)
        {
            var expectedValue = CalculateExpectedValue(
                reading.PID,
                context,
                reading.Timestamp);

            expectedValues.Add(new GhostCurvePoint
            {
                Timestamp = reading.Timestamp,
                Value = expectedValue,
                Type = CurveType.Expected
            });

            actualValues.Add(new GhostCurvePoint
            {
                Timestamp = reading.Timestamp,
                Value = reading.Value,
                Type = CurveType.Actual
            });
        }

        data.ExpectedCurve = expectedValues;
        data.ActualCurve = actualValues;

        // Calculate deviation zone
        data.DeviationZones = CalculateDeviationZones(expectedValues, actualValues);

        // Calculate statistics
        if (expectedValues.Count > 0 && actualValues.Count > 0)
        {
            var deviations = expectedValues.Zip(actualValues,
                (e, a) => Math.Abs(e.Value - a.Value)).ToList();

            data.MeanDeviation = deviations.Average();
            data.MaxDeviation = deviations.Max();
            data.DeviationPercentage = data.MeanDeviation /
                Math.Max(expectedValues.Average(v => v.Value), 0.001) * 100;
        }

        return data;
    }

    /// <summary>
    /// Batch analyze multiple anomalies for pattern correlation
    /// </summary>
    public CorrelationAnalysis AnalyzeAnomalyCorrelation(
        IEnumerable<AnomalyDetectionResult> anomalies)
    {
        var anomalyList = anomalies.ToList();
        var analysis = new CorrelationAnalysis
        {
            TotalAnomalies = anomalyList.Count,
            TimeRange = anomalyList.Count > 0
                ? (anomalyList.Max(a => a.Timestamp) - anomalyList.Min(a => a.Timestamp))
                : TimeSpan.Zero
        };

        // Group by affected parameters
        var parameterGroups = anomalyList
            .SelectMany(a => a.ContributingParameters.Select(p => new { Parameter = p, Anomaly = a }))
            .GroupBy(x => x.Parameter)
            .ToDictionary(g => g.Key, g => g.Select(x => x.Anomaly).ToList());

        analysis.ParameterCorrelations = new List<ParameterCorrelation>();

        foreach (var (param, relatedAnomalies) in parameterGroups)
        {
            analysis.ParameterCorrelations.Add(new ParameterCorrelation
            {
                ParameterName = param,
                AnomalyCount = relatedAnomalies.Count,
                AverageScore = relatedAnomalies.Average(a => a.AnomalyScore),
                Frequency = relatedAnomalies.Count / Math.Max(analysis.TimeRange.TotalMinutes, 1)
            });
        }

        // Identify root cause candidates
        analysis.RootCauseCandidates = IdentifyRootCauses(anomalyList, parameterGroups);

        // Calculate system health score
        analysis.SystemHealthScore = CalculateSystemHealth(anomalyList);

        return analysis;
    }

    #region Private Methods

    private DeviationInfo IdentifyPrimaryDeviation(AnomalyDetectionResult anomaly)
    {
        var mostSignificant = anomaly.ParameterContributions
            .OrderByDescending(kv => kv.Value)
            .FirstOrDefault();

        var parameterName = mostSignificant.Key ?? "Unknown";
        var contribution = mostSignificant.Value;

        _parameterProfiles.TryGetValue(parameterName, out var profile);

        return new DeviationInfo
        {
            ParameterName = parameterName,
            ContributionScore = contribution,
            Direction = anomaly.RecentReadings.LastOrDefault()?.Value >
                        (profile?.NominalValue ?? 0) ? DeviationDirection.High : DeviationDirection.Low,
            ExpectedRange = profile != null
                ? $"{profile.MinNormal:F1} - {profile.MaxNormal:F1} {profile.Unit}"
                : "Unknown",
            ActualValue = anomaly.RecentReadings.LastOrDefault()?.Value ?? 0
        };
    }

    private List<PatternMatch> MatchDiagnosticPatterns(
        AnomalyDetectionResult anomaly,
        VehicleContext context)
    {
        var matches = new List<PatternMatch>();

        foreach (var pattern in _knownPatterns)
        {
            var score = CalculatePatternMatchScore(pattern, anomaly, context);
            if (score > 0.5)
            {
                matches.Add(new PatternMatch
                {
                    Pattern = pattern,
                    MatchScore = score,
                    MatchedConditions = GetMatchedConditions(pattern, anomaly)
                });
            }
        }

        return matches.OrderByDescending(m => m.MatchScore).Take(3).ToList();
    }

    private double CalculatePatternMatchScore(
        DiagnosticPattern pattern,
        AnomalyDetectionResult anomaly,
        VehicleContext context)
    {
        double score = 0;
        int conditions = 0;

        // Check parameter involvement
        foreach (var requiredParam in pattern.InvolvedParameters)
        {
            conditions++;
            if (anomaly.ContributingParameters.Contains(requiredParam))
            {
                score += 1;
            }
        }

        // Check driving conditions
        if (pattern.TriggerConditions.ContainsKey("RPM_Range"))
        {
            conditions++;
            var rpmRange = pattern.TriggerConditions["RPM_Range"].Split('-');
            if (rpmRange.Length == 2 &&
                double.TryParse(rpmRange[0], out var minRpm) &&
                double.TryParse(rpmRange[1], out var maxRpm))
            {
                if (context.CurrentRPM >= minRpm && context.CurrentRPM <= maxRpm)
                    score += 1;
            }
        }

        // Check load conditions
        if (pattern.TriggerConditions.ContainsKey("Load_Threshold"))
        {
            conditions++;
            if (double.TryParse(pattern.TriggerConditions["Load_Threshold"], out var loadThreshold))
            {
                if (context.EngineLoad >= loadThreshold)
                    score += 1;
            }
        }

        return conditions > 0 ? score / conditions : 0;
    }

    private List<string> GetMatchedConditions(
        DiagnosticPattern pattern,
        AnomalyDetectionResult anomaly)
    {
        var matched = new List<string>();

        foreach (var param in pattern.InvolvedParameters)
        {
            if (anomaly.ContributingParameters.Contains(param))
            {
                matched.Add($"{param} deviation detected");
            }
        }

        return matched;
    }

    private string GenerateNarrative(
        AnomalyDetectionResult anomaly,
        DeviationInfo deviation,
        List<PatternMatch> patterns,
        VehicleContext context)
    {
        var sb = new StringBuilder();

        // Opening statement about the anomaly
        sb.Append($"{deviation.ParameterName} reading ");
        sb.Append(deviation.Direction == DeviationDirection.High ? "above" : "below");
        sb.Append($" expected for current {GetDrivingConditionDescription(context)}. ");

        // Add specific deviation info
        sb.Append($"Current value: {deviation.ActualValue:F1}, ");
        sb.Append($"Expected range: {deviation.ExpectedRange}. ");

        // Add pattern-based diagnosis
        if (patterns.Count > 0)
        {
            var topPattern = patterns[0];
            sb.AppendLine();
            sb.AppendLine();
            sb.Append($"This pattern typically indicates: {topPattern.Pattern.Description}");

            if (topPattern.MatchScore > 0.8)
            {
                sb.Append($" (High confidence: {topPattern.MatchScore * 100:F0}%)");
            }
        }

        // Add possible causes
        if (patterns.Count > 0)
        {
            sb.AppendLine();
            sb.AppendLine();
            sb.Append("Possible causes: ");
            sb.Append(string.Join(", ", patterns[0].Pattern.PossibleCauses.Take(3)));
            sb.Append(".");
        }

        return sb.ToString();
    }

    private List<RepairSuggestion> GenerateRepairSuggestions(List<PatternMatch> patterns)
    {
        var suggestions = new List<RepairSuggestion>();

        foreach (var match in patterns)
        {
            foreach (var repair in match.Pattern.RepairProcedures)
            {
                suggestions.Add(new RepairSuggestion
                {
                    Title = repair.Title,
                    Description = repair.Description,
                    EstimatedDifficulty = repair.Difficulty,
                    EstimatedCost = repair.EstimatedCost,
                    PartsRequired = repair.PartsRequired,
                    Confidence = match.MatchScore
                });
            }
        }

        return suggestions.OrderByDescending(s => s.Confidence).Distinct().ToList();
    }

    private async Task<string> GenerateLLMEnhancedNarrativeAsync(
        AnomalyExplanation explanation,
        VehicleContext context,
        CancellationToken ct)
    {
        if (_llmService == null) return explanation.Narrative;

        var prompt = $@"You are an expert automotive diagnostic technician. Analyze this vehicle diagnostic data and provide a clear, professional explanation suitable for both technicians and vehicle owners.

Vehicle: {context.Year} {context.Make} {context.Model}
Current Conditions: RPM {context.CurrentRPM:F0}, Load {context.EngineLoad:F0}%, Speed {context.VehicleSpeed:F0} km/h

Anomaly Details:
- Primary Issue: {explanation.PrimaryDeviation.ParameterName}
- Current Value: {explanation.PrimaryDeviation.ActualValue:F2}
- Expected Range: {explanation.PrimaryDeviation.ExpectedRange}
- Deviation Direction: {explanation.PrimaryDeviation.Direction}
- Anomaly Score: {explanation.Confidence:P0}

Matched Diagnostic Patterns:
{string.Join("\n", explanation.MatchedPatterns.Select(p => $"- {p.Pattern.Name} ({p.MatchScore:P0} match)"))}

Please provide:
1. A clear explanation of what this anomaly means
2. Potential root causes ranked by likelihood
3. Immediate safety concerns (if any)
4. Recommended diagnostic steps";

        try
        {
            return await _llmService.GenerateAsync(prompt, ct);
        }
        catch
        {
            return explanation.Narrative; // Fall back to rule-based narrative
        }
    }

    private double CalculateExpectedValue(string pid, VehicleContext context, DateTime timestamp)
    {
        // Calculate expected sensor values based on vehicle state
        return pid switch
        {
            "10" => CalculateExpectedMAF(context), // MAF
            "0C" => context.CurrentRPM, // RPM should match
            "04" => context.EngineLoad, // Load should match
            "05" => CalculateExpectedCoolantTemp(context), // Coolant temp
            "0F" => CalculateExpectedIAT(context), // IAT
            "11" => CalculateExpectedThrottle(context), // Throttle
            _ => 0
        };
    }

    private double CalculateExpectedMAF(VehicleContext context)
    {
        // Simplified MAF calculation based on VE table approximation
        // MAF (g/s) = (RPM * Displacement * VE * Air Density) / (120 * 1000)
        var displacement = context.EngineDisplacementL;
        var ve = context.EngineLoad / 100.0; // Approximate VE from load
        var airDensity = 1.225; // kg/m³ at sea level

        return (context.CurrentRPM * displacement * ve * airDensity) / 120.0;
    }

    private double CalculateExpectedCoolantTemp(VehicleContext context)
    {
        // Expected coolant temp based on operating conditions
        if (context.EngineRuntime < 300) // Less than 5 minutes
            return 40 + (context.EngineRuntime / 300.0 * 50); // Warming up

        return 90; // Normal operating temp
    }

    private double CalculateExpectedIAT(VehicleContext context)
    {
        // IAT increases with load due to heat soak
        var baseTemp = context.AmbientTemp;
        var heatSoak = context.EngineLoad * 0.3;
        return baseTemp + heatSoak;
    }

    private double CalculateExpectedThrottle(VehicleContext context)
    {
        // Throttle position should roughly correlate with load
        return context.EngineLoad * 0.9;
    }

    private List<DeviationZone> CalculateDeviationZones(
        List<GhostCurvePoint> expected,
        List<GhostCurvePoint> actual)
    {
        var zones = new List<DeviationZone>();
        DeviationZone? currentZone = null;

        for (int i = 0; i < Math.Min(expected.Count, actual.Count); i++)
        {
            var deviation = Math.Abs(expected[i].Value - actual[i].Value);
            var deviationPct = deviation / Math.Max(expected[i].Value, 0.001) * 100;

            var severity = deviationPct switch
            {
                > 25 => ZoneSeverity.Critical,
                > 15 => ZoneSeverity.Warning,
                > 5 => ZoneSeverity.Minor,
                _ => ZoneSeverity.Normal
            };

            if (currentZone == null || currentZone.Severity != severity)
            {
                if (currentZone != null)
                {
                    currentZone.EndTime = expected[i].Timestamp;
                    zones.Add(currentZone);
                }

                currentZone = new DeviationZone
                {
                    StartTime = expected[i].Timestamp,
                    Severity = severity,
                    AverageDeviation = deviation
                };
            }
            else
            {
                currentZone.AverageDeviation =
                    (currentZone.AverageDeviation + deviation) / 2;
            }
        }

        if (currentZone != null)
        {
            currentZone.EndTime = expected.LastOrDefault()?.Timestamp ?? DateTime.UtcNow;
            zones.Add(currentZone);
        }

        return zones;
    }

    private AnomalySeverity ClassifySeverity(AnomalyDetectionResult anomaly)
    {
        return anomaly.AnomalyScore switch
        {
            >= 0.95 => AnomalySeverity.Critical,
            >= 0.85 => AnomalySeverity.High,
            >= 0.70 => AnomalySeverity.Medium,
            >= 0.50 => AnomalySeverity.Low,
            _ => AnomalySeverity.Info
        };
    }

    private double CalculateConfidence(
        AnomalyDetectionResult anomaly,
        List<PatternMatch> patterns)
    {
        var baseConfidence = anomaly.AnomalyScore;
        var patternBoost = patterns.Count > 0
            ? patterns.Max(p => p.MatchScore) * 0.2
            : 0;

        return Math.Min(baseConfidence + patternBoost, 1.0);
    }

    private List<RootCauseCandidate> IdentifyRootCauses(
        List<AnomalyDetectionResult> anomalies,
        Dictionary<string, List<AnomalyDetectionResult>> parameterGroups)
    {
        var candidates = new List<RootCauseCandidate>();

        foreach (var (param, related) in parameterGroups.OrderByDescending(g => g.Value.Count))
        {
            if (_parameterProfiles.TryGetValue(param, out var profile))
            {
                candidates.Add(new RootCauseCandidate
                {
                    ComponentName = profile.ComponentName,
                    ParameterName = param,
                    AnomalyCount = related.Count,
                    Probability = Math.Min(related.Count / 10.0, 1.0),
                    RelatedSystems = profile.RelatedSystems
                });
            }
        }

        return candidates.Take(5).ToList();
    }

    private double CalculateSystemHealth(List<AnomalyDetectionResult> anomalies)
    {
        if (anomalies.Count == 0) return 100;

        var avgScore = anomalies.Average(a => a.AnomalyScore);
        var frequency = anomalies.Count / 60.0; // per minute

        var health = 100 - (avgScore * 50) - (frequency * 10);
        return Math.Max(0, Math.Min(100, health));
    }

    private string GetDrivingConditionDescription(VehicleContext context)
    {
        if (context.VehicleSpeed < 5)
            return $"idle conditions (RPM: {context.CurrentRPM:F0})";
        if (context.EngineLoad > 80)
            return $"high load conditions (Load: {context.EngineLoad:F0}%, RPM: {context.CurrentRPM:F0})";
        if (context.VehicleSpeed > 100)
            return $"highway cruise (Speed: {context.VehicleSpeed:F0} km/h)";

        return $"driving conditions (RPM: {context.CurrentRPM:F0}, Load: {context.EngineLoad:F0}%)";
    }

    private Dictionary<string, ParameterProfile> InitializeParameterProfiles()
    {
        return new Dictionary<string, ParameterProfile>
        {
            ["10"] = new ParameterProfile // MAF
            {
                ParameterName = "MAF Air Flow",
                ComponentName = "Mass Air Flow Sensor",
                Unit = "g/s",
                NominalValue = 15,
                MinNormal = 2,
                MaxNormal = 150,
                RelatedSystems = new[] { "Fuel System", "Air Intake" }
            },
            ["0C"] = new ParameterProfile // RPM
            {
                ParameterName = "Engine RPM",
                ComponentName = "Crankshaft Position Sensor",
                Unit = "RPM",
                NominalValue = 800,
                MinNormal = 600,
                MaxNormal = 6500,
                RelatedSystems = new[] { "Ignition System", "Fuel System" }
            },
            ["05"] = new ParameterProfile // Coolant Temp
            {
                ParameterName = "Engine Coolant Temperature",
                ComponentName = "Coolant Temperature Sensor",
                Unit = "°C",
                NominalValue = 90,
                MinNormal = -40,
                MaxNormal = 120,
                RelatedSystems = new[] { "Cooling System", "Fuel System" }
            },
            ["11"] = new ParameterProfile // Throttle Position
            {
                ParameterName = "Throttle Position",
                ComponentName = "Throttle Position Sensor",
                Unit = "%",
                NominalValue = 15,
                MinNormal = 0,
                MaxNormal = 100,
                RelatedSystems = new[] { "Electronic Throttle Control", "Fuel System" }
            },
            ["04"] = new ParameterProfile // Engine Load
            {
                ParameterName = "Calculated Engine Load",
                ComponentName = "ECU Calculation",
                Unit = "%",
                NominalValue = 25,
                MinNormal = 0,
                MaxNormal = 100,
                RelatedSystems = new[] { "Air Intake", "Fuel System" }
            }
        };
    }

    private List<DiagnosticPattern> InitializeKnownPatterns()
    {
        return new List<DiagnosticPattern>
        {
            new DiagnosticPattern
            {
                Name = "Vacuum Leak",
                Description = "Air entering the intake manifold through an unmetered source",
                InvolvedParameters = new[] { "10", "04", "06" },
                TriggerConditions = new Dictionary<string, string>
                {
                    ["RPM_Range"] = "600-1500",
                    ["Load_Threshold"] = "0"
                },
                PossibleCauses = new[]
                {
                    "Cracked or disconnected vacuum hose",
                    "Leaking intake manifold gasket",
                    "Faulty PCV valve",
                    "Leaking brake booster hose"
                },
                RepairProcedures = new[]
                {
                    new RepairProcedure
                    {
                        Title = "Smoke Test Intake System",
                        Description = "Use smoke machine to identify vacuum leak location",
                        Difficulty = RepairDifficulty.Easy,
                        EstimatedCost = "$50-100"
                    },
                    new RepairProcedure
                    {
                        Title = "Replace Intake Manifold Gasket",
                        Description = "Remove intake manifold and replace gasket",
                        Difficulty = RepairDifficulty.Moderate,
                        EstimatedCost = "$150-400",
                        PartsRequired = new[] { "Intake manifold gasket set" }
                    }
                }
            },
            new DiagnosticPattern
            {
                Name = "Dirty MAF Sensor",
                Description = "Mass Air Flow sensor contaminated, reading lower than actual airflow",
                InvolvedParameters = new[] { "10", "04" },
                TriggerConditions = new Dictionary<string, string>
                {
                    ["RPM_Range"] = "2000-4000"
                },
                PossibleCauses = new[]
                {
                    "Oil contamination from aftermarket air filter",
                    "Dust buildup on hot wire element",
                    "Failed MAF sensor"
                },
                RepairProcedures = new[]
                {
                    new RepairProcedure
                    {
                        Title = "Clean MAF Sensor",
                        Description = "Clean MAF sensor element with MAF cleaner spray",
                        Difficulty = RepairDifficulty.Easy,
                        EstimatedCost = "$10-20",
                        PartsRequired = new[] { "MAF sensor cleaner" }
                    },
                    new RepairProcedure
                    {
                        Title = "Replace MAF Sensor",
                        Description = "Replace mass air flow sensor if cleaning doesn't help",
                        Difficulty = RepairDifficulty.Easy,
                        EstimatedCost = "$100-300",
                        PartsRequired = new[] { "MAF sensor" }
                    }
                }
            },
            new DiagnosticPattern
            {
                Name = "Catalytic Converter Efficiency",
                Description = "Catalytic converter not operating at optimal efficiency",
                InvolvedParameters = new[] { "14", "15" },
                TriggerConditions = new Dictionary<string, string>
                {
                    ["Load_Threshold"] = "30"
                },
                PossibleCauses = new[]
                {
                    "Aged catalytic converter",
                    "Contaminated catalyst (coolant, oil)",
                    "Upstream O2 sensor malfunction",
                    "Engine running rich"
                },
                RepairProcedures = new[]
                {
                    new RepairProcedure
                    {
                        Title = "Diagnose Root Cause",
                        Description = "Check for coolant/oil consumption, test O2 sensors",
                        Difficulty = RepairDifficulty.Moderate,
                        EstimatedCost = "$100-200"
                    },
                    new RepairProcedure
                    {
                        Title = "Replace Catalytic Converter",
                        Description = "Replace catalytic converter if efficiency is confirmed low",
                        Difficulty = RepairDifficulty.Moderate,
                        EstimatedCost = "$500-2500",
                        PartsRequired = new[] { "Catalytic converter", "Gaskets" }
                    }
                }
            }
        };
    }

    #endregion
}

#region Interfaces

public interface ILLMService
{
    Task<string> GenerateAsync(string prompt, CancellationToken ct = default);
}

#endregion

#region Data Models

public class AnomalyExplanation
{
    public Guid AnomalyId { get; set; }
    public DateTime Timestamp { get; set; }
    public AnomalySeverity Severity { get; set; }
    public List<string> AffectedParameters { get; set; } = new();
    public DeviationInfo PrimaryDeviation { get; set; } = null!;
    public GhostCurveData GhostCurveData { get; set; } = null!;
    public List<PatternMatch> MatchedPatterns { get; set; } = new();
    public string Narrative { get; set; } = string.Empty;
    public string? EnhancedNarrative { get; set; }
    public List<RepairSuggestion> RepairSuggestions { get; set; } = new();
    public double Confidence { get; set; }
}

public class DeviationInfo
{
    public string ParameterName { get; set; } = string.Empty;
    public double ContributionScore { get; set; }
    public DeviationDirection Direction { get; set; }
    public string ExpectedRange { get; set; } = string.Empty;
    public double ActualValue { get; set; }
}

public enum DeviationDirection
{
    Low,
    High
}

public enum AnomalySeverity
{
    Info,
    Low,
    Medium,
    High,
    Critical
}

public class GhostCurveData
{
    public string ParameterName { get; set; } = string.Empty;
    public TimeSpan TimeWindow { get; set; }
    public List<GhostCurvePoint> ExpectedCurve { get; set; } = new();
    public List<GhostCurvePoint> ActualCurve { get; set; } = new();
    public List<DeviationZone> DeviationZones { get; set; } = new();
    public double MeanDeviation { get; set; }
    public double MaxDeviation { get; set; }
    public double DeviationPercentage { get; set; }
}

public class GhostCurvePoint
{
    public DateTime Timestamp { get; set; }
    public double Value { get; set; }
    public CurveType Type { get; set; }
}

public enum CurveType
{
    Expected,
    Actual
}

public class DeviationZone
{
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public ZoneSeverity Severity { get; set; }
    public double AverageDeviation { get; set; }
}

public enum ZoneSeverity
{
    Normal,
    Minor,
    Warning,
    Critical
}

public class PatternMatch
{
    public DiagnosticPattern Pattern { get; set; } = null!;
    public double MatchScore { get; set; }
    public List<string> MatchedConditions { get; set; } = new();
}

public class DiagnosticPattern
{
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string[] InvolvedParameters { get; set; } = Array.Empty<string>();
    public Dictionary<string, string> TriggerConditions { get; set; } = new();
    public string[] PossibleCauses { get; set; } = Array.Empty<string>();
    public RepairProcedure[] RepairProcedures { get; set; } = Array.Empty<RepairProcedure>();
}

public class RepairProcedure
{
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public RepairDifficulty Difficulty { get; set; }
    public string EstimatedCost { get; set; } = string.Empty;
    public string[] PartsRequired { get; set; } = Array.Empty<string>();
}

public enum RepairDifficulty
{
    Easy,
    Moderate,
    Difficult,
    Professional
}

public class RepairSuggestion
{
    public string Title { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public RepairDifficulty EstimatedDifficulty { get; set; }
    public string EstimatedCost { get; set; } = string.Empty;
    public string[] PartsRequired { get; set; } = Array.Empty<string>();
    public double Confidence { get; set; }
}

public class ParameterProfile
{
    public string ParameterName { get; set; } = string.Empty;
    public string ComponentName { get; set; } = string.Empty;
    public string Unit { get; set; } = string.Empty;
    public double NominalValue { get; set; }
    public double MinNormal { get; set; }
    public double MaxNormal { get; set; }
    public string[] RelatedSystems { get; set; } = Array.Empty<string>();
}

public class VehicleContext
{
    public string Make { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public int Year { get; set; }
    public double CurrentRPM { get; set; }
    public double EngineLoad { get; set; }
    public double VehicleSpeed { get; set; }
    public double EngineDisplacementL { get; set; }
    public double EngineRuntime { get; set; }
    public double AmbientTemp { get; set; }
}

public class AnomalyDetectionResult
{
    public Guid Id { get; set; }
    public DateTime Timestamp { get; set; }
    public double AnomalyScore { get; set; }
    public List<string> ContributingParameters { get; set; } = new();
    public Dictionary<string, double> ParameterContributions { get; set; } = new();
    public List<OBD2Reading> RecentReadings { get; set; } = new();
}

public class CorrelationAnalysis
{
    public int TotalAnomalies { get; set; }
    public TimeSpan TimeRange { get; set; }
    public List<ParameterCorrelation> ParameterCorrelations { get; set; } = new();
    public List<RootCauseCandidate> RootCauseCandidates { get; set; } = new();
    public double SystemHealthScore { get; set; }
}

public class ParameterCorrelation
{
    public string ParameterName { get; set; } = string.Empty;
    public int AnomalyCount { get; set; }
    public double AverageScore { get; set; }
    public double Frequency { get; set; }
}

public class RootCauseCandidate
{
    public string ComponentName { get; set; } = string.Empty;
    public string ParameterName { get; set; } = string.Empty;
    public int AnomalyCount { get; set; }
    public double Probability { get; set; }
    public string[] RelatedSystems { get; set; } = Array.Empty<string>();
}

public class ExplanationGeneratedEventArgs : EventArgs
{
    public AnomalyExplanation Explanation { get; }

    public ExplanationGeneratedEventArgs(AnomalyExplanation explanation)
    {
        Explanation = explanation;
    }
}

#endregion
