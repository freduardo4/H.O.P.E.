namespace HOPE.Core.Models;

/// <summary>
/// Represents an AI-generated insight from diagnostic data analysis.
/// </summary>
public class AIInsight
{
    public Guid InsightId { get; set; }
    public Guid SessionId { get; set; }

    /// <summary>
    /// Type of insight
    /// </summary>
    public InsightType Type { get; set; }

    /// <summary>
    /// Severity level
    /// </summary>
    public InsightSeverity Severity { get; set; }

    /// <summary>
    /// Human-readable description of the insight
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Detailed explanation (optional)
    /// </summary>
    public string? DetailedExplanation { get; set; }

    /// <summary>
    /// AI model confidence score (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Recommended action to address the insight
    /// </summary>
    public string? RecommendedAction { get; set; }

    /// <summary>
    /// Affected components or systems
    /// </summary>
    public List<string> AffectedComponents { get; set; } = new();

    /// <summary>
    /// Related OBD2 parameters that triggered this insight
    /// </summary>
    public List<string> RelatedPIDs { get; set; } = new();

    /// <summary>
    /// When this insight was generated
    /// </summary>
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Name of the AI model that generated this insight
    /// </summary>
    public string ModelName { get; set; } = "LSTM-Autoencoder-v1";

    /// <summary>
    /// Model version
    /// </summary>
    public string ModelVersion { get; set; } = "1.0.0";

    /// <summary>
    /// Additional metadata for the insight
    /// </summary>
    public Dictionary<string, object> Metadata { get; set; } = new();

    /// <summary>
    /// Whether this insight has been acknowledged by a technician
    /// </summary>
    public bool IsAcknowledged { get; set; }

    /// <summary>
    /// Technician notes on this insight
    /// </summary>
    public string? TechnicianNotes { get; set; }

    /// <summary>
    /// Navigation property to diagnostic session
    /// </summary>
    public DiagnosticSession? Session { get; set; }

    public override string ToString() =>
        $"[{Severity}] {Type}: {Description} (Confidence: {Confidence:P0})";
}

/// <summary>
/// Types of AI-generated insights
/// </summary>
public enum InsightType
{
    AnomalyDetected,
    MaintenancePrediction,
    PerformanceIssue,
    SensorDrift,
    FuelSystemIssue,
    IgnitionIssue,
    BoostLeak,
    ExhaustRestriction,
    KnockDetected,
    LambdaSensorFailing,
    CatalystEfficiency,
    TurbochargerIssue,
    InjectorProblem,
    MAFSensorDrift,
    ThrottlePositionAnomal,
    CoolantSystemIssue,
    OilPressureWarning,
    TransmissionIssue,
    CustomPattern
}

/// <summary>
/// Severity levels for insights
/// </summary>
public enum InsightSeverity
{
    Info,
    Low,
    Medium,
    High,
    Critical
}

/// <summary>
/// Result from AI anomaly detection
/// </summary>
public class AnomalyDetectionResult
{
    /// <summary>
    /// Anomaly score (0.0 = normal, 1.0 = highly abnormal)
    /// </summary>
    public double AnomalyScore { get; set; }

    /// <summary>
    /// Whether the score exceeds the anomaly threshold
    /// </summary>
    public bool IsAnomaly { get; set; }

    /// <summary>
    /// Reconstruction error from autoencoder
    /// </summary>
    public double ReconstructionError { get; set; }

    /// <summary>
    /// Input data window used for prediction
    /// </summary>
    public List<OBD2Reading> InputData { get; set; } = new();

    /// <summary>
    /// Timestamp when detection was performed
    /// </summary>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Parameters that contributed most to the anomaly
    /// </summary>
    public Dictionary<string, double> ParameterContributions { get; set; } = new();

    /// <summary>
    /// Model inference time in milliseconds
    /// </summary>
    public double InferenceTimeMs { get; set; }
}

/// <summary>
/// Configuration for AI anomaly detection
/// </summary>
public class AnomalyDetectionConfig
{
    /// <summary>
    /// Threshold above which a score is considered anomalous (0.0-1.0)
    /// </summary>
    public double AnomalyThreshold { get; set; } = 0.75;

    /// <summary>
    /// Sliding window size (number of time steps)
    /// </summary>
    public int WindowSize { get; set; } = 60;

    /// <summary>
    /// Stride for sliding window (1 = every reading, 5 = every 5th reading)
    /// </summary>
    public int WindowStride { get; set; } = 5;

    /// <summary>
    /// Required PIDs for anomaly detection
    /// </summary>
    public List<string> RequiredPIDs { get; set; } = new()
    {
        OBD2PIDs.EngineRPM,
        OBD2PIDs.EngineLoad,
        OBD2PIDs.CoolantTemp,
        OBD2PIDs.MAFSensor,
        OBD2PIDs.O2Sensor1Voltage,
        OBD2PIDs.ShortTermFuelTrim,
        OBD2PIDs.LongTermFuelTrim,
        OBD2PIDs.ThrottlePosition,
        OBD2PIDs.IntakeAirTemp,
        OBD2PIDs.FuelPressure
    };

    /// <summary>
    /// Minimum confidence to generate an insight
    /// </summary>
    public double MinConfidenceForInsight { get; set; } = 0.70;

    /// <summary>
    /// Enable real-time anomaly detection during sessions
    /// </summary>
    public bool EnableRealTimeDetection { get; set; } = true;

    /// <summary>
    /// Path to ONNX model file
    /// </summary>
    public string ModelPath { get; set; } = "models/anomaly_detector.onnx";
}
