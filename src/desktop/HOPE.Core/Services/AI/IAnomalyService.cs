using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Interface for AI-based anomaly detection service
/// </summary>
public interface IAnomalyService
{
    /// <summary>
    /// Analyze a sequence of readings and return anomaly score
    /// </summary>
    /// <param name="readings">Recent OBD2 readings (typically 60 seconds worth)</param>
    /// <returns>Anomaly score between 0.0 (normal) and 1.0 (highly anomalous)</returns>
    Task<AnomalyResult> AnalyzeAsync(IEnumerable<OBD2Reading> readings);

    /// <summary>
    /// Check if the model is loaded and ready
    /// </summary>
    bool IsModelLoaded { get; }

    /// <summary>
    /// Load the ONNX model for inference
    /// </summary>
    Task LoadModelAsync(string modelPath);
}

/// <summary>
/// Result of anomaly detection analysis
/// </summary>
public record AnomalyResult
{
    /// <summary>
    /// Overall anomaly score (0.0 = normal, 1.0 = highly anomalous)
    /// </summary>
    public double Score { get; init; }

    /// <summary>
    /// Whether the score exceeds the anomaly threshold
    /// </summary>
    public bool IsAnomaly { get; init; }

    /// <summary>
    /// Confidence level of the prediction (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; init; }

    /// <summary>
    /// Human-readable explanation
    /// </summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>
    /// Parameters that contributed most to the anomaly (if any)
    /// </summary>
    public List<string> ContributingParameters { get; init; } = new();

    /// <summary>
    /// Timestamp of analysis
    /// </summary>
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;
}
