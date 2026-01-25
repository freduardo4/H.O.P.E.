namespace HOPE.Core.Models;

/// <summary>
/// Represents a diagnostic or tuning session with a vehicle.
/// </summary>
public class DiagnosticSession
{
    public Guid SessionId { get; set; }
    public Guid VehicleId { get; set; }
    public Guid TechnicianId { get; set; }

    public SessionType SessionType { get; set; } = SessionType.Diagnostic;
    public SessionStatus Status { get; set; } = SessionStatus.InProgress;

    public DateTime StartedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }

    /// <summary>
    /// Baseline snapshot of vehicle state at session start
    /// </summary>
    public SessionSnapshot? BaselineSnapshot { get; set; }

    /// <summary>
    /// Technician notes and observations
    /// </summary>
    public string? Notes { get; set; }

    /// <summary>
    /// Diagnostic Trouble Codes (DTCs) found during session
    /// </summary>
    public List<DiagnosticTroubleCode> DTCs { get; set; } = new();

    /// <summary>
    /// OBD2 data points collected during session
    /// </summary>
    public List<OBD2Reading> OBD2Data { get; set; } = new();

    /// <summary>
    /// AI-generated insights for this session
    /// </summary>
    public List<AIInsight> AIInsights { get; set; } = new();

    /// <summary>
    /// ECU calibration uploaded/modified during session
    /// </summary>
    public Guid? ECUCalibrationId { get; set; }

    /// <summary>
    /// Navigation property to vehicle
    /// </summary>
    public Vehicle? Vehicle { get; set; }

    /// <summary>
    /// Navigation property to ECU calibration
    /// </summary>
    public ECUCalibration? ECUCalibration { get; set; }

    /// <summary>
    /// Duration of the session
    /// </summary>
    public TimeSpan? Duration => CompletedAt.HasValue
        ? CompletedAt.Value - StartedAt
        : DateTime.UtcNow - StartedAt;

    public override string ToString() =>
        $"Session {SessionId:N} - {SessionType} ({Status}) - Started: {StartedAt:g}";
}

/// <summary>
/// Type of diagnostic session
/// </summary>
public enum SessionType
{
    Diagnostic,
    Tuning,
    Custom,
    PrePurchaseInspection,
    Dyno
}

/// <summary>
/// Status of diagnostic session
/// </summary>
public enum SessionStatus
{
    InProgress,
    Completed,
    Failed,
    Cancelled
}

/// <summary>
/// Snapshot of vehicle state at a point in time
/// </summary>
public class SessionSnapshot
{
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public int Odometer { get; set; }
    public Dictionary<string, double> SensorReadings { get; set; } = new();
    public List<string> ActiveDTCs { get; set; } = new();
}

/// <summary>
/// Diagnostic Trouble Code (DTC) with metadata
/// </summary>
public class DiagnosticTroubleCode
{
    public string Code { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public DTCType Type { get; set; }
    public DTCSeverity Severity { get; set; }
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
    public bool IsIntermittent { get; set; }
}

public enum DTCType
{
    Powertrain,  // P-codes
    Chassis,     // C-codes
    Body,        // B-codes
    Network      // U-codes
}

public enum DTCSeverity
{
    Info,
    Warning,
    Critical
}
