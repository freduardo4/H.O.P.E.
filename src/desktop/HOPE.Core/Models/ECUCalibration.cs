namespace HOPE.Core.Models;

/// <summary>
/// Represents an ECU calibration file with versioning and metadata.
/// </summary>
public class ECUCalibration
{
    public Guid CalibrationId { get; set; }
    public Guid VehicleId { get; set; }
    public Guid SessionId { get; set; }

    /// <summary>
    /// Version string (e.g., "Stock", "Stage 1", "Stage 2", "Custom v1.2")
    /// </summary>
    public string Version { get; set; } = "Stock";

    /// <summary>
    /// Calibration type/category
    /// </summary>
    public CalibrationType Type { get; set; } = CalibrationType.Stock;

    /// <summary>
    /// File storage location (S3 key, local path, etc.)
    /// </summary>
    public string FileStoragePath { get; set; } = string.Empty;

    /// <summary>
    /// SHA-256 hash of the calibration file for integrity verification
    /// </summary>
    public string FileHash { get; set; } = string.Empty;

    /// <summary>
    /// File size in bytes
    /// </summary>
    public long FileSizeBytes { get; set; }

    /// <summary>
    /// Checksum algorithm used (CRC16, CRC32, Bosch, etc.)
    /// </summary>
    public string ChecksumType { get; set; } = "CRC16";

    /// <summary>
    /// Checksum value from ECU file
    /// </summary>
    public string ChecksumValue { get; set; } = string.Empty;

    /// <summary>
    /// Whether checksum validation passed
    /// </summary>
    public bool IsChecksumValid { get; set; }

    /// <summary>
    /// Calibration modifications/parameters (JSON)
    /// </summary>
    public CalibrationModifications? Modifications { get; set; }

    /// <summary>
    /// Parent calibration ID (for version history tracking)
    /// </summary>
    public Guid? ParentCalibrationId { get; set; }

    public DateTime UploadedAt { get; set; } = DateTime.UtcNow;
    public Guid UploadedBy { get; set; }

    /// <summary>
    /// Whether this calibration is currently active on the vehicle
    /// </summary>
    public bool IsActive { get; set; }

    /// <summary>
    /// Notes about this calibration
    /// </summary>
    public string? Notes { get; set; }

    /// <summary>
    /// Navigation property to vehicle
    /// </summary>
    public Vehicle? Vehicle { get; set; }

    /// <summary>
    /// Navigation property to diagnostic session
    /// </summary>
    public DiagnosticSession? Session { get; set; }

    public override string ToString() => $"{Type} - {Version} ({FileSizeBytes / 1024}KB)";
}

/// <summary>
/// Type/stage of ECU calibration
/// </summary>
public enum CalibrationType
{
    Stock,
    Stage1,
    Stage2,
    Stage3,
    Custom,
    E85,
    Economy,
    Valet
}

/// <summary>
/// Modifications made to the ECU calibration
/// </summary>
public class CalibrationModifications
{
    /// <summary>
    /// Estimated horsepower gain (if applicable)
    /// </summary>
    public double? HorsepowerGain { get; set; }

    /// <summary>
    /// Estimated torque gain (if applicable)
    /// </summary>
    public double? TorqueGain { get; set; }

    /// <summary>
    /// Boost pressure changes (PSI)
    /// </summary>
    public BoostModifications? Boost { get; set; }

    /// <summary>
    /// Fuel map modifications
    /// </summary>
    public FuelModifications? Fuel { get; set; }

    /// <summary>
    /// Ignition timing modifications
    /// </summary>
    public IgnitionModifications? Ignition { get; set; }

    /// <summary>
    /// Torque limiter changes
    /// </summary>
    public TorqueLimiterModifications? TorqueLimiter { get; set; }

    /// <summary>
    /// Other modifications
    /// </summary>
    public Dictionary<string, string> Other { get; set; } = new();
}

public class BoostModifications
{
    public double? StockMaxBoost { get; set; }
    public double? ModifiedMaxBoost { get; set; }
    public double? BoostIncrease { get; set; }
    public string? BoostControlStrategy { get; set; }
}

public class FuelModifications
{
    public double? MaxFuelPressure { get; set; }
    public double? TargetAFR { get; set; }
    public string? InjectorScaling { get; set; }
    public bool? LambdaDisabled { get; set; }
}

public class IgnitionModifications
{
    public double? MaxTimingAdvance { get; set; }
    public double? TimingAtPeakTorque { get; set; }
    public string? KnockControl { get; set; }
}

public class TorqueLimiterModifications
{
    public int? Gear1Limit { get; set; }
    public int? Gear2Limit { get; set; }
    public int? Gear3Limit { get; set; }
    public int? Gear4Limit { get; set; }
    public int? Gear5Limit { get; set; }
    public int? Gear6Limit { get; set; }
}
