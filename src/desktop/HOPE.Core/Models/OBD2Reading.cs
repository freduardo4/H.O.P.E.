namespace HOPE.Core.Models;

/// <summary>
/// Represents a single OBD2 parameter reading at a specific time.
/// </summary>
public record OBD2Reading
{
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;
    public Guid SessionId { get; init; }

    /// <summary>
    /// Parameter ID (PID) in hex format (e.g., "0x0C" for Engine RPM)
    /// </summary>
    public string PID { get; init; } = string.Empty;

    /// <summary>
    /// Human-readable parameter name (e.g., "Engine RPM")
    /// </summary>
    public string Name { get; init; } = string.Empty;

    /// <summary>
    /// Numeric value of the reading
    /// </summary>
    public double Value { get; init; }

    /// <summary>
    /// Unit of measurement (e.g., "RPM", "°C", "PSI")
    /// </summary>
    public string Unit { get; init; } = string.Empty;

    /// <summary>
    /// Raw hex response from ECU (for debugging)
    /// </summary>
    public string? RawResponse { get; init; }

    public override string ToString() => $"{Name}: {Value:F2} {Unit} @ {Timestamp:HH:mm:ss.fff}";
}

/// <summary>
/// Standard OBD2 PIDs (SAE J1979)
/// </summary>
public static class OBD2PIDs
{
    // Engine Parameters
    public const string EngineRPM = "0C";              // RPM
    public const string EngineLoad = "04";             // %
    public const string CoolantTemp = "05";            // °C
    public const string IntakeAirTemp = "0F";          // °C
    public const string ThrottlePosition = "11";       // %
    public const string VehicleSpeed = "0D";           // km/h

    // Air/Fuel
    public const string MAFSensor = "10";              // g/s
    public const string FuelPressure = "0A";           // kPa
    public const string ShortTermFuelTrim = "06";      // %
    public const string LongTermFuelTrim = "07";       // %

    // Oxygen Sensors
    public const string O2Sensor1Voltage = "14";       // V
    public const string O2Sensor2Voltage = "15";       // V
    public const string O2Sensor1ShortTerm = "14";     // %
    public const string O2Sensor2ShortTerm = "15";     // %

    // Ignition
    public const string TimingAdvance = "0E";          // degrees before TDC
    public const string IntakeManifoldPressure = "0B"; // kPa

    // Status
    public const string EngineRuntime = "1F";          // seconds
    public const string DistanceSinceDTCCleared = "31"; // km
    public const string BarometricPressure = "33";     // kPa
}

/// <summary>
/// OBD2 parameter metadata
/// </summary>
public class OBD2Parameter
{
    public string PID { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Unit { get; set; } = string.Empty;
    public double MinValue { get; set; }
    public double MaxValue { get; set; }
    public string Formula { get; set; } = string.Empty;
    public int ByteCount { get; set; }

    /// <summary>
    /// Determines if this parameter should be logged for AI training
    /// </summary>
    public bool IsAIRelevant { get; set; }

    /// <summary>
    /// Priority for real-time display (1 = highest)
    /// </summary>
    public int DisplayPriority { get; set; }
}

/// <summary>
/// Configuration for OBD2 data streaming
/// </summary>
public class OBD2StreamConfig
{
    /// <summary>
    /// PIDs to stream continuously
    /// </summary>
    public List<string> ActivePIDs { get; set; } = new();

    /// <summary>
    /// Polling interval in milliseconds (100-1000ms typical)
    /// </summary>
    public int PollingIntervalMs { get; set; } = 200;

    /// <summary>
    /// Maximum age for data before considered stale (ms)
    /// </summary>
    public int StaleDataThresholdMs { get; set; } = 2000;

    /// <summary>
    /// Enable logging to database
    /// </summary>
    public bool EnableLogging { get; set; } = true;

    /// <summary>
    /// Downsample rate for logging (1 = log every reading, 5 = log every 5th)
    /// </summary>
    public int LoggingDownsampleRate { get; set; } = 1;
}
