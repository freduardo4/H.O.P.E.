namespace HOPE.Core.Models;

/// <summary>
/// Represents a vehicle in the HOPE system.
/// </summary>
public class Vehicle
{
    public Guid VehicleId { get; set; }
    public Guid CustomerId { get; set; }

    /// <summary>
    /// Vehicle Identification Number (VIN) - 17 characters
    /// </summary>
    public string VIN { get; set; } = string.Empty;

    public string Make { get; set; } = string.Empty;
    public string Model { get; set; } = string.Empty;
    public int Year { get; set; }

    /// <summary>
    /// Engine code (e.g., "BAM" for VW 1.8T, "N54" for BMW)
    /// </summary>
    public string? EngineCode { get; set; }

    /// <summary>
    /// ECU type (e.g., "Bosch ME7.5", "Siemens EMS3155")
    /// </summary>
    public string? ECUType { get; set; }

    /// <summary>
    /// Current active calibration ID
    /// </summary>
    public Guid? CurrentCalibrationId { get; set; }

    /// <summary>
    /// Additional metadata (modifications, notes)
    /// </summary>
    public Dictionary<string, string> Metadata { get; set; } = new();

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Navigation property for diagnostic sessions
    /// </summary>
    public List<DiagnosticSession> DiagnosticSessions { get; set; } = new();

    /// <summary>
    /// Navigation property for ECU calibrations
    /// </summary>
    public List<ECUCalibration> ECUCalibrations { get; set; } = new();

    public override string ToString() => $"{Year} {Make} {Model} (VIN: {VIN})";
}
