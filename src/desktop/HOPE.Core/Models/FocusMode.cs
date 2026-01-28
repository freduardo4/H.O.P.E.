namespace HOPE.Core.Models;

/// <summary>
/// Operational modes that focus the system on specific diagnostic or performance goals.
/// </summary>
public enum FocusMode
{
    /// <summary>
    /// Standard diagnostic mode with balanced polling.
    /// </summary>
    Standard,

    /// <summary>
    /// Wide Open Throttle mode for performance logging (50Hz+).
    /// </summary>
    WOT,

    /// <summary>
    /// Fuel economy monitoring with optimized polling for efficiency.
    /// </summary>
    Economy,

    /// <summary>
    /// Deep diagnostic mode for sensor-level inspection.
    /// </summary>
    Diagnostic
}
