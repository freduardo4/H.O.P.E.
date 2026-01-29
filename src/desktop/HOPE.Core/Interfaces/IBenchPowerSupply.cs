using System.Threading;
using System.Threading.Tasks;

namespace HOPE.Core.Interfaces;

/// <summary>
/// Interface for controlling bench power supply (Ignition/Battery).
/// </summary>
public interface IBenchPowerSupply
{
    /// <summary>
    /// Set the ignition state (e.g., via J2534 SetProgrammingVoltage on specific pin).
    /// </summary>
    Task<bool> SetIgnitionAsync(bool on, CancellationToken ct = default);

    /// <summary>
    /// Set the permanent power state (Terminal 30).
    /// Note: Standard J2534 adapters may not support this, usually requires specific bench harness control.
    /// </summary>
    Task<bool> SetPowerAsync(bool on, CancellationToken ct = default);

    /// <summary>
    /// Gets whether the adapter supports controlling permanent power.
    /// </summary>
    bool CanControlPower { get; }
}
