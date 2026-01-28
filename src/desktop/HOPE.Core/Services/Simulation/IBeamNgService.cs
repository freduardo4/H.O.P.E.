using HOPE.Core.Models.Simulation;

namespace HOPE.Core.Services.Simulation;

/// <summary>
/// Interface for BeamNG.drive telemetry and control service.
/// </summary>
public interface IBeamNgService : IDisposable
{
    /// <summary>
    /// Stream of telemetry data from the simulation.
    /// </summary>
    IObservable<BeamNgTelemetry> TelemetryStream { get; }

    /// <summary>
    /// Gets whether the service is currently listening.
    /// </summary>
    bool IsConnected { get; }

    /// <summary>
    /// Start listening for telemetry on the specified port.
    /// </summary>
    Task StartAsync(int port = 4444, CancellationToken ct = default);

    /// <summary>
    /// Stop listening for telemetry.
    /// </summary>
    Task StopAsync();
}
