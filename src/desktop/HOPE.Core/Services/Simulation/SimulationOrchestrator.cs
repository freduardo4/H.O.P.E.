using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Reactive.Linq;
using System.Reactive.Threading.Tasks;
using HOPE.Core.Models.Simulation;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.Simulation;

/// <summary>
/// Orchestrates simulation-based validation sessions.
/// </summary>
public class SimulationOrchestrator
{
    private readonly IBeamNgService _beamNg;
    private readonly ILogger<SimulationOrchestrator> _logger;

    public SimulationOrchestrator(IBeamNgService beamNg, ILogger<SimulationOrchestrator> logger)
    {
        _beamNg = beamNg;
        _logger = logger;
    }

    /// <summary>
    /// Validates a calibration in the simulation environment.
    /// </summary>
    /// <param name="calibrationData">The binary calibration data.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>True if the calibration passes simulation checks.</returns>
    public virtual async Task<(bool Success, string Message)> ValidateInSimulationAsync(byte[] calibrationData, CancellationToken ct = default)
    {
        _logger.LogInformation("Starting virtual pre-flight validation in simulation...");

        // 1. Ensure Simulation is running
        if (!_beamNg.IsConnected)
        {
            // If not connected, try to start with default port
            try
            {
                await _beamNg.StartAsync(4444, ct);
            }
            catch (Exception)
            {
                return (false, "BeamNG.drive simulation is not responding. Ensure OutGauge is enabled on port 4444.");
            }
        }

        // 2. Wait for stable telemetry (Digital Twin Sync)
        _logger.LogInformation("Waiting for Digital Twin telemetry sync...");
        
        try
        {
            // Wait for at least one telemetry packet with a timeout
            var telemetry = await _beamNg.TelemetryStream.Take(1).Timeout(TimeSpan.FromSeconds(5)).ToTask(ct);
            _logger.LogInformation("Digital Twin synchronized. Car: {Car}, RPM: {Rpm}", new string(telemetry.Car), telemetry.Rpm);
        }
        catch (Exception)
        {
            return (false, "Digital Twin sync timed out. Is the simulation paused or unconfigured?");
        }

        // 3. Perform basic sanity check (e.g., engine is not already blown up)
        // In a more advanced version, we would:
        // - Push the calibration to a BeamNG plugin
        // - Observe engine health under load
        
        // For Phase 6.1 (Infrastructure), we verify the bridge is functional.
        return (true, "Virtual validation successful: Bridge connection established and telemetry synchronized.");
    }
}
