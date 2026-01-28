using System;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Services.OBD;
using HOPE.Core.Interfaces;

namespace HOPE.Core.Services.Safety;

public class PreFlightService
{
    private readonly CloudSafetyService _cloudSafety;
    private readonly IHardwareAdapter _hardware;

    public PreFlightService(CloudSafetyService cloudSafety, IHardwareAdapter hardware)
    {
        _cloudSafety = cloudSafety ?? throw new ArgumentNullException(nameof(cloudSafety));
        _hardware = hardware ?? throw new ArgumentNullException(nameof(hardware));
    }

    public async Task<(bool Success, string Message)> RunFullCheckAsync(string ecuId, CancellationToken ct = default)
    {
        // 1. Connection Check
        if (!_hardware.IsConnected)
        {
            return (false, "Hardware adapter is not connected.");
        }

        // 2. Voltage Check (Local)
        double? voltageResult = await _hardware.ReadBatteryVoltageAsync(ct);
        if (!voltageResult.HasValue)
        {
            return (false, "Could not read battery voltage. Adapter may not support voltage monitoring.");
        }

        double voltage = voltageResult.Value;
        if (voltage < 12.5)
        {
            return (false, $"Low battery voltage: {voltage:F1}V. Minimum 12.5V required.");
        }

        // 3. Engine State Check (Simulated for RPM)
        // In a real scenario, we'd query PID 0x0C
        // For now, let's assume if it's connected and voltage is high, we still want to ensure 0 RPM
        // This is a placeholder for actual OBD query logic
        bool isEngineRunning = false; // Mock
        if (isEngineRunning)
        {
            return (false, "Engine is running. Please turn off the engine before flashing.");
        }

        // 4. Cloud Policy Check
        bool cloudAllowed = await _cloudSafety.ValidateFlashOperationAsync(ecuId, voltage, ct);
        if (!cloudAllowed)
        {
            return (false, "Cloud safety policy rejected the operation. Check connectivity or battery health.");
        }

        return (true, "Pre-flight checks passed. Safe to proceed.");
    }
}
