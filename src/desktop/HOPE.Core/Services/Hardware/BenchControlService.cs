using System;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.Hardware;

public class BenchControlService
{
    private readonly IHardwareAdapter _hardware;
    private readonly ILogger<BenchControlService> _logger;
    
    // Safety lock: if strictly true, don't allow powering off.
    private bool _safetyLockEnabled;

    public BenchControlService(IHardwareAdapter hardware, ILogger<BenchControlService> logger)
    {
        _hardware = hardware;
        _logger = logger;
    }

    /// <summary>
    /// Sets the safety lock. When locked, ignition/power cannot be turned off.
    /// Use this during critical operations like flashing.
    /// </summary>
    public void SetSafetyLock(bool enabled)
    {
        _safetyLockEnabled = enabled;
        _logger.LogInformation("Bench safety lock set to {Enabled}", enabled);
    }

    public async Task<bool> SetIgnitionAsync(bool on, CancellationToken ct = default)
    {
        if (!on && _safetyLockEnabled)
        {
            _logger.LogInformation("Blocked attempt to turn off ignition while unsafe.");
            return false;
        }

        if (_hardware is IBenchPowerSupply bench)
        {
            _logger.LogInformation($"Setting Ignition {(on ? "ON" : "OFF")} via IBenchPowerSupply...");
            return await bench.SetIgnitionAsync(on, ct);
        }
        
        // Fallback for standard J2534: Try using SetProgrammingVoltage on Pin 12 (standard-ish for some bench cables)
        // or check if it's a known adapter like Scanmatik which might map this differently.
        // For now, we assume Pin 12 is the target API for "Ignition" control on generic J2534 bench implementations.
        // Scanmatik documentation usually suggests using the programming voltage API for its Aux/Ignition pins.
        
        /* 
         * NOTE: Scanmatik 2 PRO Aux output often maps to specific SetProgrammingVoltage calls.
         * Defaulting to Pin 12 is a common convention for FEPS, but for bench power it might be different.
         * We will try Pin 12 (FEPS) as a proxy for Ignition on many commercial J2534 bench cables.
         */
         
        // 12V roughly for ON, 0V for OFF (or short to ground, but 0V is safer request)
        // Actually, SetProgrammingVoltage usually expects millivolts in the int API, but our interface takes double volts.
        double voltage = on ? 12.0 : 0.0;
        
        // Pin 12 is often OEM Programming Voltage.
        const int PIN_12 = 12; 
        
        try 
        {
             _logger.LogInformation($"Attempting to set Ignition {(on ? "ON" : "OFF")} via Pin 12...");
             return await _hardware.SetProgrammingVoltageAsync(PIN_12, voltage, ct);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to control ignition via generic J2534 Pin 12.");
            return false;
        }
    }

    public async Task<bool> SetPowerAsync(bool on, CancellationToken ct = default)
    {
        if (!on && _safetyLockEnabled)
        {
            _logger.LogInformation("Blocked attempt to turn off power while unsafe.");
            return false;
        }

        if (_hardware is IBenchPowerSupply bench && bench.CanControlPower)
        {
             _logger.LogInformation($"Setting Power {(on ? "ON" : "OFF")}...");
             return await bench.SetPowerAsync(on, ct);
        }

        _logger.LogWarning("Hardware does not support managing main power line.");
        return false;
    }
}
