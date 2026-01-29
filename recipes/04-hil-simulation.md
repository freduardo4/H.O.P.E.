# Recipe 04: Hardware-in-the-Loop (HiL) Simulation

This recipe demonstrates how to use the HiL Testing Tier to inject synthetic hardware faults into the BeamNG.drive simulation environment for pre-flash validation.

## 1. Prerequisites
- BeamNG.drive installed and modded with the HOPE Lua bridge.
- Desktop App built and running.

## 2. Setting Up a Simulation
1.  Launch BeamNG.drive and load a vehicle.
2.  Enable the OutGauge/HOPE protocol in the game's hardware settings.
3.  In the HOPE Desktop App, navigate to **Simulation > HiL Testing**.

## 3. Injecting Faults via Code (C#)
If you are developing automated test sequences, use the `IHiLService`:

```csharp
// Inject a Voltage Drop fault
_hilService.InjectFault(new HiLFault 
{ 
    Type = FaultType.VoltageDrop, 
    Severity = 0.5 // 50% drop
});

// Clear all faults
_hilService.ClearFaults();
```

## 4. Manual Fault Injection
Use the **Simulation Dashboard** UI to toggle faults during a live drive:
- **Sensor Noise**: Adds Gaussian jitter to MAF/RPM signals.
- **Packet Loss**: Intermittently drops OutGauge packets to test ECU communication timeout logic.
- **Value Drift**: Gradually offsets sensor values to test long-term adaptation safety.

## 5. Verification
Observe the **Terminal** or **Diagnostics Log** to see how the AI Anomaly Detector and Safety Interlocks respond to the injected faults.
