# Recipe 01: OBD Diagnostic Session

## Overview
This recipe demonstrates how to initiate an OBD2 diagnostic session using the H.O.P.E. Core library.

## Prerequisites
- A J2534-compatible device (or use the Mock adapter for testing).
- `HOPE.Core.dll` referenced in your project.

## Code Example (C#)

```csharp
using HOPE.Core.Services.OBD;
using HOPE.Core.Hardware;

public async Task RunDiagnosticSession()
{
    // 1. Initialize Hardware (Use Mock for dev, J2534 for real hardware)
    using var adapter = new MockHardwareAdapter(); 
    // OR: new J2534Adapter("OpenPort 2.0");

    // 2. Initialize Service
    var obdService = new Obd2Service(adapter);

    // 3. Connect
    bool connected = await obdService.ConnectAsync();
    if (!connected) 
    {
        Console.WriteLine("Failed to connect to ECU.");
        return;
    }

    // 4. Read Data
    var rpm = await obdService.ReadPidAsync(ObdPids.EngineRPM);
    var speed = await obdService.ReadPidAsync(ObdPids.VehicleSpeed);
    
    Console.WriteLine($"RPM: {rpm.Value}, Speed: {speed.Value} km/h");

    // 5. Stream Data (High Frequency)
    obdService.HighFrequencyMode = true;
    var subscription = obdService.RealTimeStream.Subscribe(frame => 
    {
        Console.WriteLine($"[{frame.Timestamp}] PID {frame.Pid}: {frame.Value}");
    });

    await Task.Delay(5000); // Record for 5 seconds
    subscription.Dispose();
}
```

## Running via CLI
(Future capability)
```powershell
.\scripts\hope.ps1 session --start --log-file "my-session.log"
```
