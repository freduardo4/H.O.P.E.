using System.Runtime.InteropServices;

namespace HOPE.Core.Models.Simulation;

/// <summary>
/// Telemetry data from BeamNG.drive via OutGauge protocol.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct BeamNgTelemetry
{
    public uint Time;               // time in milliseconds (to check order)
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
    public char[] Car;              // Car name (up to 3 characters and \0)
    public ushort Flags;            // OutGauge flags
    public byte Gear;               // Reverse: 0, Neutral: 1, First: 2...
    public byte PlayerId;           // Unique ID of the player in a race
    public float Speed;             // Meters per second
    public float Rpm;               // Engine RPM
    public float Turbo;             // Turbo pressure (bar)
    public float EngineTemp;        // Engine temperature (Celsius)
    public float Fuel;              // Fuel Level (0 to 1)
    public float OilPressure;       // Oil pressure (bar)
    public float OilTemp;           // Oil temperature (Celsius)
    public uint DashLights;         // Dash lights states
    public uint ShowLights;         // Shown lights states
    public float Throttle;          // Throttle (0 to 1)
    public float Brake;             // Brake (0 to 1)
    public float Clutch;            // Clutch (0 to 1)
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public char[] Display1;         // Top display line
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
    public char[] Display2;         // Bottom display line
    public int Id;                  // ID of the car
}

public enum OutGaugeFlags : ushort
{
    Shift = 1,          // Shift light
    Ctrl = 2,           // Control
    Turbo = 4,          // Turbo
    Km = 8,             // Km/H
    Bar = 16,           // Bar
    Slight = 32,        // Shift light (alternate?)
    Llight = 64,        // Low beam
    Hlight = 128,       // High beam
    Foglight = 256,     // Fog light
    Rearfog = 512,      // Rear fog light
    Handbrake = 1024,  // Handbrake
    Absworking = 2048,  // ABS working
    Escworking = 4096,  // ESC working
    Tcsactive = 8192,   // TCS active
    Fuelwarn = 16384,   // Fuel warning
    Overheated = 32768  // Overheated
}
