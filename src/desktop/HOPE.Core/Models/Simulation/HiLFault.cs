using System;

namespace HOPE.Core.Models.Simulation;

public enum FaultType
{
    None,
    VoltageDrop,
    SensorNoise,
    PacketLoss,
    ValueDrift,
    ActuatorSeize,
    HighTemp
}

public class HiLFault
{
    public FaultType Type { get; set; }
    public double Intensity { get; set; } // 0.0 to 1.0
    public int DurationMs { get; set; }
    public DateTime OccurredAt { get; set; }
    public string? TargetParameter { get; set; } // e.g., "Rpm", "Voltage"
}
