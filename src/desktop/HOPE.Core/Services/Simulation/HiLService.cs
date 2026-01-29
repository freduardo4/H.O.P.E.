using System;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Linq;
using System.Threading.Tasks;
using HOPE.Core.Models.Simulation;

namespace HOPE.Core.Services.Simulation;

public class HiLService : IHiLService
{
    private readonly ConcurrentDictionary<FaultType, HiLFault> _activeFaults = new();
    private readonly Random _random = new();

    public bool IsActive => !_activeFaults.IsEmpty;

    public IReadOnlyList<HiLFault> ActiveFaults => _activeFaults.Values.ToList().AsReadOnly();

    public void InjectFault(HiLFault fault)
    {
        _activeFaults[fault.Type] = fault;
        
        if (fault.DurationMs > 0)
        {
            Task.Delay(fault.DurationMs).ContinueWith(t => _activeFaults.TryRemove(fault.Type, out _));
        }
    }

    public void ClearFaults()
    {
        _activeFaults.Clear();
    }

    public BeamNgTelemetry ProcessTelemetry(BeamNgTelemetry raw)
    {
        if (!IsActive) return raw;

        var result = raw;

        foreach (var fault in _activeFaults.Values)
        {
            switch (fault.Type)
            {
                case FaultType.VoltageDrop:
                    // Force a significant voltage drop
                    // In a real OutGauge packet, we might not have 'Voltage' directly, 
                    // but we might simulate it for the UI/Safety layer.
                    // Assuming result.Voltage (this might need adjusting based on the actual struct)
                    // Let's assume for now we add noise or drops to parameters.
                    break;

                case FaultType.SensorNoise:
                    if (fault.TargetParameter == "Rpm")
                    {
                        result.Rpm += (float)(_random.NextDouble() * 1000 * fault.Intensity);
                    }
                    else if (fault.TargetParameter == "Speed")
                    {
                        result.Speed += (float)(_random.NextDouble() * 5 * fault.Intensity);
                    }
                    break;

                case FaultType.ValueDrift:
                    if (fault.TargetParameter == "EngineTemp")
                    {
                        result.EngineTemp += (float)(fault.Intensity * 50); // Drift by up to 50 degrees
                    }
                    break;
                    
                case FaultType.HighTemp:
                    result.EngineTemp = 130.0f; // Critical temp
                    break;
                
                case FaultType.PacketLoss:
                    // This would be handled at the service level (dropping the packet)
                    break;
            }
        }

        return result;
    }
}
