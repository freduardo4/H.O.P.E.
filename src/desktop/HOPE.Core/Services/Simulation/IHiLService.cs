using System.Collections.Generic;
using HOPE.Core.Models.Simulation;

namespace HOPE.Core.Services.Simulation;

public interface IHiLService
{
    void InjectFault(HiLFault fault);
    void ClearFaults();
    bool IsActive { get; }
    IReadOnlyList<HiLFault> ActiveFaults { get; }
    
    /// <summary>
    /// Process a telemetry packet and apply active faults.
    /// </summary>
    BeamNgTelemetry ProcessTelemetry(BeamNgTelemetry raw);
}
