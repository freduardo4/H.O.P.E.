using HOPE.Core.Models;

namespace HOPE.Core.Services.ECU;

public interface IECUService
{
    /// <summary>
    /// Reads the full ECU calibration data
    /// </summary>
    Task<ECUCalibration> ReadCalibrationAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Reads a specific map by name (e.g., "FuelMap")
    /// </summary>
    Task<double[,]> ReadMapAsync(string mapName, CancellationToken cancellationToken = default);

    /// <summary>
    /// Writes a single value to ECU memory (RAM or Flash)
    /// </summary>
    Task<bool> WriteValueAsync(long address, double value, CancellationToken cancellationToken = default);
}
