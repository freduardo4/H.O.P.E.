using HOPE.Core.Services.Safety;

namespace HOPE.Core.Interfaces;

public interface ICloudSafetyService
{
    Task<bool> ValidateFlashOperationAsync(string ecuId, double voltage, CancellationToken ct = default);
    Task LogSafetyEventAsync(SafetyEvent evt, CancellationToken ct = default);
}
