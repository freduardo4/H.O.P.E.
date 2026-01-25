using HOPE.Core.Models;

namespace HOPE.Core.Services.Database;

public interface IDatabaseService
{
    Task InitializeAsync();
    Task<Guid> StartSessionAsync(Guid vehicleId);
    Task EndSessionAsync(Guid sessionId);
    Task LogReadingAsync(OBD2Reading reading);
    Task LogReadingsAsync(IEnumerable<OBD2Reading> readings);
    Task<List<DiagnosticSession>> GetSessionsAsync();
    Task<List<OBD2Reading>> GetSessionDataAsync(Guid sessionId);
}
