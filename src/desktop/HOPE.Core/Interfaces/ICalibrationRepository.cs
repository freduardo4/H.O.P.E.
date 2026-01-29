using HOPE.Core.Services.ECU;

namespace HOPE.Core.Interfaces;

public interface ICalibrationRepository : IDisposable
{
    string RepositoryPath { get; }
    event EventHandler<CalibrationCommitEventArgs>? CommitCreated;

    Task InitializeAsync(CancellationToken ct = default);
    Task<CalibrationFile> ReadFromEcuAsync(IHardwareAdapter adapter, EcuReadConfig config, IProgress<CalibrationProgress>? progress = null, CancellationToken ct = default);
    Task StageAsync(CalibrationFile calibration, CancellationToken ct = default);
    Task<string> CommitAsync(string message, string author = "HOPE User", CancellationToken ct = default);
    Task<CalibrationDiff> DiffAsync(string commitA, string commitB, CancellationToken ct = default);
    Task<CalibrationDiff> DiffCalibrationsAsync(CalibrationFile calA, CalibrationFile calB, CancellationToken ct = default);
    Task<ChecksumValidationResult> ValidateChecksumAsync(CalibrationFile calibration, CancellationToken ct = default);
    Task<CalibrationFile> RollbackAsync(string commitHash, CancellationToken ct = default);
    Task<CalibrationFile> GetCalibrationAsync(string commitHash, CancellationToken ct = default);
    Task<List<CalibrationCommit>> GetHistoryAsync(int limit = 50, CancellationToken ct = default);
    Task CreateTagAsync(string tagName, string commitHash, string? message = null, CancellationToken ct = default);
    Task ExportToBinaryAsync(CalibrationFile calibration, string outputPath, CancellationToken ct = default);
}
