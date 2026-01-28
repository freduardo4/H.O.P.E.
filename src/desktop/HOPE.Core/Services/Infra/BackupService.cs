using System.IO;
using System.IO.Compression;

namespace HOPE.Core.Services.Infra;

public interface IBackupService
{
    Task CreateLocalBackupAsync();
}

public class BackupService : IBackupService
{
    private readonly string _repoPath;
    private readonly string _backupPath;

    public BackupService(string repoPath)
    {
        _repoPath = repoPath;
        _backupPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "HOPE", "Backups");
    }

    public async Task CreateLocalBackupAsync()
    {
        if (!Directory.Exists(_repoPath)) return;
        if (!Directory.Exists(_backupPath)) Directory.CreateDirectory(_backupPath);

        string timestamp = DateTime.Now.ToString("yyyyMMdd-HHmmss");
        string zipFile = Path.Combine(_backupPath, $"hope-repo-backup-{timestamp}.zip");

        await Task.Run(() => ZipFile.CreateFromDirectory(_repoPath, zipFile));
    }
}
