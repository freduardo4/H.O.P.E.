using HOPE.Core.Models;
using System.Reactive.Subjects;

namespace HOPE.Core.Services.Database;

public class SyncService : ISyncService
{
    private readonly IDatabaseService _dbService;
    private readonly BehaviorSubject<SyncStatusEventArgs> _syncStatus = new(new SyncStatusEventArgs());

    public event EventHandler<SyncStatusEventArgs>? SyncStatusChanged;

    public SyncService(IDatabaseService dbService)
    {
        _dbService = dbService;
    }

    public async Task PushChangesAsync()
    {
        OnSyncStatusChanged(true, "Pushing local changes...");
        
        try
        {
            // Simulate gathering dirty records from SQLite
            // In a real implementation, we would query a 'SyncMetadata' table
            var sessions = await _dbService.GetSessionsAsync();
            
            // Simulate API call to HOPE Central
            await Task.Delay(2000); 

            OnSyncStatusChanged(false, "Push complete.", DateTime.Now);
        }
        catch (Exception ex)
        {
            OnSyncStatusChanged(false, $"Push failed: {ex.Message}");
        }
    }

    public async Task PullChangesAsync()
    {
        OnSyncStatusChanged(true, "Pulling remote changes...");

        try
        {
            // Simulate fetching remote state
            await Task.Delay(1500);

            // Last-Write-Wins (LWW) Logic Implementation (Simulated)
            // 1. Fetch remote record
            // 2. Compare Remote.LastModified vs Local.LastModified
            // 3. Keep the most recent one
            
            OnSyncStatusChanged(false, "Pull complete. Merged 0 conflicts.", DateTime.Now);
        }
        catch (Exception ex)
        {
            OnSyncStatusChanged(false, $"Pull failed: {ex.Message}");
        }
    }

    private void OnSyncStatusChanged(bool isSyncing, string message, DateTime? lastSync = null)
    {
        var args = new SyncStatusEventArgs 
        { 
            IsSyncing = isSyncing, 
            Message = message, 
            LastSyncTime = lastSync 
        };
        SyncStatusChanged?.Invoke(this, args);
    }
}
