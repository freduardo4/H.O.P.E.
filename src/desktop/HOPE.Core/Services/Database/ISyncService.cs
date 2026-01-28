namespace HOPE.Core.Services.Database;

/// <summary>
/// Service for synchronizing local SQLite data with HOPE Central cloud.
/// </summary>
public interface ISyncService
{
    /// <summary>
    /// Pushes local 'dirty' changes to the remote server.
    /// </summary>
    Task PushChangesAsync();

    /// <summary>
    /// Pulls remote changes and merges them into the local database using CRDT logic.
    /// </summary>
    Task PullChangesAsync();

    /// <summary>
    /// Event fired when synchronization status changes.
    /// </summary>
    event EventHandler<SyncStatusEventArgs> SyncStatusChanged;
}

public class SyncStatusEventArgs : EventArgs
{
    public bool IsSyncing { get; set; }
    public string Message { get; set; } = string.Empty;
    public DateTime? LastSyncTime { get; set; }
}
