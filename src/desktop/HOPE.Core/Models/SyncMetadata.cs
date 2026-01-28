namespace HOPE.Core.Models;

public class SyncMetadata
{
    public Guid EntityId { get; set; }
    public string EntityType { get; set; } = string.Empty;
    public DateTime LastModifiedLocal { get; set; }
    public DateTime? LastModifiedRemote { get; set; }
    public bool IsDirty { get; set; }
    public string ETag { get; set; } = string.Empty;

    public static SyncMetadata New(Guid id, string type) => new()
    {
        EntityId = id,
        EntityType = type,
        LastModifiedLocal = DateTime.UtcNow,
        IsDirty = true
    };
}
