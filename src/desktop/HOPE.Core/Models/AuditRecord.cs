using System;

namespace HOPE.Core.Models;

/// <summary>
/// Represents an entry in the hash-chained cryptographic audit trail.
/// </summary>
public class AuditRecord
{
    public long Id { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Action { get; set; } = string.Empty;
    public Guid EntityId { get; set; }
    public string Metadata { get; set; } = string.Empty;
    
    /// <summary>
    /// SHA-256 hash of the content (Action, EntityId, Metadata, Timestamp)
    /// </summary>
    public string DataHash { get; set; } = string.Empty;
    
    /// <summary>
    /// The RecordHash of the previous entry in the chain
    /// </summary>
    public string PreviousHash { get; set; } = string.Empty;
    
    /// <summary>
    /// Cumulative SHA-256 hash (DataHash + PreviousHash)
    /// </summary>
    public string RecordHash { get; set; } = string.Empty;
}
