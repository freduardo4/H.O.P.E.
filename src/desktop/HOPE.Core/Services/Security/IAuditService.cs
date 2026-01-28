using System;
using System.Threading.Tasks;
using System.Collections.Generic;
using HOPE.Core.Models;

namespace HOPE.Core.Services.Security;

/// <summary>
/// Service for maintaining an immutable, hash-chained audit log.
/// </summary>
public interface IAuditService
{
    /// <summary>
    /// Appends a new activity to the audit trail.
    /// </summary>
    Task LogActivityAsync(string action, Guid entityId, string metadata = "");

    /// <summary>
    /// Returns all audit records.
    /// </summary>
    Task<IEnumerable<AuditRecord>> GetAuditLogsAsync();

    /// <summary>
    /// Verifies the integrity of the audit chain.
    /// </summary>
    Task<bool> VerifyChainAsync();
}
