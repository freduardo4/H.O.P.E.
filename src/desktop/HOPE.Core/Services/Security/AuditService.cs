using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.Database;

namespace HOPE.Core.Services.Security;

public class AuditService : IAuditService
{
    private readonly IDatabaseService _dbService;

    public AuditService(IDatabaseService dbService)
    {
        _dbService = dbService;
    }

    public async Task LogActivityAsync(string action, Guid entityId, string metadata = "")
    {
        var timestamp = DateTime.UtcNow;
        var lastRecord = await _dbService.GetLastAuditRecordAsync();
        var previousHash = lastRecord?.RecordHash ?? string.Empty;

        var dataToHash = $"{action}|{entityId}|{metadata}|{timestamp:O}";
        var dataHash = ComputeHash(dataToHash);
        var recordHash = ComputeHash(dataHash + previousHash);

        var record = new AuditRecord
        {
            Action = action,
            EntityId = entityId,
            Metadata = metadata,
            Timestamp = timestamp,
            DataHash = dataHash,
            PreviousHash = previousHash,
            RecordHash = recordHash
        };

        await _dbService.AddAuditRecordAsync(record);
    }

    public async Task<IEnumerable<AuditRecord>> GetAuditLogsAsync()
    {
        return await _dbService.GetAuditLogsAsync();
    }

    public async Task<bool> VerifyChainAsync()
    {
        var logs = await _dbService.GetAuditLogsAsync();
        string expectedPreviousHash = string.Empty;

        foreach (var record in logs)
        {
            // Verify data integrity
            var dataToHash = $"{record.Action}|{record.EntityId}|{record.Metadata}|{record.Timestamp:O}";
            var actualDataHash = ComputeHash(dataToHash);
            if (actualDataHash != record.DataHash) return false;

            // Verify previous hash link
            if (record.PreviousHash != expectedPreviousHash) return false;

            // Verify record hash
            var actualRecordHash = ComputeHash(record.DataHash + record.PreviousHash);
            if (actualRecordHash != record.RecordHash) return false;

            expectedPreviousHash = record.RecordHash;
        }

        return true;
    }

    private string ComputeHash(string input)
    {
        using var sha256 = SHA256.Create();
        var bytes = Encoding.UTF8.GetBytes(input);
        var hashBytes = sha256.ComputeHash(bytes);
        return Convert.ToHexString(hashBytes);
    }
}
