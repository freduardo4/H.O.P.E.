using System.Security.Cryptography;
using System.Text;
using HOPE.Core.Services.Database;

namespace HOPE.Core.Services.Security;

/// <summary>
/// Implementation of the Private Side-chain Calibration Ledger.
/// Links each calibration change to the previous one via SHA-256 hash chaining.
/// </summary>
public class CalibrationLedgerService : ICalibrationLedgerService
{
    private readonly IDatabaseService _dbService;

    public CalibrationLedgerService(IDatabaseService dbService)
    {
        _dbService = dbService;
    }

    public async Task<string> CommmitChangeAsync(Guid calibrationId, byte[] binaryData, string author, string changeSummary)
    {
        var timestamp = DateTime.UtcNow;
        
        // 1. Calculate binary hash
        using var sha256 = SHA256.Create();
        var binaryHash = Convert.ToHexString(sha256.ComputeHash(binaryData));

        // 2. Get last block to chain from
        var lastBlock = await _dbService.GetLastLedgerEntryAsync();
        var prevHash = lastBlock?.BlockHash ?? "0000000000000000000000000000000000000000000000000000000000000000";
        var height = (lastBlock?.BlockHeight ?? 0) + 1;

        // 3. Create block content
        var blockContent = $"{height}|{timestamp:O}|{calibrationId}|{author}|{changeSummary}|{binaryHash}|{prevHash}";
        var blockHash = Convert.ToHexString(sha256.ComputeHash(Encoding.UTF8.GetBytes(blockContent)));

        // 4. Create Ledger Entry
        var entry = new LedgerEntry
        {
            BlockHeight = height,
            Timestamp = timestamp,
            CalibrationId = calibrationId,
            Author = author,
            ChangeSummary = changeSummary,
            BinaryHash = binaryHash,
            PreviousBlockHash = prevHash,
            BlockHash = blockHash,
            DigitalSignature = "SIM_SIGNED_BY_" + author.ToUpper() // Placeholder for PKI 2.2
        };

        // 5. Persist to verifiable database
        await _dbService.AddLedgerEntryAsync(entry);

        return blockHash;
    }

    public async Task<LedgerVerificationResult> VerifyLedgerAsync()
    {
        var entries = (await _dbService.GetLedgerEntriesAsync()).OrderBy(e => e.BlockHeight);
        string expectedPrevHash = "0000000000000000000000000000000000000000000000000000000000000000";
        int validCount = 0;

        using var sha256 = SHA256.Create();

        foreach (var entry in entries)
        {
            // Verify link
            if (entry.PreviousBlockHash != expectedPrevHash)
                return new LedgerVerificationResult(false, $"Broken chain at height {entry.BlockHeight}. Link expected {expectedPrevHash}, found {entry.PreviousBlockHash}", validCount);

            // Re-calculate block hash
            var blockContent = $"{entry.BlockHeight}|{entry.Timestamp:O}|{entry.CalibrationId}|{entry.Author}|{entry.ChangeSummary}|{entry.BinaryHash}|{entry.PreviousBlockHash}";
            var calculatedHash = Convert.ToHexString(sha256.ComputeHash(Encoding.UTF8.GetBytes(blockContent)));

            if (entry.BlockHash != calculatedHash)
                return new LedgerVerificationResult(false, $"Mismatched hash at height {entry.BlockHeight}", validCount);

            expectedPrevHash = entry.BlockHash;
            validCount++;
        }

        return new LedgerVerificationResult(true, "Ledger integrity verified.", validCount);
    }

    public async Task<IEnumerable<LedgerEntry>> GetHistoryAsync(Guid calibrationId)
    {
        var all = await _dbService.GetLedgerEntriesAsync();
        return all.Where(e => e.CalibrationId == calibrationId).OrderByDescending(e => e.Timestamp);
    }
}
