using HOPE.Core.Models;

namespace HOPE.Core.Services.Security;

/// <summary>
/// Provides a cryptographically secure, verifiable ledger for ECU calibration changes.
/// </summary>
public interface ICalibrationLedgerService
{
    /// <summary>
    /// Records a calibration modification to the ledger.
    /// </summary>
    Task<string> CommmitChangeAsync(Guid calibrationId, byte[] binaryData, string author, string changeSummary);

    /// <summary>
    /// Verifies the integrity of the entire calibration ledger.
    /// </summary>
    Task<LedgerVerificationResult> VerifyLedgerAsync();

    /// <summary>
    /// Gets the full audit history for a specific calibration.
    /// </summary>
    Task<IEnumerable<LedgerEntry>> GetHistoryAsync(Guid calibrationId);
}

public class LedgerEntry
{
    public long BlockHeight { get; set; }
    public DateTime Timestamp { get; set; }
    public Guid CalibrationId { get; set; }
    public string Author { get; set; } = string.Empty;
    public string ChangeSummary { get; set; } = string.Empty;
    public string BinaryHash { get; set; } = string.Empty;
    public string PreviousBlockHash { get; set; } = string.Empty;
    public string BlockHash { get; set; } = string.Empty;
    public string DigitalSignature { get; set; } = string.Empty; // For Phase 2.2 PKI
}

public record LedgerVerificationResult(bool IsValid, string Message, int ValidBlocks);
