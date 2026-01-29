using HOPE.Core.Services.Protocols;

namespace HOPE.Core.Interfaces;

public interface IUdsProtocolService : IDisposable
{
    IObservable<UdsResponse> ResponseStream { get; }
    UdsSession CurrentSession { get; }
    bool IsSecurityUnlocked { get; }
    event EventHandler<UdsSessionChangedEventArgs>? SessionChanged;

    Task<UdsResponse> DiagnosticSessionControlAsync(UdsSession session, CancellationToken ct = default);
    Task<UdsResponse> EcuResetAsync(EcuResetType resetType, CancellationToken ct = default);
    Task<UdsSecurityResponse> RequestSecuritySeedAsync(byte securityLevel, CancellationToken ct = default);
    Task<UdsResponse> SendSecurityKeyAsync(byte securityLevel, byte[] key, CancellationToken ct = default);
    Task<bool> PerformSecurityAccessAsync(byte securityLevel, Func<byte[], byte[]> keyAlgorithm, CancellationToken ct = default);
    Task<UdsDataResponse> ReadDataByIdentifierAsync(ushort dataIdentifier, CancellationToken ct = default);
    Task<List<UdsDataResponse>> ReadMultipleDataByIdentifierAsync(IEnumerable<ushort> dataIdentifiers, CancellationToken ct = default);
    Task<UdsResponse> WriteDataByIdentifierAsync(ushort dataIdentifier, byte[] data, CancellationToken ct = default);
    Task<UdsResponse> InputOutputControlByIdentifierAsync(ushort dataIdentifier, IoControlParameter controlParameter, byte[]? controlState = null, CancellationToken ct = default);
    Task<UdsRoutineResponse> StartRoutineAsync(ushort routineId, byte[]? optionRecord = null, CancellationToken ct = default);
    Task<UdsRoutineResponse> StopRoutineAsync(ushort routineId, byte[]? optionRecord = null, CancellationToken ct = default);
    Task<UdsRoutineResponse> RequestRoutineResultsAsync(ushort routineId, CancellationToken ct = default);
    Task<UdsTransferResponse> RequestDownloadAsync(uint memoryAddress, uint memorySize, byte dataFormat = 0x00, CancellationToken ct = default);
    Task<UdsTransferResponse> RequestUploadAsync(uint memoryAddress, uint memorySize, byte dataFormat = 0x00, CancellationToken ct = default);
    Task<UdsResponse> TransferDataAsync(byte blockSequence, byte[] data, CancellationToken ct = default);
    Task<UdsResponse> RequestTransferExitAsync(byte[]? transferRequestParameter = null, CancellationToken ct = default);
    Task<UdsResponse> ClearDiagnosticInformationAsync(uint groupOfDtc = 0xFFFFFF, CancellationToken ct = default);
    Task<UdsDtcResponse> ReadDtcInformationAsync(DtcReportType reportType, byte dtcStatusMask = 0xFF, CancellationToken ct = default);
    Task<UdsResponse> TesterPresentAsync(bool suppressPositiveResponse = true, CancellationToken ct = default);
}
