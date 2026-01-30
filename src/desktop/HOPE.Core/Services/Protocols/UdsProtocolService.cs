using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Interfaces;
using HOPE.Core.Models;

namespace HOPE.Core.Services.Protocols;

/// <summary>
/// Unified Diagnostic Services (UDS) protocol handler - ISO 14229.
/// Provides standardized diagnostic communication with vehicle ECUs.
/// </summary>
public class UdsProtocolService : IUdsProtocolService, IDisposable
{
    private readonly IHardwareAdapter _adapter;
    private readonly Subject<UdsResponse> _responseSubject = new();
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    private UdsSession _currentSession = UdsSession.Default;
    private bool _securityUnlocked;
    private DateTime _lastTesterPresent = DateTime.MinValue;
    private CancellationTokenSource? _testerPresentCts;
    private Task? _testerPresentTask;

    /// <summary>
    /// Observable stream of UDS responses
    /// </summary>
    public IObservable<UdsResponse> ResponseStream => _responseSubject.AsObservable();

    /// <summary>
    /// Gets the current diagnostic session type
    /// </summary>
    public UdsSession CurrentSession => _currentSession;

    /// <summary>
    /// Gets whether security access has been granted
    /// </summary>
    public bool IsSecurityUnlocked => _securityUnlocked;

    /// <summary>
    /// Event raised when session changes
    /// </summary>
    public event EventHandler<UdsSessionChangedEventArgs>? SessionChanged;

    public UdsProtocolService(IHardwareAdapter adapter)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
    }

    #region Diagnostic Session Control (0x10)

    /// <summary>
    /// Start a diagnostic session (UDS 0x10)
    /// </summary>
    public async Task<UdsResponse> DiagnosticSessionControlAsync(
        UdsSession session,
        CancellationToken ct = default)
    {
        await _sessionLock.WaitAsync(ct);
        try
        {
            var request = new byte[] { UdsServiceId.DiagnosticSessionControl, (byte)session };
            var response = await SendUdsRequestAsync(request, ct);

            if (response.IsPositive)
            {
                var previousSession = _currentSession;
                _currentSession = session;
                _securityUnlocked = false; // Security resets on session change

                // Start tester present if in extended session
                if (session != UdsSession.Default)
                {
                    StartTesterPresentHeartbeat();
                }
                else
                {
                    StopTesterPresentHeartbeat();
                }

                SessionChanged?.Invoke(this, new UdsSessionChangedEventArgs(previousSession, session));
            }

            return response;
        }
        finally
        {
            _sessionLock.Release();
        }
    }

    #endregion

    #region ECU Reset (0x11)

    /// <summary>
    /// Reset the ECU (UDS 0x11)
    /// </summary>
    public async Task<UdsResponse> EcuResetAsync(
        EcuResetType resetType,
        CancellationToken ct = default)
    {
        var request = new byte[] { UdsServiceId.EcuReset, (byte)resetType };
        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Security Access (0x27)

    /// <summary>
    /// Request security seed (UDS 0x27)
    /// </summary>
    public async Task<UdsSecurityResponse> RequestSecuritySeedAsync(
        byte securityLevel,
        CancellationToken ct = default)
    {
        // Odd numbers are seed requests, even numbers are key sends
        var subFunction = (byte)((securityLevel * 2) - 1);
        var request = new byte[] { UdsServiceId.SecurityAccess, subFunction };
        var response = await SendUdsRequestAsync(request, ct);

        return new UdsSecurityResponse
        {
            IsPositive = response.IsPositive,
            Seed = response.IsPositive && response.Data.Length > 2
                ? response.Data.Skip(2).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode,
            RawData = response.Data
        };
    }

    /// <summary>
    /// Send security key (UDS 0x27)
    /// </summary>
    public async Task<UdsResponse> SendSecurityKeyAsync(
        byte securityLevel,
        byte[] key,
        CancellationToken ct = default)
    {
        var subFunction = (byte)(securityLevel * 2);
        var request = new byte[2 + key.Length];
        request[0] = UdsServiceId.SecurityAccess;
        request[1] = subFunction;
        Array.Copy(key, 0, request, 2, key.Length);

        var response = await SendUdsRequestAsync(request, ct);

        if (response.IsPositive)
        {
            _securityUnlocked = true;
        }

        return response;
    }

    /// <summary>
    /// Perform complete security access with a key algorithm
    /// </summary>
    public async Task<bool> PerformSecurityAccessAsync(
        byte securityLevel,
        Func<byte[], byte[]> keyAlgorithm,
        CancellationToken ct = default)
    {
        // Get seed
        var seedResponse = await RequestSecuritySeedAsync(securityLevel, ct);
        if (!seedResponse.IsPositive || seedResponse.Seed.Length == 0)
            return false;

        // If seed is all zeros, security is already unlocked
        if (seedResponse.Seed.All(b => b == 0))
        {
            _securityUnlocked = true;
            return true;
        }

        // Calculate key
        var key = keyAlgorithm(seedResponse.Seed);

        // Send key
        var keyResponse = await SendSecurityKeyAsync(securityLevel, key, ct);
        return keyResponse.IsPositive;
    }

    #endregion

    #region Read Data By Identifier (0x22)

    /// <summary>
    /// Read data by identifier (UDS 0x22)
    /// </summary>
    public async Task<UdsDataResponse> ReadDataByIdentifierAsync(
        ushort dataIdentifier,
        CancellationToken ct = default)
    {
        var request = new byte[]
        {
            UdsServiceId.ReadDataByIdentifier,
            (byte)(dataIdentifier >> 8),
            (byte)(dataIdentifier & 0xFF)
        };

        var response = await SendUdsRequestAsync(request, ct);

        return new UdsDataResponse
        {
            IsPositive = response.IsPositive,
            DataIdentifier = dataIdentifier,
            Data = response.IsPositive && response.Data.Length > 3
                ? response.Data.Skip(3).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    /// <summary>
    /// Read multiple data identifiers at once
    /// </summary>
    public async Task<List<UdsDataResponse>> ReadMultipleDataByIdentifierAsync(
        IEnumerable<ushort> dataIdentifiers,
        CancellationToken ct = default)
    {
        var dids = dataIdentifiers.ToArray();
        var request = new byte[1 + dids.Length * 2];
        request[0] = UdsServiceId.ReadDataByIdentifier;

        for (int i = 0; i < dids.Length; i++)
        {
            request[1 + i * 2] = (byte)(dids[i] >> 8);
            request[2 + i * 2] = (byte)(dids[i] & 0xFF);
        }

        var response = await SendUdsRequestAsync(request, ct);
        var results = new List<UdsDataResponse>();

        if (response.IsPositive)
        {
            // Parse response containing multiple DIDs
            // Format: 62 [DID1_H] [DID1_L] [Data1...] [DID2_H] [DID2_L] [Data2...] ...
            int offset = 1;
            while (offset < response.Data.Length - 2)
            {
                var did = (ushort)((response.Data[offset] << 8) | response.Data[offset + 1]);
                offset += 2;

                // Find length by looking for next DID or end
                int dataEnd = response.Data.Length;
                for (int i = offset; i < response.Data.Length - 1; i++)
                {
                    var potentialDid = (ushort)((response.Data[i] << 8) | response.Data[i + 1]);
                    if (dids.Contains(potentialDid))
                    {
                        dataEnd = i;
                        break;
                    }
                }

                var data = response.Data.Skip(offset).Take(dataEnd - offset).ToArray();
                results.Add(new UdsDataResponse
                {
                    IsPositive = true,
                    DataIdentifier = did,
                    Data = data
                });

                offset = dataEnd;
            }
        }
        else
        {
            // Return negative response for all DIDs
            foreach (var did in dids)
            {
                results.Add(new UdsDataResponse
                {
                    IsPositive = false,
                    DataIdentifier = did,
                    NegativeResponseCode = response.NegativeResponseCode
                });
            }
        }

        return results;
    }

    #endregion

    #region Write Data By Identifier (0x2E)

    /// <summary>
    /// Write data by identifier (UDS 0x2E)
    /// </summary>
    public async Task<UdsResponse> WriteDataByIdentifierAsync(
        ushort dataIdentifier,
        byte[] data,
        CancellationToken ct = default)
    {
        var request = new byte[3 + data.Length];
        request[0] = UdsServiceId.WriteDataByIdentifier;
        request[1] = (byte)(dataIdentifier >> 8);
        request[2] = (byte)(dataIdentifier & 0xFF);
        Array.Copy(data, 0, request, 3, data.Length);

        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Input Output Control By Identifier (0x2F)

    /// <summary>
    /// Control an input/output by identifier (UDS 0x2F)
    /// </summary>
    public async Task<UdsResponse> InputOutputControlByIdentifierAsync(
        ushort dataIdentifier,
        IoControlParameter controlParameter,
        byte[]? controlState = null,
        CancellationToken ct = default)
    {
        var stateBytes = controlState ?? Array.Empty<byte>();
        var request = new byte[4 + stateBytes.Length];
        request[0] = UdsServiceId.InputOutputControlByIdentifier;
        request[1] = (byte)(dataIdentifier >> 8);
        request[2] = (byte)(dataIdentifier & 0xFF);
        request[3] = (byte)controlParameter;

        if (stateBytes.Length > 0)
            Array.Copy(stateBytes, 0, request, 4, stateBytes.Length);

        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Routine Control (0x31)

    /// <summary>
    /// Start a routine (UDS 0x31)
    /// </summary>
    public async Task<UdsRoutineResponse> StartRoutineAsync(
        ushort routineId,
        byte[]? optionRecord = null,
        CancellationToken ct = default)
    {
        return await RoutineControlAsync(RoutineControlType.Start, routineId, optionRecord, ct);
    }

    /// <summary>
    /// Stop a routine (UDS 0x31)
    /// </summary>
    public async Task<UdsRoutineResponse> StopRoutineAsync(
        ushort routineId,
        byte[]? optionRecord = null,
        CancellationToken ct = default)
    {
        return await RoutineControlAsync(RoutineControlType.Stop, routineId, optionRecord, ct);
    }

    /// <summary>
    /// Request routine results (UDS 0x31)
    /// </summary>
    public async Task<UdsRoutineResponse> RequestRoutineResultsAsync(
        ushort routineId,
        CancellationToken ct = default)
    {
        return await RoutineControlAsync(RoutineControlType.RequestResults, routineId, null, ct);
    }

    private async Task<UdsRoutineResponse> RoutineControlAsync(
        RoutineControlType controlType,
        ushort routineId,
        byte[]? optionRecord,
        CancellationToken ct)
    {
        var options = optionRecord ?? Array.Empty<byte>();
        var request = new byte[4 + options.Length];
        request[0] = UdsServiceId.RoutineControl;
        request[1] = (byte)controlType;
        request[2] = (byte)(routineId >> 8);
        request[3] = (byte)(routineId & 0xFF);

        if (options.Length > 0)
            Array.Copy(options, 0, request, 4, options.Length);

        var response = await SendUdsRequestAsync(request, ct);

        return new UdsRoutineResponse
        {
            IsPositive = response.IsPositive,
            RoutineId = routineId,
            RoutineInfo = response.IsPositive && response.Data.Length > 4
                ? response.Data.Skip(4).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Request Download/Upload (0x34, 0x35)

    /// <summary>
    /// Request download (prepare ECU for receiving data) - UDS 0x34
    /// </summary>
    public async Task<UdsTransferResponse> RequestDownloadAsync(
        uint memoryAddress,
        uint memorySize,
        byte dataFormat = 0x00,
        CancellationToken ct = default)
    {
        // Determine address and length format information
        byte addressLengthFormat = 0x44; // 4 bytes address, 4 bytes size

        var request = new byte[11];
        request[0] = UdsServiceId.RequestDownload;
        request[1] = dataFormat;
        request[2] = addressLengthFormat;
        request[3] = (byte)(memoryAddress >> 24);
        request[4] = (byte)(memoryAddress >> 16);
        request[5] = (byte)(memoryAddress >> 8);
        request[6] = (byte)(memoryAddress & 0xFF);
        request[7] = (byte)(memorySize >> 24);
        request[8] = (byte)(memorySize >> 16);
        request[9] = (byte)(memorySize >> 8);
        request[10] = (byte)(memorySize & 0xFF);

        var response = await SendUdsRequestAsync(request, ct);

        ushort maxBlockSize = 0;
        if (response.IsPositive && response.Data.Length >= 3)
        {
            var lengthFormat = response.Data[1];
            var sizeBytes = lengthFormat >> 4;
            if (sizeBytes > 0 && response.Data.Length >= 2 + sizeBytes)
            {
                for (int i = 0; i < sizeBytes; i++)
                {
                    maxBlockSize = (ushort)((maxBlockSize << 8) | response.Data[2 + i]);
                }
            }
        }

        return new UdsTransferResponse
        {
            IsPositive = response.IsPositive,
            MaxBlockLength = maxBlockSize,
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    /// <summary>
    /// Request upload (prepare ECU for sending data) - UDS 0x35
    /// </summary>
    public async Task<UdsTransferResponse> RequestUploadAsync(
        uint memoryAddress,
        uint memorySize,
        byte dataFormat = 0x00,
        CancellationToken ct = default)
    {
        byte addressLengthFormat = 0x44;

        var request = new byte[11];
        request[0] = UdsServiceId.RequestUpload;
        request[1] = dataFormat;
        request[2] = addressLengthFormat;
        request[3] = (byte)(memoryAddress >> 24);
        request[4] = (byte)(memoryAddress >> 16);
        request[5] = (byte)(memoryAddress >> 8);
        request[6] = (byte)(memoryAddress & 0xFF);
        request[7] = (byte)(memorySize >> 24);
        request[8] = (byte)(memorySize >> 16);
        request[9] = (byte)(memorySize >> 8);
        request[10] = (byte)(memorySize & 0xFF);

        var response = await SendUdsRequestAsync(request, ct);

        ushort maxBlockSize = 0;
        if (response.IsPositive && response.Data.Length >= 3)
        {
            var lengthFormat = response.Data[1];
            var sizeBytes = lengthFormat >> 4;
            if (sizeBytes > 0 && response.Data.Length >= 2 + sizeBytes)
            {
                for (int i = 0; i < sizeBytes; i++)
                {
                    maxBlockSize = (ushort)((maxBlockSize << 8) | response.Data[2 + i]);
                }
            }
        }

        return new UdsTransferResponse
        {
            IsPositive = response.IsPositive,
            MaxBlockLength = maxBlockSize,
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Transfer Data (0x36)

    /// <summary>
    /// Transfer data block (UDS 0x36)
    /// </summary>
    public async Task<UdsResponse> TransferDataAsync(
        byte blockSequence,
        byte[] data,
        CancellationToken ct = default)
    {
        var request = new byte[2 + data.Length];
        request[0] = UdsServiceId.TransferData;
        request[1] = blockSequence;
        Array.Copy(data, 0, request, 2, data.Length);

        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Request Transfer Exit (0x37)

    /// <summary>
    /// Request transfer exit (UDS 0x37)
    /// </summary>
    public async Task<UdsResponse> RequestTransferExitAsync(
        byte[]? transferRequestParameter = null,
        CancellationToken ct = default)
    {
        var param = transferRequestParameter ?? Array.Empty<byte>();
        var request = new byte[1 + param.Length];
        request[0] = UdsServiceId.RequestTransferExit;

        if (param.Length > 0)
            Array.Copy(param, 0, request, 1, param.Length);

        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Clear Diagnostic Information (0x14)

    /// <summary>
    /// Clear diagnostic information / DTCs (UDS 0x14)
    /// </summary>
    public async Task<UdsResponse> ClearDiagnosticInformationAsync(
        uint groupOfDtc = 0xFFFFFF, // All DTCs
        CancellationToken ct = default)
    {
        var request = new byte[]
        {
            UdsServiceId.ClearDiagnosticInformation,
            (byte)(groupOfDtc >> 16),
            (byte)(groupOfDtc >> 8),
            (byte)(groupOfDtc & 0xFF)
        };

        return await SendUdsRequestAsync(request, ct);
    }

    #endregion

    #region Read DTC Information (0x19)

    /// <summary>
    /// Read DTC information (UDS 0x19)
    /// </summary>
    public async Task<UdsDtcResponse> ReadDtcInformationAsync(
        DtcReportType reportType,
        byte dtcStatusMask = 0xFF,
        CancellationToken ct = default)
    {
        var request = reportType switch
        {
            DtcReportType.ReportNumberOfDTCByStatusMask => new byte[]
            {
                UdsServiceId.ReadDtcInformation,
                (byte)reportType,
                dtcStatusMask
            },
            DtcReportType.ReportDTCByStatusMask => new byte[]
            {
                UdsServiceId.ReadDtcInformation,
                (byte)reportType,
                dtcStatusMask
            },
            _ => new byte[]
            {
                UdsServiceId.ReadDtcInformation,
                (byte)reportType
            }
        };

        var response = await SendUdsRequestAsync(request, ct);
        var dtcs = new List<UdsDtc>();

        if (response.IsPositive && response.Data.Length > 3)
        {
            // Parse DTC data (format: [high][mid][low][status])
            int offset = 3;
            while (offset + 4 <= response.Data.Length)
            {
                var dtcCode = (uint)((response.Data[offset] << 16) |
                                     (response.Data[offset + 1] << 8) |
                                     response.Data[offset + 2]);
                var status = response.Data[offset + 3];

                dtcs.Add(new UdsDtc
                {
                    DtcCode = dtcCode,
                    FormattedCode = FormatDtcCode(dtcCode),
                    StatusByte = status,
                    IsActive = (status & 0x01) != 0,
                    IsStored = (status & 0x08) != 0,
                    IsPending = (status & 0x04) != 0
                });

                offset += 4;
            }
        }

        return new UdsDtcResponse
        {
            IsPositive = response.IsPositive,
            DTCs = dtcs,
            DtcCount = dtcs.Count,
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    private static string FormatDtcCode(uint dtcCode)
    {
        var firstByte = (dtcCode >> 16) & 0xFF;
        var secondByte = (dtcCode >> 8) & 0xFF;
        var thirdByte = dtcCode & 0xFF;

        var prefix = (firstByte >> 6) switch
        {
            0 => 'P', // Powertrain
            1 => 'C', // Chassis
            2 => 'B', // Body
            3 => 'U', // Network
            _ => 'P'
        };

        var firstDigit = (firstByte >> 4) & 0x03;
        var secondDigit = firstByte & 0x0F;

        return $"{prefix}{firstDigit}{secondDigit:X}{secondByte:X2}";
    }

    #endregion

    #region Tester Present (0x3E)

    /// <summary>
    /// Send tester present to keep session alive (UDS 0x3E)
    /// </summary>
    public async Task<UdsResponse> TesterPresentAsync(
        bool suppressPositiveResponse = true,
        CancellationToken ct = default)
    {
        var subFunction = suppressPositiveResponse ? (byte)0x80 : (byte)0x00;
        var request = new byte[] { UdsServiceId.TesterPresent, subFunction };

        _lastTesterPresent = DateTime.UtcNow;

        if (suppressPositiveResponse)
        {
            // Don't wait for response
            await _adapter.SendMessageAsync(request, 100, ct);
            return new UdsResponse { IsPositive = true };
        }

        return await SendUdsRequestAsync(request, ct);
    }

    private void StartTesterPresentHeartbeat()
    {
        StopTesterPresentHeartbeat();

        _testerPresentCts = new CancellationTokenSource();
        _testerPresentTask = Task.Run(async () =>
        {
            while (!_testerPresentCts.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(2000, _testerPresentCts.Token);
                    await TesterPresentAsync(true, _testerPresentCts.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch
                {
                    // Ignore errors in heartbeat
                }
            }
        });
    }

    private void StopTesterPresentHeartbeat()
    {
        _testerPresentCts?.Cancel();
        _testerPresentCts?.Dispose();
        _testerPresentCts = null;
    }

    #endregion

    #region Core Communication

    private async Task<UdsResponse> SendUdsRequestAsync(byte[] request, CancellationToken ct)
    {
        try
        {
            var responseBytes = await _adapter.SendMessageAsync(request, 5000, ct);

            var response = ParseUdsResponse(responseBytes);
            _responseSubject.OnNext(response);

            // Handle "response pending" (0x78)
            while (response.NegativeResponseCode == UdsNrc.RequestCorrectlyReceivedResponsePending)
            {
                await Task.Delay(100, ct);
                responseBytes = await _adapter.SendMessageAsync(Array.Empty<byte>(), 5000, ct);
                response = ParseUdsResponse(responseBytes);
            }

            return response;
        }
        catch (Exception ex)
        {
            return new UdsResponse
            {
                IsPositive = false,
                NegativeResponseCode = UdsNrc.GeneralReject,
                ErrorMessage = ex.Message
            };
        }
    }

    private static UdsResponse ParseUdsResponse(byte[] data)
    {
        if (data == null || data.Length == 0)
        {
            return new UdsResponse
            {
                IsPositive = false,
                NegativeResponseCode = UdsNrc.NoResponseFromSubnetComponent,
                ErrorMessage = "No response received"
            };
        }

        // Check for potential corruption (e.g. all null bytes)
        if (data.All(b => b == 0x00))
        {
            return new UdsResponse
            {
                IsPositive = false,
                NegativeResponseCode = UdsNrc.GeneralReject,
                ErrorMessage = "Response data corrupted (all null bytes)"
            };
        }

        // Negative response format: 7F [ServiceId] [NRC]
        if (data[0] == 0x7F && data.Length >= 3)
        {
            return new UdsResponse
            {
                IsPositive = false,
                ServiceId = data[1],
                NegativeResponseCode = (UdsNrc)data[2],
                Data = data,
                ErrorMessage = GetNrcDescription((UdsNrc)data[2])
            };
        }

        // Positive response: ServiceId + 0x40
        return new UdsResponse
        {
            IsPositive = true,
            ServiceId = (byte)(data[0] - 0x40),
            Data = data
        };
    }

    private static string GetNrcDescription(UdsNrc nrc)
    {
        return nrc switch
        {
            UdsNrc.GeneralReject => "General reject",
            UdsNrc.ServiceNotSupported => "Service not supported",
            UdsNrc.SubFunctionNotSupported => "Sub-function not supported",
            UdsNrc.IncorrectMessageLengthOrInvalidFormat => "Incorrect message length or invalid format",
            UdsNrc.ResponseTooLong => "Response too long",
            UdsNrc.BusyRepeatRequest => "Busy - repeat request",
            UdsNrc.ConditionsNotCorrect => "Conditions not correct",
            UdsNrc.RequestSequenceError => "Request sequence error",
            UdsNrc.NoResponseFromSubnetComponent => "No response from subnet component",
            UdsNrc.FailurePreventsExecutionOfRequestedAction => "Failure prevents execution of requested action",
            UdsNrc.RequestOutOfRange => "Request out of range",
            UdsNrc.SecurityAccessDenied => "Security access denied",
            UdsNrc.InvalidKey => "Invalid key",
            UdsNrc.ExceededNumberOfAttempts => "Exceeded number of attempts",
            UdsNrc.RequiredTimeDelayNotExpired => "Required time delay not expired",
            UdsNrc.UploadDownloadNotAccepted => "Upload/download not accepted",
            UdsNrc.TransferDataSuspended => "Transfer data suspended",
            UdsNrc.GeneralProgrammingFailure => "General programming failure",
            UdsNrc.WrongBlockSequenceCounter => "Wrong block sequence counter",
            UdsNrc.RequestCorrectlyReceivedResponsePending => "Request correctly received - response pending",
            UdsNrc.SubFunctionNotSupportedInActiveSession => "Sub-function not supported in active session",
            UdsNrc.ServiceNotSupportedInActiveSession => "Service not supported in active session",
            _ => $"Unknown NRC: 0x{(byte)nrc:X2}"
        };
    }

    #endregion

    public void Dispose()
    {
        StopTesterPresentHeartbeat();
        _responseSubject.Dispose();
        _sessionLock.Dispose();
    }
}

#region UDS Constants and Types

/// <summary>
/// UDS Service IDs (ISO 14229)
/// </summary>
public static class UdsServiceId
{
    public const byte DiagnosticSessionControl = 0x10;
    public const byte EcuReset = 0x11;
    public const byte ClearDiagnosticInformation = 0x14;
    public const byte ReadDtcInformation = 0x19;
    public const byte ReadDataByIdentifier = 0x22;
    public const byte ReadMemoryByAddress = 0x23;
    public const byte SecurityAccess = 0x27;
    public const byte CommunicationControl = 0x28;
    public const byte WriteDataByIdentifier = 0x2E;
    public const byte InputOutputControlByIdentifier = 0x2F;
    public const byte RoutineControl = 0x31;
    public const byte RequestDownload = 0x34;
    public const byte RequestUpload = 0x35;
    public const byte TransferData = 0x36;
    public const byte RequestTransferExit = 0x37;
    public const byte TesterPresent = 0x3E;
    public const byte ControlDtcSetting = 0x85;
}

/// <summary>
/// UDS Diagnostic Session Types
/// </summary>
public enum UdsSession : byte
{
    Default = 0x01,
    Programming = 0x02,
    ExtendedDiagnostic = 0x03,
    SafetySystemDiagnostic = 0x04
}

/// <summary>
/// ECU Reset Types
/// </summary>
public enum EcuResetType : byte
{
    HardReset = 0x01,
    KeyOffOnReset = 0x02,
    SoftReset = 0x03
}

/// <summary>
/// Routine Control Types
/// </summary>
public enum RoutineControlType : byte
{
    Start = 0x01,
    Stop = 0x02,
    RequestResults = 0x03
}

/// <summary>
/// I/O Control Parameters
/// </summary>
public enum IoControlParameter : byte
{
    ReturnControlToEcu = 0x00,
    ResetToDefault = 0x01,
    FreezeCurrentState = 0x02,
    ShortTermAdjustment = 0x03
}

/// <summary>
/// DTC Report Types
/// </summary>
public enum DtcReportType : byte
{
    ReportNumberOfDTCByStatusMask = 0x01,
    ReportDTCByStatusMask = 0x02,
    ReportDTCSnapshotIdentification = 0x03,
    ReportDTCSnapshotRecordByDTCNumber = 0x04,
    ReportDTCSnapshotRecordByRecordNumber = 0x05,
    ReportDTCExtendedDataRecordByDTCNumber = 0x06,
    ReportSupportedDTC = 0x0A,
    ReportFirstTestFailedDTC = 0x0B,
    ReportMostRecentTestFailedDTC = 0x0E
}

/// <summary>
/// UDS Negative Response Codes
/// </summary>
public enum UdsNrc : byte
{
    GeneralReject = 0x10,
    ServiceNotSupported = 0x11,
    SubFunctionNotSupported = 0x12,
    IncorrectMessageLengthOrInvalidFormat = 0x13,
    ResponseTooLong = 0x14,
    BusyRepeatRequest = 0x21,
    ConditionsNotCorrect = 0x22,
    RequestSequenceError = 0x24,
    NoResponseFromSubnetComponent = 0x25,
    FailurePreventsExecutionOfRequestedAction = 0x26,
    RequestOutOfRange = 0x31,
    SecurityAccessDenied = 0x33,
    InvalidKey = 0x35,
    ExceededNumberOfAttempts = 0x36,
    RequiredTimeDelayNotExpired = 0x37,
    UploadDownloadNotAccepted = 0x70,
    TransferDataSuspended = 0x71,
    GeneralProgrammingFailure = 0x72,
    WrongBlockSequenceCounter = 0x73,
    RequestCorrectlyReceivedResponsePending = 0x78,
    SubFunctionNotSupportedInActiveSession = 0x7E,
    ServiceNotSupportedInActiveSession = 0x7F
}

#endregion

#region Response Types

public class UdsResponse
{
    public bool IsPositive { get; init; }
    public byte ServiceId { get; init; }
    public byte[] Data { get; init; } = Array.Empty<byte>();
    public UdsNrc NegativeResponseCode { get; init; }
    public string? ErrorMessage { get; init; }
}

public class UdsSecurityResponse : UdsResponse
{
    public byte[] Seed { get; init; } = Array.Empty<byte>();
    public byte[] RawData { get; init; } = Array.Empty<byte>();
}

public class UdsDataResponse : UdsResponse
{
    public ushort DataIdentifier { get; init; }
    public new byte[] Data { get; init; } = Array.Empty<byte>();
}

public class UdsRoutineResponse : UdsResponse
{
    public ushort RoutineId { get; init; }
    public byte[] RoutineInfo { get; init; } = Array.Empty<byte>();
}

public class UdsTransferResponse : UdsResponse
{
    public ushort MaxBlockLength { get; init; }
}

public class UdsDtcResponse : UdsResponse
{
    public List<UdsDtc> DTCs { get; init; } = new();
    public int DtcCount { get; init; }
}

public class UdsDtc
{
    public uint DtcCode { get; init; }
    public string FormattedCode { get; init; } = string.Empty;
    public byte StatusByte { get; init; }
    public bool IsActive { get; init; }
    public bool IsStored { get; init; }
    public bool IsPending { get; init; }
}

public class UdsSessionChangedEventArgs : EventArgs
{
    public UdsSession PreviousSession { get; }
    public UdsSession NewSession { get; }

    public UdsSessionChangedEventArgs(UdsSession previous, UdsSession newSession)
    {
        PreviousSession = previous;
        NewSession = newSession;
    }
}

#endregion
