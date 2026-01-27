using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Interfaces;

namespace HOPE.Core.Services.Protocols;

/// <summary>
/// KWP2000 (ISO 14230-4) protocol handler for older vehicle diagnostics.
/// Supports both 5-baud initialization and fast initialization.
/// </summary>
public class Kwp2000ProtocolService : IDisposable
{
    private readonly IHardwareAdapter _adapter;
    private readonly Subject<Kwp2000Response> _responseSubject = new();
    private readonly SemaphoreSlim _sessionLock = new(1, 1);

    private byte _keyByte1;
    private byte _keyByte2;
    private byte _ecuAddress = 0x01;
    private byte _testerAddress = 0xF1;
    private bool _isInitialized;
    private Kwp2000Session _currentSession = Kwp2000Session.Default;
    private CancellationTokenSource? _keepAliveCts;

    /// <summary>
    /// Observable stream of KWP2000 responses
    /// </summary>
    public IObservable<Kwp2000Response> ResponseStream => _responseSubject.AsObservable();

    /// <summary>
    /// Gets whether the protocol is initialized
    /// </summary>
    public bool IsInitialized => _isInitialized;

    /// <summary>
    /// Gets the current session type
    /// </summary>
    public Kwp2000Session CurrentSession => _currentSession;

    /// <summary>
    /// Gets or sets the ECU address (default 0x01)
    /// </summary>
    public byte EcuAddress
    {
        get => _ecuAddress;
        set => _ecuAddress = value;
    }

    /// <summary>
    /// Gets or sets the tester address (default 0xF1)
    /// </summary>
    public byte TesterAddress
    {
        get => _testerAddress;
        set => _testerAddress = value;
    }

    public Kwp2000ProtocolService(IHardwareAdapter adapter)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
    }

    #region Initialization

    /// <summary>
    /// Initialize communication using 5-baud initialization
    /// </summary>
    public async Task<bool> Initialize5BaudAsync(byte initAddress = 0x33, CancellationToken ct = default)
    {
        if (_adapter.Type == HardwareType.ELM327)
        {
            // Configure ELM327 for ISO 14230-4 5-baud init
            await _adapter.SendCommandAsync("ATSP4", ct); // ISO 14230-4 (KWP 5 baud init)
            await _adapter.SendCommandAsync($"ATIIA{initAddress:X2}", ct); // Set init address
            await _adapter.SendCommandAsync("ATSI", ct); // Slow init

            var response = await _adapter.SendCommandAsync("0100", ct);

            if (!response.Contains("ERROR") && !response.Contains("UNABLE"))
            {
                _isInitialized = true;
                StartKeepAlive();
                return true;
            }
        }
        else if (_adapter.Type == HardwareType.J2534)
        {
            // J2534 handles initialization at hardware level
            var success = await _adapter.SetProtocolAsync(VehicleProtocol.ISO14230_KWP2000_5Baud, ct);
            if (success)
            {
                _isInitialized = true;
                StartKeepAlive();
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Initialize communication using fast initialization
    /// </summary>
    public async Task<bool> InitializeFastAsync(CancellationToken ct = default)
    {
        if (_adapter.Type == HardwareType.ELM327)
        {
            // Configure ELM327 for ISO 14230-4 fast init
            await _adapter.SendCommandAsync("ATSP5", ct); // ISO 14230-4 (KWP fast init)
            await _adapter.SendCommandAsync("ATFI", ct); // Fast init

            var response = await _adapter.SendCommandAsync("0100", ct);

            if (!response.Contains("ERROR") && !response.Contains("UNABLE"))
            {
                _isInitialized = true;
                StartKeepAlive();
                return true;
            }
        }
        else if (_adapter.Type == HardwareType.J2534)
        {
            var success = await _adapter.SetProtocolAsync(VehicleProtocol.ISO14230_KWP2000_Fast, ct);
            if (success)
            {
                _isInitialized = true;
                StartKeepAlive();
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Stop communication
    /// </summary>
    public async Task StopCommunicationAsync(CancellationToken ct = default)
    {
        StopKeepAlive();

        var request = BuildKwpMessage(Kwp2000ServiceId.StopCommunication, Array.Empty<byte>());
        await SendKwpRequestAsync(request, ct);

        _isInitialized = false;
        _currentSession = Kwp2000Session.Default;
    }

    #endregion

    #region Diagnostic Session Control (0x10)

    /// <summary>
    /// Start a diagnostic session
    /// </summary>
    public async Task<Kwp2000Response> StartDiagnosticSessionAsync(
        Kwp2000Session session,
        CancellationToken ct = default)
    {
        await _sessionLock.WaitAsync(ct);
        try
        {
            var request = BuildKwpMessage(Kwp2000ServiceId.StartDiagnosticSession, new[] { (byte)session });
            var response = await SendKwpRequestAsync(request, ct);

            if (response.IsPositive)
            {
                _currentSession = session;
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
    /// Reset the ECU
    /// </summary>
    public async Task<Kwp2000Response> EcuResetAsync(
        Kwp2000ResetMode resetMode = Kwp2000ResetMode.PowerOnReset,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.EcuReset, new[] { (byte)resetMode });
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Security Access (0x27)

    /// <summary>
    /// Request security seed
    /// </summary>
    public async Task<Kwp2000SecurityResponse> RequestSecuritySeedAsync(
        byte accessType = 0x01,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.SecurityAccess, new[] { accessType });
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000SecurityResponse
        {
            IsPositive = response.IsPositive,
            Seed = response.IsPositive && response.Data.Length > 2
                ? response.Data.Skip(2).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    /// <summary>
    /// Send security key
    /// </summary>
    public async Task<Kwp2000Response> SendSecurityKeyAsync(
        byte accessType,
        byte[] key,
        CancellationToken ct = default)
    {
        var data = new byte[1 + key.Length];
        data[0] = (byte)(accessType + 1); // Key send is access type + 1
        Array.Copy(key, 0, data, 1, key.Length);

        var request = BuildKwpMessage(Kwp2000ServiceId.SecurityAccess, data);
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Read Data By Local Identifier (0x21)

    /// <summary>
    /// Read data by local identifier (KWP2000 specific)
    /// </summary>
    public async Task<Kwp2000DataResponse> ReadDataByLocalIdentifierAsync(
        byte localIdentifier,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.ReadDataByLocalIdentifier, new[] { localIdentifier });
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000DataResponse
        {
            IsPositive = response.IsPositive,
            LocalIdentifier = localIdentifier,
            Data = response.IsPositive && response.Data.Length > 2
                ? response.Data.Skip(2).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Read Data By Common Identifier (0x22)

    /// <summary>
    /// Read data by common identifier
    /// </summary>
    public async Task<Kwp2000DataResponse> ReadDataByCommonIdentifierAsync(
        ushort identifier,
        CancellationToken ct = default)
    {
        var data = new byte[] { (byte)(identifier >> 8), (byte)(identifier & 0xFF) };
        var request = BuildKwpMessage(Kwp2000ServiceId.ReadDataByCommonIdentifier, data);
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000DataResponse
        {
            IsPositive = response.IsPositive,
            CommonIdentifier = identifier,
            Data = response.IsPositive && response.Data.Length > 3
                ? response.Data.Skip(3).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Read Memory By Address (0x23)

    /// <summary>
    /// Read memory by address
    /// </summary>
    public async Task<Kwp2000MemoryResponse> ReadMemoryByAddressAsync(
        uint address,
        byte size,
        CancellationToken ct = default)
    {
        var data = new byte[]
        {
            (byte)(address >> 16),
            (byte)(address >> 8),
            (byte)(address & 0xFF),
            size
        };

        var request = BuildKwpMessage(Kwp2000ServiceId.ReadMemoryByAddress, data);
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000MemoryResponse
        {
            IsPositive = response.IsPositive,
            Address = address,
            Data = response.IsPositive && response.Data.Length > 1
                ? response.Data.Skip(1).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Write Memory By Address (0x3D)

    /// <summary>
    /// Write memory by address
    /// </summary>
    public async Task<Kwp2000Response> WriteMemoryByAddressAsync(
        uint address,
        byte[] data,
        CancellationToken ct = default)
    {
        var messageData = new byte[3 + data.Length];
        messageData[0] = (byte)(address >> 16);
        messageData[1] = (byte)(address >> 8);
        messageData[2] = (byte)(address & 0xFF);
        Array.Copy(data, 0, messageData, 3, data.Length);

        var request = BuildKwpMessage(Kwp2000ServiceId.WriteMemoryByAddress, messageData);
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Clear Diagnostic Information (0x14)

    /// <summary>
    /// Clear diagnostic information
    /// </summary>
    public async Task<Kwp2000Response> ClearDiagnosticInformationAsync(
        ushort group = 0xFFFF,
        CancellationToken ct = default)
    {
        var data = new byte[] { (byte)(group >> 8), (byte)(group & 0xFF) };
        var request = BuildKwpMessage(Kwp2000ServiceId.ClearDiagnosticInformation, data);
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Read Status of DTCs (0x17)

    /// <summary>
    /// Read status of diagnostic trouble codes
    /// </summary>
    public async Task<Kwp2000DtcResponse> ReadStatusOfDtcsAsync(
        ushort group = 0xFFFF,
        CancellationToken ct = default)
    {
        var data = new byte[] { (byte)(group >> 8), (byte)(group & 0xFF) };
        var request = BuildKwpMessage(Kwp2000ServiceId.ReadStatusOfDtc, data);
        var response = await SendKwpRequestAsync(request, ct);

        var dtcs = new List<Kwp2000Dtc>();

        if (response.IsPositive && response.Data.Length > 1)
        {
            int offset = 1;
            while (offset + 3 <= response.Data.Length)
            {
                var highByte = response.Data[offset];
                var lowByte = response.Data[offset + 1];
                var status = response.Data[offset + 2];

                dtcs.Add(new Kwp2000Dtc
                {
                    Code = (ushort)((highByte << 8) | lowByte),
                    FormattedCode = FormatKwpDtc(highByte, lowByte),
                    Status = status,
                    IsActive = (status & 0x80) != 0,
                    IsPending = (status & 0x40) != 0
                });

                offset += 3;
            }
        }

        return new Kwp2000DtcResponse
        {
            IsPositive = response.IsPositive,
            DTCs = dtcs,
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    private static string FormatKwpDtc(byte high, byte low)
    {
        var prefix = (high >> 6) switch
        {
            0 => 'P',
            1 => 'C',
            2 => 'B',
            3 => 'U',
            _ => 'P'
        };

        return $"{prefix}{(high & 0x3F):X2}{low:X2}";
    }

    #endregion

    #region Read Freeze Frame Data (0x12)

    /// <summary>
    /// Read freeze frame data
    /// </summary>
    public async Task<Kwp2000DataResponse> ReadFreezeFrameDataAsync(
        byte frameNumber,
        byte localId,
        CancellationToken ct = default)
    {
        var data = new byte[] { frameNumber, localId };
        var request = BuildKwpMessage(Kwp2000ServiceId.ReadFreezeFrameData, data);
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000DataResponse
        {
            IsPositive = response.IsPositive,
            LocalIdentifier = localId,
            Data = response.IsPositive && response.Data.Length > 2
                ? response.Data.Skip(2).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Input Output Control (0x30)

    /// <summary>
    /// Control an actuator via I/O control
    /// </summary>
    public async Task<Kwp2000Response> InputOutputControlAsync(
        byte localIdentifier,
        Kwp2000IoControl controlParam,
        byte[]? controlState = null,
        CancellationToken ct = default)
    {
        var state = controlState ?? Array.Empty<byte>();
        var data = new byte[2 + state.Length];
        data[0] = localIdentifier;
        data[1] = (byte)controlParam;

        if (state.Length > 0)
            Array.Copy(state, 0, data, 2, state.Length);

        var request = BuildKwpMessage(Kwp2000ServiceId.InputOutputControlByLocalIdentifier, data);
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Start Routine By Local Identifier (0x31)

    /// <summary>
    /// Start a routine by local identifier
    /// </summary>
    public async Task<Kwp2000Response> StartRoutineByLocalIdentifierAsync(
        byte routineLocalId,
        byte[]? routineOptionRecord = null,
        CancellationToken ct = default)
    {
        var options = routineOptionRecord ?? Array.Empty<byte>();
        var data = new byte[1 + options.Length];
        data[0] = routineLocalId;

        if (options.Length > 0)
            Array.Copy(options, 0, data, 1, options.Length);

        var request = BuildKwpMessage(Kwp2000ServiceId.StartRoutineByLocalIdentifier, data);
        return await SendKwpRequestAsync(request, ct);
    }

    /// <summary>
    /// Stop a routine by local identifier
    /// </summary>
    public async Task<Kwp2000Response> StopRoutineByLocalIdentifierAsync(
        byte routineLocalId,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.StopRoutineByLocalIdentifier, new[] { routineLocalId });
        return await SendKwpRequestAsync(request, ct);
    }

    /// <summary>
    /// Request routine results by local identifier
    /// </summary>
    public async Task<Kwp2000DataResponse> RequestRoutineResultsByLocalIdentifierAsync(
        byte routineLocalId,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.RequestRoutineResultsByLocalIdentifier, new[] { routineLocalId });
        var response = await SendKwpRequestAsync(request, ct);

        return new Kwp2000DataResponse
        {
            IsPositive = response.IsPositive,
            LocalIdentifier = routineLocalId,
            Data = response.IsPositive && response.Data.Length > 2
                ? response.Data.Skip(2).ToArray()
                : Array.Empty<byte>(),
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Request Download (0x34)

    /// <summary>
    /// Request download (prepare ECU for receiving data)
    /// </summary>
    public async Task<Kwp2000TransferResponse> RequestDownloadAsync(
        uint address,
        uint size,
        byte dataFormat = 0x00,
        CancellationToken ct = default)
    {
        var data = new byte[]
        {
            dataFormat,
            (byte)(address >> 16),
            (byte)(address >> 8),
            (byte)(address & 0xFF),
            (byte)(size >> 16),
            (byte)(size >> 8),
            (byte)(size & 0xFF)
        };

        var request = BuildKwpMessage(Kwp2000ServiceId.RequestDownload, data);
        var response = await SendKwpRequestAsync(request, ct);

        byte maxBlockSize = 0;
        if (response.IsPositive && response.Data.Length >= 2)
        {
            maxBlockSize = response.Data[1];
        }

        return new Kwp2000TransferResponse
        {
            IsPositive = response.IsPositive,
            MaxBlockSize = maxBlockSize,
            NegativeResponseCode = response.NegativeResponseCode
        };
    }

    #endregion

    #region Transfer Data (0x36)

    /// <summary>
    /// Transfer data block
    /// </summary>
    public async Task<Kwp2000Response> TransferDataAsync(
        byte[] blockData,
        CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.TransferData, blockData);
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Request Transfer Exit (0x37)

    /// <summary>
    /// Request transfer exit
    /// </summary>
    public async Task<Kwp2000Response> RequestTransferExitAsync(CancellationToken ct = default)
    {
        var request = BuildKwpMessage(Kwp2000ServiceId.RequestTransferExit, Array.Empty<byte>());
        return await SendKwpRequestAsync(request, ct);
    }

    #endregion

    #region Tester Present (0x3E)

    /// <summary>
    /// Send tester present to keep session alive
    /// </summary>
    public async Task<Kwp2000Response> TesterPresentAsync(
        bool responseRequired = false,
        CancellationToken ct = default)
    {
        var subFunction = responseRequired ? (byte)0x01 : (byte)0x02;
        var request = BuildKwpMessage(Kwp2000ServiceId.TesterPresent, new[] { subFunction });

        if (!responseRequired)
        {
            await _adapter.SendMessageAsync(request, 100, ct);
            return new Kwp2000Response { IsPositive = true };
        }

        return await SendKwpRequestAsync(request, ct);
    }

    private void StartKeepAlive()
    {
        StopKeepAlive();

        _keepAliveCts = new CancellationTokenSource();
        Task.Run(async () =>
        {
            while (!_keepAliveCts.Token.IsCancellationRequested)
            {
                try
                {
                    await Task.Delay(2000, _keepAliveCts.Token);
                    await TesterPresentAsync(false, _keepAliveCts.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch
                {
                    // Ignore keep-alive errors
                }
            }
        });
    }

    private void StopKeepAlive()
    {
        _keepAliveCts?.Cancel();
        _keepAliveCts?.Dispose();
        _keepAliveCts = null;
    }

    #endregion

    #region Core Communication

    private byte[] BuildKwpMessage(byte serviceId, byte[] data)
    {
        // KWP2000 message format: [Format] [Target] [Source] [Length] [ServiceID] [Data...] [Checksum]
        // For ISO 14230-4, we use the simplified format with length in format byte

        var length = 1 + data.Length; // Service ID + data

        if (length <= 63)
        {
            // Physical addressing with length in format byte
            var message = new byte[4 + data.Length];
            message[0] = (byte)(0x80 | length); // Format byte with length
            message[1] = _ecuAddress;
            message[2] = _testerAddress;
            message[3] = serviceId;

            if (data.Length > 0)
                Array.Copy(data, 0, message, 4, data.Length);

            return message;
        }
        else
        {
            // Additional length byte required
            var message = new byte[5 + data.Length];
            message[0] = 0x80; // Format byte without length
            message[1] = _ecuAddress;
            message[2] = _testerAddress;
            message[3] = (byte)length;
            message[4] = serviceId;

            if (data.Length > 0)
                Array.Copy(data, 0, message, 5, data.Length);

            return message;
        }
    }

    private async Task<Kwp2000Response> SendKwpRequestAsync(byte[] request, CancellationToken ct)
    {
        try
        {
            var responseBytes = await _adapter.SendMessageAsync(request, 5000, ct);
            var response = ParseKwpResponse(responseBytes);

            _responseSubject.OnNext(response);

            return response;
        }
        catch (Exception ex)
        {
            return new Kwp2000Response
            {
                IsPositive = false,
                NegativeResponseCode = Kwp2000Nrc.GeneralReject,
                ErrorMessage = ex.Message
            };
        }
    }

    private static Kwp2000Response ParseKwpResponse(byte[] data)
    {
        if (data == null || data.Length < 4)
        {
            return new Kwp2000Response
            {
                IsPositive = false,
                NegativeResponseCode = Kwp2000Nrc.NoResponse,
                ErrorMessage = "No response received"
            };
        }

        // Skip header bytes and get to service ID
        var offset = 3; // Format, Target, Source
        if ((data[0] & 0x3F) == 0)
        {
            offset = 4; // Additional length byte
        }

        if (offset >= data.Length)
        {
            return new Kwp2000Response
            {
                IsPositive = false,
                ErrorMessage = "Invalid response format"
            };
        }

        var serviceResponse = data[offset];

        // Negative response: 0x7F
        if (serviceResponse == 0x7F && data.Length > offset + 2)
        {
            return new Kwp2000Response
            {
                IsPositive = false,
                ServiceId = data[offset + 1],
                NegativeResponseCode = (Kwp2000Nrc)data[offset + 2],
                Data = data,
                ErrorMessage = GetKwpNrcDescription((Kwp2000Nrc)data[offset + 2])
            };
        }

        // Positive response: Service ID + 0x40
        return new Kwp2000Response
        {
            IsPositive = true,
            ServiceId = (byte)(serviceResponse - 0x40),
            Data = data.Skip(offset).ToArray()
        };
    }

    private static string GetKwpNrcDescription(Kwp2000Nrc nrc)
    {
        return nrc switch
        {
            Kwp2000Nrc.GeneralReject => "General reject",
            Kwp2000Nrc.ServiceNotSupported => "Service not supported",
            Kwp2000Nrc.SubFunctionNotSupported => "Sub-function not supported or invalid format",
            Kwp2000Nrc.BusyRepeatRequest => "Busy - repeat request",
            Kwp2000Nrc.ConditionsNotCorrect => "Conditions not correct",
            Kwp2000Nrc.RoutineNotComplete => "Routine not complete",
            Kwp2000Nrc.RequestOutOfRange => "Request out of range",
            Kwp2000Nrc.SecurityAccessDenied => "Security access denied",
            Kwp2000Nrc.InvalidKey => "Invalid key",
            Kwp2000Nrc.ExceededAttempts => "Exceeded number of attempts",
            Kwp2000Nrc.RequiredTimeDelayNotExpired => "Required time delay not expired",
            Kwp2000Nrc.DownloadNotAccepted => "Download not accepted",
            Kwp2000Nrc.UploadNotAccepted => "Upload not accepted",
            Kwp2000Nrc.TransferSuspended => "Transfer suspended",
            Kwp2000Nrc.TransferAborted => "Transfer aborted",
            Kwp2000Nrc.AddressOrLengthFormatNotValid => "Address or length format not valid",
            Kwp2000Nrc.WrongBlockSequence => "Wrong block sequence counter",
            Kwp2000Nrc.ResponsePending => "Response pending",
            Kwp2000Nrc.ServiceNotSupportedInActiveSession => "Service not supported in active diagnostic session",
            _ => $"Unknown NRC: 0x{(byte)nrc:X2}"
        };
    }

    #endregion

    public void Dispose()
    {
        StopKeepAlive();
        _responseSubject.Dispose();
        _sessionLock.Dispose();
    }
}

#region KWP2000 Constants and Types

public static class Kwp2000ServiceId
{
    public const byte StartDiagnosticSession = 0x10;
    public const byte EcuReset = 0x11;
    public const byte ReadFreezeFrameData = 0x12;
    public const byte ClearDiagnosticInformation = 0x14;
    public const byte ReadStatusOfDtc = 0x17;
    public const byte ReadDtcByStatus = 0x18;
    public const byte ReadDataByLocalIdentifier = 0x21;
    public const byte ReadDataByCommonIdentifier = 0x22;
    public const byte ReadMemoryByAddress = 0x23;
    public const byte SecurityAccess = 0x27;
    public const byte DisableNormalMessageTransmission = 0x28;
    public const byte EnableNormalMessageTransmission = 0x29;
    public const byte DynamicallyDefineLocalIdentifier = 0x2C;
    public const byte WriteDataByCommonIdentifier = 0x2E;
    public const byte InputOutputControlByLocalIdentifier = 0x30;
    public const byte StartRoutineByLocalIdentifier = 0x31;
    public const byte StopRoutineByLocalIdentifier = 0x32;
    public const byte RequestRoutineResultsByLocalIdentifier = 0x33;
    public const byte RequestDownload = 0x34;
    public const byte RequestUpload = 0x35;
    public const byte TransferData = 0x36;
    public const byte RequestTransferExit = 0x37;
    public const byte WriteDataByLocalIdentifier = 0x3B;
    public const byte WriteMemoryByAddress = 0x3D;
    public const byte TesterPresent = 0x3E;
    public const byte StopCommunication = 0x82;
}

public enum Kwp2000Session : byte
{
    Default = 0x81,
    Programming = 0x85,
    Extended = 0x86,
    Adjustment = 0x87,
    FlashReprogramming = 0x89
}

public enum Kwp2000ResetMode : byte
{
    PowerOnReset = 0x01,
    NonVolatileMemoryReset = 0x82
}

public enum Kwp2000IoControl : byte
{
    ReturnControlToEcu = 0x00,
    ReportCurrentState = 0x01,
    ReportInputOutputConditions = 0x02,
    ReportScalingAndOffset = 0x03,
    FreezeCurrentState = 0x04,
    ExecuteControl = 0x05,
    ShortTermAdjustment = 0x07,
    LongTermAdjustment = 0x08
}

public enum Kwp2000Nrc : byte
{
    GeneralReject = 0x10,
    ServiceNotSupported = 0x11,
    SubFunctionNotSupported = 0x12,
    BusyRepeatRequest = 0x21,
    ConditionsNotCorrect = 0x22,
    RoutineNotComplete = 0x23,
    RequestOutOfRange = 0x31,
    SecurityAccessDenied = 0x33,
    InvalidKey = 0x35,
    ExceededAttempts = 0x36,
    RequiredTimeDelayNotExpired = 0x37,
    DownloadNotAccepted = 0x40,
    UploadNotAccepted = 0x50,
    TransferSuspended = 0x71,
    TransferAborted = 0x72,
    AddressOrLengthFormatNotValid = 0x73,
    WrongBlockSequence = 0x74,
    ResponsePending = 0x78,
    ServiceNotSupportedInActiveSession = 0x80,
    NoResponse = 0xFF
}

#endregion

#region Response Types

public class Kwp2000Response
{
    public bool IsPositive { get; init; }
    public byte ServiceId { get; init; }
    public byte[] Data { get; init; } = Array.Empty<byte>();
    public Kwp2000Nrc NegativeResponseCode { get; init; }
    public string? ErrorMessage { get; init; }
}

public class Kwp2000SecurityResponse : Kwp2000Response
{
    public byte[] Seed { get; init; } = Array.Empty<byte>();
}

public class Kwp2000DataResponse : Kwp2000Response
{
    public byte LocalIdentifier { get; init; }
    public ushort CommonIdentifier { get; init; }
    public new byte[] Data { get; init; } = Array.Empty<byte>();
}

public class Kwp2000MemoryResponse : Kwp2000Response
{
    public uint Address { get; init; }
    public new byte[] Data { get; init; } = Array.Empty<byte>();
}

public class Kwp2000TransferResponse : Kwp2000Response
{
    public byte MaxBlockSize { get; init; }
}

public class Kwp2000DtcResponse : Kwp2000Response
{
    public List<Kwp2000Dtc> DTCs { get; init; } = new();
}

public class Kwp2000Dtc
{
    public ushort Code { get; init; }
    public string FormattedCode { get; init; } = string.Empty;
    public byte Status { get; init; }
    public bool IsActive { get; init; }
    public bool IsPending { get; init; }
}

#endregion
