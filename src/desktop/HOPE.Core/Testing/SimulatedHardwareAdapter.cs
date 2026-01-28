using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.Protocols;

namespace HOPE.Core.Testing;

/// <summary>
/// A simulated hardware adapter for testing purposes.
/// Implements IHardwareAdapter and provides internal ECU state simulation.
/// </summary>
public class SimulatedHardwareAdapter : IHardwareAdapter
{
    private readonly Subject<byte[]> _messageSubject = new();
    private bool _isConnected;
    private UdsSession _currentSession = UdsSession.Default;
    private bool _securityUnlocked;
    private byte[] _lastSeed = Array.Empty<byte>();
    private readonly byte[] _simulatedMemory = new byte[0x10000]; // 64KB simulated memory
    private uint _lastDownloadAddress;
    private uint _currentTransferOffset;
    
    // Configurable simulation parameters
    public double SimulatedVoltage { get; set; } = 13.5;
    public int SimulatedLatencyMs { get; set; } = 10;
    public bool InjectError { get; set; }
    public HardwareErrorType InjectedErrorType { get; set; } = HardwareErrorType.CommunicationError;
    public int? DropVoltageAfterMessages { get; set; }
    public double VoltageAfterDrop { get; set; } = 11.0;

    private int _messageCount = 0;

    public HardwareType Type => HardwareType.J2534;
    public string AdapterName => "Simulated ECU Adapter";
    public bool IsConnected => _isConnected;
    public bool SupportsHighFrequency => true;
    public bool SupportsVoltageMonitoring => true;
    public bool SupportsBiDirectionalControl => true;

    public event EventHandler<HardwareConnectionEventArgs>? ConnectionChanged;
    public event EventHandler<HardwareErrorEventArgs>? ErrorOccurred;

    public async Task<bool> ConnectAsync(string connectionString, int baudRate = 500000, CancellationToken cancellationToken = default)
    {
        await Task.Delay(50, cancellationToken);
        _isConnected = true;
        ConnectionChanged?.Invoke(this, new HardwareConnectionEventArgs(true, AdapterName, Type));
        return true;
    }

    public async Task DisconnectAsync()
    {
        await Task.Delay(50);
        _isConnected = false;
        ConnectionChanged?.Invoke(this, new HardwareConnectionEventArgs(false, AdapterName, Type));
    }

    public async Task<byte[]> SendMessageAsync(byte[] data, int timeout = 1000, CancellationToken cancellationToken = default)
    {
        if (!_isConnected) throw new InvalidOperationException("Adapter not connected");

        _messageCount++;
        if (DropVoltageAfterMessages.HasValue && _messageCount >= DropVoltageAfterMessages.Value)
        {
            SimulatedVoltage = VoltageAfterDrop;
        }

        if (SimulatedLatencyMs > 0)
            await Task.Delay(SimulatedLatencyMs, cancellationToken);

        if (InjectError)
        {
            ErrorOccurred?.Invoke(this, new HardwareErrorEventArgs("Injected error", InjectedErrorType));
            return Array.Empty<byte>();
        }

        // Handle KWP2000 or UDS
        if (data != null && data.Length > 0)
        {
            bool isKwp = (data[0] & 0x80) != 0;
            byte serviceId;
            byte[] payload;
            byte[] responsePayload = Array.Empty<byte>();

            if (isKwp)
            {
                // KWP format: [Format] [Target] [Source] [Length?] [Service] [Data...]
                int headerSize = (data[0] & 0x3F) != 0 ? 3 : 4;
                serviceId = data[headerSize];
                payload = data.Skip(headerSize + 1).ToArray();
            }
            else
            {
                serviceId = data[0];
                payload = data.Skip(1).ToArray();
            }

            switch (serviceId)
            {
                case 0x81: // StartCommunication (KWP)
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), 0x01 }; // Key byte 1
                    break;

                case UdsServiceId.DiagnosticSessionControl:
                    byte session = isKwp ? data[data.Length - 1] : data[1];
                    _currentSession = (UdsSession)session;
                    _securityUnlocked = false; 
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), session };
                    break;

                case UdsServiceId.SecurityAccess:
                    byte subFunction = isKwp ? data[data.Length - (payload.Length > 0 ? payload.Length : 0) - 1 + 1] : data[1]; 
                    // Simpler for both:
                    subFunction = isKwp ? (data.Length > (isKwp ? ((data[0] & 0x3F) != 0 ? 4 : 5) : 0) ? data[isKwp ? ((data[0] & 0x3F) != 0 ? 4 : 5) : 1] : data[data.Length-1]) : data[1];
                    // Let's refine the index for subfunction in KWP:
                    int subIndex = isKwp ? ((data[0] & 0x3F) != 0 ? 4 : 5) : 1;
                    if (subIndex >= data.Length) subIndex = data.Length - 1;
                    subFunction = data[subIndex];

                    if (subFunction % 2 != 0) 
                    {
                        _lastSeed = new byte[] { 0x12, 0x34, 0x56, 0x78 };
                        responsePayload = new byte[] { (byte)(serviceId + 0x40), subFunction, 0x12, 0x34, 0x56, 0x78 };
                    }
                    else 
                    {
                        bool valid = true;
                        int keyStart = subIndex + 1;
                        if (data.Length < keyStart + _lastSeed.Length) valid = false;
                        else
                        {
                            for (int i = 0; i < _lastSeed.Length; i++)
                            {
                                if (data[keyStart + i] != (byte)(_lastSeed[i] + 1))
                                {
                                    valid = false;
                                    break;
                                }
                            }
                        }

                        if (valid)
                        {
                            _securityUnlocked = true;
                            responsePayload = new byte[] { (byte)(serviceId + 0x40), subFunction };
                        }
                        else
                        {
                            return NegativeResponse(serviceId, UdsNrc.InvalidKey, isKwp);
                        }
                    }
                    break;

                case UdsServiceId.RequestDownload:
                    if (!_securityUnlocked) return NegativeResponse(serviceId, UdsNrc.SecurityAccessDenied, isKwp);
                    _lastDownloadAddress = (uint)((data[3] << 24) | (data[4] << 16) | (data[5] << 8) | data[6]);
                    _currentTransferOffset = 0;
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), 0x21, 0x01, 0x00 };
                    break;

                case UdsServiceId.TransferData:
                    int dataStart = isKwp ? (isKwp ? ((data[0] & 0x3F) != 0 ? 4 : 5) : 1) : 2;
                    if (data.Length > dataStart)
                    {
                        int transferSize = data.Length - dataStart;
                        for (int i = 0; i < transferSize; i++)
                        {
                            uint addr = _lastDownloadAddress + _currentTransferOffset + (uint)i;
                            if (addr < _simulatedMemory.Length)
                                _simulatedMemory[addr] = data[i + dataStart];
                        }
                        _currentTransferOffset += (uint)transferSize;
                    }
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), data.Length > 1 ? data[1] : (byte)0x01 };
                    break;

                case UdsServiceId.RequestTransferExit:
                    responsePayload = new byte[] { (byte)(serviceId + 0x40) };
                    break;

                case UdsServiceId.EcuReset:
                    _currentSession = UdsSession.Default;
                    _securityUnlocked = false;
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), data.Length > 1 ? data[1] : (byte)0x01 };
                    break;

                case UdsServiceId.TesterPresent:
                    responsePayload = new byte[] { (byte)(serviceId + 0x40), data.Length > 1 ? data[1] : (byte)0x00 };
                    break;

                case 0x21: // ReadDataByLocalIdentifier (KWP)
                    responsePayload = new byte[] { 0x61, data.Length > 1 ? data[data.Length-1] : (byte)0x01, 0x42, 0x43 }; // Dummy
                    break;

                case UdsServiceId.ReadMemoryByAddress:
                    if (data.Length < (isKwp ? 4 : 8)) return NegativeResponse(serviceId, UdsNrc.IncorrectMessageLengthOrInvalidFormat, isKwp);
                    
                    uint readAddress;
                    int readSize;

                    if (isKwp)
                    {
                        // KWP: 23 [Addr High] [Addr Mid] [Addr Low] [Size]
                        int addrIndex = (data[0] & 0x3F) != 0 ? 4 : 5;
                        readAddress = ((uint)data[addrIndex] << 16) | ((uint)data[addrIndex+1] << 8) | (uint)data[addrIndex+2];
                        readSize = data[addrIndex+3];
                    }
                    else
                    {
                        readAddress = ((uint)data[2] << 24) | ((uint)data[3] << 16) | ((uint)data[4] << 8) | (uint)data[5];
                        readSize = (data[6] << 8) | data[7];
                    }
                    
                    var readBuffer = new byte[1 + readSize];
                    readBuffer[0] = (byte)(serviceId + 0x40);
                    
                    for (int i = 0; i < readSize; i++)
                    {
                        uint addr = readAddress + (uint)i;
                        readBuffer[i + 1] = addr < _simulatedMemory.Length ? _simulatedMemory[addr] : (byte)0xFF;
                    }
                    responsePayload = readBuffer;
                    break;

                default:
                    return NegativeResponse(serviceId, UdsNrc.ServiceNotSupported, isKwp);
            }

            byte[] finalResponse;
            if (isKwp)
            {
                // Wrap in KWP header: [Format] [Target] [Source] [Payload...]
                finalResponse = new byte[3 + responsePayload.Length];
                finalResponse[0] = (byte)(0x80 | responsePayload.Length);
                finalResponse[1] = data[2]; // Target = Original Source
                finalResponse[2] = data[1]; // Source = Original Target
                Array.Copy(responsePayload, 0, finalResponse, 3, responsePayload.Length);
            }
            else
            {
                finalResponse = responsePayload;
            }

            _messageSubject.OnNext(finalResponse);
            return finalResponse;
        }

        return Array.Empty<byte>();
    }

    private byte[] NegativeResponse(byte serviceId, UdsNrc nrc, bool isKwp = false)
    {
        byte[] payload = new byte[] { 0x7F, serviceId, (byte)nrc };
        if (isKwp)
        {
            byte[] resp = new byte[6]; // Simplified KWP NRC
            resp[0] = 0x83;
            resp[1] = 0xF1; // Tester
            resp[2] = 0x01; // ECU
            resp[3] = 0x7F;
            resp[4] = serviceId;
            resp[5] = (byte)nrc;
            _messageSubject.OnNext(resp);
            return resp;
        }
        else
        {
            _messageSubject.OnNext(payload);
            return payload;
        }
    }

    public void SetVoltage(double voltage)
    {
        SimulatedVoltage = voltage;
    }

    public async Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default)
    {
        await Task.Delay(SimulatedLatencyMs, cancellationToken);
        return "OK";
    }

    public IObservable<byte[]> StreamMessages() => _messageSubject.AsObservable();

    public async Task<HardwareAdapterInfo> GetAdapterInfoAsync(CancellationToken cancellationToken = default)
    {
        return new HardwareAdapterInfo
        {
            Type = Type,
            Name = AdapterName,
            FirmwareVersion = "1.0.0-SIM",
            SerialNumber = "SIM-12345678",
            SupportedProtocols = new List<VehicleProtocol> { VehicleProtocol.ISO15765_CAN_11bit_500k },
            CanReadVoltage = true,
            CanPerformBiDirectional = true,
            MaxBaudRate = 1000000
        };
    }

    public async Task<double?> ReadBatteryVoltageAsync(CancellationToken cancellationToken = default)
    {
        await Task.Delay(5, cancellationToken);
        return SimulatedVoltage;
    }

    public async Task<bool> SetProtocolAsync(VehicleProtocol protocol, CancellationToken cancellationToken = default)
    {
        await Task.Delay(10, cancellationToken);
        return true;
    }

    public void Dispose()
    {
        _messageSubject.Dispose();
    }
}
