using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using Microsoft.Win32;

namespace HOPE.Core.Hardware;

public class J2534Adapter : IHardwareAdapter, IBenchPowerSupply
{
    private const int J1850VPW = 1;
    private const int J1850PWM = 2;
    private const int ISO9141 = 3;
    private const int ISO14230 = 4;
    private const int CAN = 5;
    private const int ISO15765 = 6;
    private const int CAN_29BIT_ID = 0x00000100;
    private const int ISO9141_NO_CHECKSUM = 0x00000200;
    private const int CAN_ID_BOTH = 0x00000800;
    private const int ISO9141_K_LINE_ONLY = 0x00001000;
    private const int PASS_FILTER = 0x01;
    private const int BLOCK_FILTER = 0x02;
    private const int FLOW_CONTROL_FILTER = 0x03;
    private const int GET_CONFIG = 0x01;
    private const int SET_CONFIG = 0x02;
    private const int READ_VBATT = 0x03;
    private const int FIVE_BAUD_INIT = 0x04;
    private const int FAST_INIT = 0x05;
    private const int CLEAR_TX_BUFFER = 0x07;
    private const int CLEAR_RX_BUFFER = 0x08;
    private const int STATUS_NOERROR = 0x00;
    private const int ERR_NOT_SUPPORTED = 0x01;
    private const int ERR_INVALID_CHANNEL_ID = 0x02;
    private const int ERR_INVALID_PROTOCOL_ID = 0x03;
    private const int ERR_NULL_PARAMETER = 0x04;
    private const int ERR_INVALID_IOCTL_VALUE = 0x05;
    private const int ERR_INVALID_FLAGS = 0x06;
    private const int ERR_FAILED = 0x07;
    private const int ERR_DEVICE_NOT_CONNECTED = 0x08;
    private const int ERR_TIMEOUT = 0x09;
    private const int ERR_INVALID_MSG = 0x0A;
    private const int ERR_INVALID_TIME_INTERVAL = 0x0B;
    private const int ERR_EXCEEDED_LIMIT = 0x0C;
    private const int ERR_INVALID_MSG_ID = 0x0D;
    private const int ERR_DEVICE_IN_USE = 0x0E;
    private const int ERR_INVALID_IOCTL_ID = 0x0F;
    private const int ERR_BUFFER_EMPTY = 0x10;
    private const int ERR_BUFFER_FULL = 0x11;
    private const int ERR_BUFFER_OVERFLOW = 0x12;
    private const int ERR_PIN_INVALID = 0x13;
    private const int ERR_CHANNEL_IN_USE = 0x14;
    private const int ERR_MSG_PROTOCOL_ID = 0x15;
    private const int ERR_INVALID_FILTER_ID = 0x16;
    private const int ERR_NO_FLOW_CONTROL = 0x17;
    private const int ERR_NOT_UNIQUE = 0x18;
    private const int ERR_INVALID_BAUDRATE = 0x19;
    private const int ERR_INVALID_DEVICE_ID = 0x1A;

    [StructLayout(LayoutKind.Sequential)]
    private struct PASSTHRU_MSG
    {
        public int ProtocolID;
        public int RxStatus;
        public int TxFlags;
        public int Timestamp;
        public int DataSize;
        public int ExtraDataIndex;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4128)]
        public byte[] Data;
    }

    private delegate int PassThruOpenDelegate(IntPtr pName, ref int pDeviceID);
    private delegate int PassThruCloseDelegate(int DeviceID);
    private delegate int PassThruConnectDelegate(int DeviceID, int ProtocolID, int Flags, int BaudRate, ref int pChannelID);
    private delegate int PassThruDisconnectDelegate(int ChannelID);
    private delegate int PassThruReadMsgsDelegate(int ChannelID, ref PASSTHRU_MSG pMsg, ref int pNumMsgs, int Timeout);
    private delegate int PassThruWriteMsgsDelegate(int ChannelID, ref PASSTHRU_MSG pMsg, ref int pNumMsgs, int Timeout);
    private delegate int PassThruStartMsgFilterDelegate(int ChannelID, int FilterType, ref PASSTHRU_MSG pMaskMsg, ref PASSTHRU_MSG pPatternMsg, ref PASSTHRU_MSG pFlowControlMsg, ref int pMsgID);
    private delegate int PassThruStopMsgFilterDelegate(int ChannelID, int MsgID);
    private delegate int PassThruSetProgrammingVoltageDelegate(int DeviceID, int PinNumber, int Voltage);
    private delegate int PassThruReadVersionDelegate(int DeviceID, StringBuilder pFirmwareVersion, StringBuilder pDllVersion, StringBuilder pApiVersion);
    private delegate int PassThruGetLastErrorDelegate(StringBuilder pErrorDescription);
    private delegate int PassThruIoctlDelegate(int ChannelID, int IoctlID, IntPtr pInput, IntPtr pOutput);

    private IntPtr _dllHandle = IntPtr.Zero;
    private int _deviceId;
    private int _channelId;
    private bool _isConnected;
    private bool _disposed;
    private string _dllPath = string.Empty;
    private string _adapterName = string.Empty;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private readonly Subject<byte[]> _messageSubject = new();
    private CancellationTokenSource? _streamCts;

    private PassThruOpenDelegate? _passThruOpen;
    private PassThruCloseDelegate? _passThruClose;
    private PassThruConnectDelegate? _passThruConnect;
    private PassThruDisconnectDelegate? _passThruDisconnect;
    private PassThruReadMsgsDelegate? _passThruReadMsgs;
    private PassThruWriteMsgsDelegate? _passThruWriteMsgs;
    private PassThruStartMsgFilterDelegate? _passThruStartMsgFilter;
    private PassThruStopMsgFilterDelegate? _passThruStopMsgFilter;
    private PassThruReadVersionDelegate? _passThruReadVersion;
    private PassThruGetLastErrorDelegate? _passThruGetLastError;
    private PassThruIoctlDelegate? _passThruIoctl;
    private PassThruSetProgrammingVoltageDelegate? _passThruSetProgrammingVoltage;

    public HardwareType Type => HardwareType.J2534;
    public string AdapterName => _adapterName;
    public bool IsConnected => _isConnected;
    public bool SupportsHighFrequency => true;
    public bool SupportsVoltageMonitoring => true;
    public bool SupportsBiDirectionalControl => true;
    public bool HasQuantizedVoltageReporting => _adapterName.Contains("Scanmatik", StringComparison.OrdinalIgnoreCase);

    public event EventHandler<HardwareConnectionEventArgs>? ConnectionChanged;
    public event EventHandler<HardwareErrorEventArgs>? ErrorOccurred;

    public static List<J2534DeviceInfo> GetInstalledDevices()
    {
        var devices = new List<J2534DeviceInfo>();
        try
        {
            using var key32 = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\WOW6432Node\PassThruSupport.04.04");
            if (key32 != null) devices.AddRange(EnumerateDevices(key32));
            using var key64 = Registry.LocalMachine.OpenSubKey(@"SOFTWARE\PassThruSupport.04.04");
            if (key64 != null) devices.AddRange(EnumerateDevices(key64));
        }
        catch { }
        return devices;
    }

    private static List<J2534DeviceInfo> EnumerateDevices(RegistryKey parentKey)
    {
        var devices = new List<J2534DeviceInfo>();
        foreach (var subKeyName in parentKey.GetSubKeyNames())
        {
            try
            {
                using var deviceKey = parentKey.OpenSubKey(subKeyName);
                if (deviceKey == null) continue;
                var device = new J2534DeviceInfo
                {
                    Name = deviceKey.GetValue("Name")?.ToString() ?? subKeyName,
                    Vendor = deviceKey.GetValue("Vendor")?.ToString() ?? "Unknown",
                    FunctionLibrary = deviceKey.GetValue("FunctionLibrary")?.ToString() ?? string.Empty,
                    ConfigApplication = deviceKey.GetValue("ConfigApplication")?.ToString()
                };
                var protocols = deviceKey.GetValue("Protocols")?.ToString();
                if (!string.IsNullOrEmpty(protocols)) device.SupportedProtocols = protocols.Split(',').Select(p => p.Trim()).ToList();
                if (!string.IsNullOrEmpty(device.FunctionLibrary)) devices.Add(device);
            }
            catch { }
        }
        return devices;
    }

    public async Task<bool> ConnectAsync(string connectionString, int baudRate = 500000, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (_isConnected) return true;
            _dllPath = connectionString;
            if (!connectionString.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
            {
                var devices = GetInstalledDevices();
                var device = devices.FirstOrDefault(d => d.Name.Equals(connectionString, StringComparison.OrdinalIgnoreCase));
                if (device == null) { RaiseError("Device not found", HardwareErrorType.DeviceNotFound); return false; }
                _dllPath = device.FunctionLibrary; _adapterName = device.Name;
            }
            else _adapterName = Path.GetFileNameWithoutExtension(_dllPath);

            _dllHandle = NativeLibrary.Load(_dllPath);
            if (_dllHandle == IntPtr.Zero) { RaiseError($"Failed to load J2534 DLL: {_dllPath}", HardwareErrorType.DriverError); return false; }

            if (!LoadFunctions()) { RaiseError("Failed to load J2534 API functions", HardwareErrorType.DriverError); return false; }

            _deviceId = 0;
            int result = _passThruOpen!(IntPtr.Zero, ref _deviceId);
            if (result != STATUS_NOERROR) { RaiseError($"PassThruOpen failed: {GetErrorMessage(result)}", HardwareErrorType.ConnectionFailed, null, result); return false; }

            _channelId = 0;
            result = _passThruConnect!(_deviceId, ISO15765, 0, baudRate, ref _channelId);
            if (result != STATUS_NOERROR) { _passThruClose!(_deviceId); RaiseError($"PassThruConnect failed: {GetErrorMessage(result)}", HardwareErrorType.ConnectionFailed, null, result); return false; }

            _isConnected = true;
            ConnectionChanged?.Invoke(this, new HardwareConnectionEventArgs(true, _adapterName, HardwareType.J2534));
            StartMessageStream();
            return true;
        }
        catch (Exception ex) { RaiseError($"Connection failed: {ex.Message}", HardwareErrorType.ConnectionFailed, ex); return false; }
        finally { _semaphore.Release(); }
    }

    private bool LoadFunctions()
    {
        try
        {
            _passThruOpen = Marshal.GetDelegateForFunctionPointer<PassThruOpenDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruOpen"));
            _passThruClose = Marshal.GetDelegateForFunctionPointer<PassThruCloseDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruClose"));
            _passThruConnect = Marshal.GetDelegateForFunctionPointer<PassThruConnectDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruConnect"));
            _passThruDisconnect = Marshal.GetDelegateForFunctionPointer<PassThruDisconnectDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruDisconnect"));
            _passThruReadMsgs = Marshal.GetDelegateForFunctionPointer<PassThruReadMsgsDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruReadMsgs"));
            _passThruWriteMsgs = Marshal.GetDelegateForFunctionPointer<PassThruWriteMsgsDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruWriteMsgs"));
            _passThruStartMsgFilter = Marshal.GetDelegateForFunctionPointer<PassThruStartMsgFilterDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruStartMsgFilter"));
            _passThruStopMsgFilter = Marshal.GetDelegateForFunctionPointer<PassThruStopMsgFilterDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruStopMsgFilter"));
            _passThruReadVersion = Marshal.GetDelegateForFunctionPointer<PassThruReadVersionDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruReadVersion"));
            _passThruGetLastError = Marshal.GetDelegateForFunctionPointer<PassThruGetLastErrorDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruGetLastError"));
            _passThruIoctl = Marshal.GetDelegateForFunctionPointer<PassThruIoctlDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruIoctl"));
            _passThruSetProgrammingVoltage = Marshal.GetDelegateForFunctionPointer<PassThruSetProgrammingVoltageDelegate>(NativeLibrary.GetExport(_dllHandle, "PassThruSetProgrammingVoltage"));
            return true;
        }
        catch { return false; }
    }

    public async Task DisconnectAsync()
    {
        await _semaphore.WaitAsync();
        try
        {
            _streamCts?.Cancel();
            if (_passThruDisconnect != null && _channelId != 0) _passThruDisconnect(_channelId);
            if (_passThruClose != null && _deviceId != 0) _passThruClose(_deviceId);
            if (_dllHandle != IntPtr.Zero) { NativeLibrary.Free(_dllHandle); _dllHandle = IntPtr.Zero; }
            _isConnected = false; 
            _deviceId = 0; 
            _channelId = 0;
            ConnectionChanged?.Invoke(this, new HardwareConnectionEventArgs(false, _adapterName, HardwareType.J2534));
        }
        finally { _semaphore.Release(); }
    }

    public async Task<byte[]> SendMessageAsync(byte[] data, int timeout = 1000, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (!_isConnected || _passThruWriteMsgs == null || _passThruReadMsgs == null) throw new InvalidOperationException("Not connected to J2534 device");
            var txMsg = new PASSTHRU_MSG { ProtocolID = ISO15765, TxFlags = 0, DataSize = data.Length + 4, Data = new byte[4128] };
            txMsg.Data[2] = 0x07; txMsg.Data[3] = 0xE0; Array.Copy(data, 0, txMsg.Data, 4, data.Length);
            int numMsgs = 1;
            int result = _passThruWriteMsgs(_channelId, ref txMsg, ref numMsgs, timeout);
            if (result != STATUS_NOERROR) throw new IOException($"PassThruWriteMsgs failed: {GetErrorMessage(result)}");
            var rxMsg = new PASSTHRU_MSG { Data = new byte[4128] }; numMsgs = 1;
            result = _passThruReadMsgs(_channelId, ref rxMsg, ref numMsgs, timeout);
            if (result == ERR_BUFFER_EMPTY || numMsgs == 0) return Array.Empty<byte>();
            if (result != STATUS_NOERROR) throw new IOException($"PassThruReadMsgs failed: {GetErrorMessage(result)}");
            int dataLength = rxMsg.DataSize - 4; if (dataLength <= 0) return Array.Empty<byte>();
            var responseData = new byte[dataLength]; Array.Copy(rxMsg.Data, 4, responseData, 0, dataLength);
            return responseData;
        }
        finally { _semaphore.Release(); }
    }

    public async Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default)
    {
        var bytes = ParseHexString(command);
        var response = await SendMessageAsync(bytes, 1000, cancellationToken);
        return BitConverter.ToString(response).Replace("-", " ");
    }

    private static byte[] ParseHexString(string hex)
    {
        hex = hex.Replace(" ", "").Replace("-", "");
        var bytes = new byte[hex.Length / 2];
        for (int i = 0; i < bytes.Length; i++) bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
        return bytes;
    }

    public IObservable<byte[]> StreamMessages() => _messageSubject.AsObservable();

    private void StartMessageStream()
    {
        _streamCts = new CancellationTokenSource();
        var token = _streamCts.Token;
        Task.Run(async () =>
        {
            while (!token.IsCancellationRequested && _isConnected)
            {
                try
                {
                    var rxMsg = new PASSTHRU_MSG { Data = new byte[4128] }; int numMsgs = 1;
                    int result = _passThruReadMsgs!(_channelId, ref rxMsg, ref numMsgs, 100);
                    if (result == STATUS_NOERROR && numMsgs > 0 && rxMsg.DataSize > 4)
                    {
                        var data = new byte[rxMsg.DataSize - 4]; Array.Copy(rxMsg.Data, 4, data, 0, data.Length);
                        _messageSubject.OnNext(data);
                    }
                    await Task.Delay(10, token);
                }
                catch (OperationCanceledException) { break; }
                catch { }
            }
        }, token);
    }

    public async Task<HardwareAdapterInfo> GetAdapterInfoAsync(CancellationToken cancellationToken = default)
    {
        var info = new HardwareAdapterInfo
        {
            Type = HardwareType.J2534, Name = _adapterName, CanReadVoltage = true, CanPerformBiDirectional = true, MaxBaudRate = 1000000,
            SupportedProtocols = new List<VehicleProtocol> { VehicleProtocol.ISO15765_CAN_11bit_500k, VehicleProtocol.ISO15765_CAN_29bit_500k, VehicleProtocol.ISO14230_KWP2000_Fast, VehicleProtocol.J1850_PWM, VehicleProtocol.J1850_VPW }
        };
        if (_isConnected && _passThruReadVersion != null)
        {
            var firmware = new StringBuilder(80); var dll = new StringBuilder(80); var api = new StringBuilder(80);
            if (_passThruReadVersion(_deviceId, firmware, dll, api) == STATUS_NOERROR) info.FirmwareVersion = firmware.ToString();
        }
        return await Task.FromResult(info);
    }

    public async Task<double?> ReadBatteryVoltageAsync(CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (!_isConnected || _passThruIoctl == null) return null;
            IntPtr voltagePtr = Marshal.AllocHGlobal(sizeof(int));
            try
            {
                if (_passThruIoctl(_channelId, READ_VBATT, IntPtr.Zero, voltagePtr) == STATUS_NOERROR) return Marshal.ReadInt32(voltagePtr) / 1000.0;
                return null;
            }
            finally { Marshal.FreeHGlobal(voltagePtr); }
        }
        finally { _semaphore.Release(); }
    }

    public async Task<bool> SetProtocolAsync(VehicleProtocol protocol, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (!_isConnected || _passThruDisconnect == null || _passThruConnect == null) return false;
            _passThruDisconnect(_channelId);
            int j2534Protocol; int flags = 0; int baudRate;
            switch (protocol)
            {
                case VehicleProtocol.J1850_PWM: j2534Protocol = J1850PWM; baudRate = 41600; break;
                case VehicleProtocol.J1850_VPW: j2534Protocol = J1850VPW; baudRate = 10400; break;
                case VehicleProtocol.ISO9141: j2534Protocol = ISO9141; baudRate = 10400; break;
                case VehicleProtocol.ISO14230_KWP2000_5Baud:
                case VehicleProtocol.ISO14230_KWP2000_Fast: j2534Protocol = ISO14230; baudRate = 10400; break;
                case VehicleProtocol.ISO15765_CAN_11bit_500k: j2534Protocol = ISO15765; baudRate = 500000; break;
                case VehicleProtocol.ISO15765_CAN_29bit_500k: j2534Protocol = ISO15765; flags = CAN_29BIT_ID; baudRate = 500000; break;
                case VehicleProtocol.ISO15765_CAN_11bit_250k: j2534Protocol = ISO15765; baudRate = 250000; break;
                case VehicleProtocol.ISO15765_CAN_29bit_250k: j2534Protocol = ISO15765; flags = CAN_29BIT_ID; baudRate = 250000; break;
                default: j2534Protocol = ISO15765; baudRate = 500000; break;
            }
            return _passThruConnect(_deviceId, j2534Protocol, flags, baudRate, ref _channelId) == STATUS_NOERROR;
        }
        finally { _semaphore.Release(); }
    }

    public async Task<bool> SetProgrammingVoltageAsync(int pinNumber, double voltage, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (!_isConnected || _passThruSetProgrammingVoltage == null) return false;
            return _passThruSetProgrammingVoltage(_deviceId, pinNumber, (int)(voltage * 1000)) == STATUS_NOERROR;
        }
        finally { _semaphore.Release(); }
    }

    // IBenchPowerSupply Implementation
    public bool CanControlPower => false; // Standard J2534 usually only controls ignition/programming voltage, not main power.

    public async Task<bool> SetIgnitionAsync(bool on, CancellationToken ct = default)
    {
        // For generic J2534, we treat "Ignition" as setting Programming Voltage on Pin 12 (Ford FEPS style)
        // or potentially Pin 1 (Generic OEM). We default to Pin 12 as it's the most common "switched" pin available via API.
        // Some specialized J2534 devices (like Scanmatik) map SetProgrammingVoltage to their Bench cable ignition pin.
        
        const int IGNITION_PIN = 12; 
        double voltage = on ? 13.5 : 0.0; // 13.5V for "ON", 0V for "OFF"
        
        return await SetProgrammingVoltageAsync(IGNITION_PIN, voltage, ct);
    }

    public Task<bool> SetPowerAsync(bool on, CancellationToken ct = default)
    {
        // Not supported by standard J2534 API
        return Task.FromResult(false);
    }

    private string GetErrorMessage(int errorCode)
    {
        if (_passThruGetLastError != null) { var errorMsg = new StringBuilder(256); _passThruGetLastError(errorMsg); return errorMsg.ToString(); }
        return errorCode switch { ERR_NOT_SUPPORTED => "Feature not supported", ERR_INVALID_CHANNEL_ID => "Invalid channel ID", ERR_INVALID_PROTOCOL_ID => "Invalid protocol ID", ERR_NULL_PARAMETER => "Null parameter", ERR_FAILED => "Operation failed", ERR_DEVICE_NOT_CONNECTED => "Device not connected", ERR_TIMEOUT => "Timeout", ERR_INVALID_MSG => "Invalid message", ERR_DEVICE_IN_USE => "Device in use", ERR_BUFFER_EMPTY => "Buffer empty", ERR_BUFFER_FULL => "Buffer full", _ => $"Error code: 0x{errorCode:X2}" };
    }

    private void RaiseError(string message, HardwareErrorType errorType, Exception? ex = null, int? errorCode = null) => ErrorOccurred?.Invoke(this, new HardwareErrorEventArgs(message, errorType, ex, errorCode));

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true; 
        _streamCts?.Cancel(); 
        _messageSubject.Dispose();
        if (_isConnected) DisconnectAsync().GetAwaiter().GetResult();
        _semaphore.Dispose();
    }
}

public class J2534DeviceInfo
{
    public string Name { get; set; } = string.Empty;
    public string Vendor { get; set; } = string.Empty;
    public string FunctionLibrary { get; set; } = string.Empty;
    public string? ConfigApplication { get; set; }
    public List<string> SupportedProtocols { get; set; } = new();
}
