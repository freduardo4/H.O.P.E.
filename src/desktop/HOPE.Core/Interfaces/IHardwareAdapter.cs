namespace HOPE.Core.Interfaces;

/// <summary>
/// Hardware adapter abstraction for diagnostic communication.
/// Supports ELM327 (serial), J2534 (Pass-Thru), and mock adapters.
/// </summary>
public interface IHardwareAdapter : IDisposable
{
    /// <summary>
    /// Gets the hardware adapter type
    /// </summary>
    HardwareType Type { get; }

    /// <summary>
    /// Gets the adapter name/identifier
    /// </summary>
    string AdapterName { get; }

    /// <summary>
    /// Gets whether the adapter is currently connected
    /// </summary>
    bool IsConnected { get; }

    /// <summary>
    /// Gets whether the adapter supports high-frequency sampling (10-50Hz)
    /// </summary>
    bool SupportsHighFrequency { get; }

    /// <summary>
    /// Gets whether the adapter supports voltage monitoring
    /// </summary>
    bool SupportsVoltageMonitoring { get; }

    /// <summary>
    /// Gets whether the adapter supports bi-directional control
    /// </summary>
    bool SupportsBiDirectionalControl { get; }

    /// <summary>
    /// Gets whether the adapter reports quantized voltage (e.g., Scanmatik 7V/13.7V)
    /// </summary>
    bool HasQuantizedVoltageReporting { get; }

    /// <summary>
    /// Connect to the hardware adapter
    /// </summary>
    /// <param name="connectionString">Port name (COM3) or device ID</param>
    /// <param name="baudRate">Baud rate for serial adapters</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if connection successful</returns>
    Task<bool> ConnectAsync(string connectionString, int baudRate = 500000, CancellationToken cancellationToken = default);

    /// <summary>
    /// Disconnect from the hardware adapter
    /// </summary>
    Task DisconnectAsync();

    /// <summary>
    /// Send a message and receive response
    /// </summary>
    /// <param name="data">Message data bytes</param>
    /// <param name="timeout">Timeout in milliseconds</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response bytes</returns>
    Task<byte[]> SendMessageAsync(byte[] data, int timeout = 1000, CancellationToken cancellationToken = default);

    /// <summary>
    /// Send a raw command string (for ELM327 AT commands)
    /// </summary>
    /// <param name="command">Command string</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response string</returns>
    Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default);

    /// <summary>
    /// Stream incoming messages
    /// </summary>
    /// <returns>Observable stream of message bytes</returns>
    IObservable<byte[]> StreamMessages();

    /// <summary>
    /// Get adapter information
    /// </summary>
    /// <returns>Adapter info</returns>
    Task<HardwareAdapterInfo> GetAdapterInfoAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Read battery voltage (J2534 only)
    /// </summary>
    /// <returns>Voltage in volts, or null if not supported</returns>
    Task<double?> ReadBatteryVoltageAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Set programming voltage on a specific pin (J2534 only)
    /// </summary>
    /// <param name="pinNumber">OBD2 pin number</param>
    /// <param name="voltage">Voltage in volts (0 to 24)</param>
    /// <returns>True if successful</returns>
    Task<bool> SetProgrammingVoltageAsync(int pinNumber, double voltage, CancellationToken cancellationToken = default);

    /// <summary>
    /// Set the communication protocol
    /// </summary>
    /// <param name="protocol">Protocol to use</param>
    /// <returns>True if successful</returns>
    Task<bool> SetProtocolAsync(VehicleProtocol protocol, CancellationToken cancellationToken = default);

    /// <summary>
    /// Event raised when connection status changes
    /// </summary>
    event EventHandler<HardwareConnectionEventArgs>? ConnectionChanged;

    /// <summary>
    /// Event raised when an error occurs
    /// </summary>
    event EventHandler<HardwareErrorEventArgs>? ErrorOccurred;
}

/// <summary>
/// Types of hardware adapters
/// </summary>
public enum HardwareType
{
    /// <summary>ELM327 serial adapter</summary>
    ELM327,

    /// <summary>J2534 Pass-Thru device</summary>
    J2534,

    /// <summary>Mock adapter for testing</summary>
    Mock
}

/// <summary>
/// Vehicle communication protocols
/// </summary>
public enum VehicleProtocol
{
    /// <summary>Auto-detect protocol</summary>
    Auto = 0,

    /// <summary>J1850 PWM (Ford)</summary>
    J1850_PWM = 1,

    /// <summary>J1850 VPW (GM)</summary>
    J1850_VPW = 2,

    /// <summary>ISO 9141-2 (older European/Asian)</summary>
    ISO9141 = 3,

    /// <summary>ISO 14230-4 KWP2000 (5-baud init)</summary>
    ISO14230_KWP2000_5Baud = 4,

    /// <summary>ISO 14230-4 KWP2000 (fast init)</summary>
    ISO14230_KWP2000_Fast = 5,

    /// <summary>ISO 15765-4 CAN (11-bit ID, 500 kbaud)</summary>
    ISO15765_CAN_11bit_500k = 6,

    /// <summary>ISO 15765-4 CAN (29-bit ID, 500 kbaud)</summary>
    ISO15765_CAN_29bit_500k = 7,

    /// <summary>ISO 15765-4 CAN (11-bit ID, 250 kbaud)</summary>
    ISO15765_CAN_11bit_250k = 8,

    /// <summary>ISO 15765-4 CAN (29-bit ID, 250 kbaud)</summary>
    ISO15765_CAN_29bit_250k = 9,

    /// <summary>SAE J1939 (heavy-duty)</summary>
    SAE_J1939 = 10
}

/// <summary>
/// Hardware adapter information
/// </summary>
public class HardwareAdapterInfo
{
    public HardwareType Type { get; set; }
    public string Name { get; set; } = string.Empty;
    public string FirmwareVersion { get; set; } = string.Empty;
    public string SerialNumber { get; set; } = string.Empty;
    public List<VehicleProtocol> SupportedProtocols { get; set; } = new();
    public bool CanReadVoltage { get; set; }
    public bool CanPerformBiDirectional { get; set; }
    public int MaxBaudRate { get; set; }
}

/// <summary>
/// Event args for connection status changes
/// </summary>
public class HardwareConnectionEventArgs : EventArgs
{
    public bool IsConnected { get; }
    public string AdapterName { get; }
    public HardwareType AdapterType { get; }

    public HardwareConnectionEventArgs(bool isConnected, string adapterName, HardwareType adapterType)
    {
        IsConnected = isConnected;
        AdapterName = adapterName;
        AdapterType = adapterType;
    }
}

/// <summary>
/// Event args for hardware errors
/// </summary>
public class HardwareErrorEventArgs : EventArgs
{
    public string Message { get; }
    public HardwareErrorType ErrorType { get; }
    public Exception? Exception { get; }
    public int? ErrorCode { get; }

    public HardwareErrorEventArgs(string message, HardwareErrorType errorType, Exception? exception = null, int? errorCode = null)
    {
        Message = message;
        ErrorType = errorType;
        Exception = exception;
        ErrorCode = errorCode;
    }
}

/// <summary>
/// Types of hardware errors
/// </summary>
public enum HardwareErrorType
{
    ConnectionFailed,
    ConnectionLost,
    Timeout,
    InvalidResponse,
    ProtocolError,
    DeviceNotFound,
    DriverError,
    VoltageWarning,
    CommunicationError,
    Unknown
}
