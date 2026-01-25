using System.Reactive;
using HOPE.Core.Models;

namespace HOPE.Core.Services.OBD;

/// <summary>
/// Interface for OBD2 diagnostic operations.
/// Supports ELM327 and potentially other adapters in the future.
/// </summary>
public interface IOBD2Service
{
    /// <summary>
    /// Gets the current connection status
    /// </summary>
    bool IsConnected { get; }

    /// <summary>
    /// Gets the connected adapter type
    /// </summary>
    string AdapterType { get; }

    /// <summary>
    /// Gets the detected vehicle protocol
    /// </summary>
    string? DetectedProtocol { get; }

    /// <summary>
    /// Connect to OBD2 adapter on specified port
    /// </summary>
    /// <param name="portName">COM port name (e.g., "COM3")</param>
    /// <param name="baudRate">Baud rate (typically 9600 for ELM327)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if connection successful</returns>
    Task<bool> ConnectAsync(string portName, int baudRate = 9600, CancellationToken cancellationToken = default);

    /// <summary>
    /// Disconnect from OBD2 adapter
    /// </summary>
    Task DisconnectAsync();

    /// <summary>
    /// Get list of available COM ports
    /// </summary>
    /// <returns>List of port names</returns>
    string[] GetAvailablePorts();

    /// <summary>
    /// Send a raw command to the OBD2 adapter
    /// </summary>
    /// <param name="command">Command string (e.g., "ATZ", "0100")</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response from adapter</returns>
    Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default);

    /// <summary>
    /// Get list of PIDs supported by the vehicle
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of supported PIDs</returns>
    Task<List<string>> GetSupportedPIDsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Read a single OBD2 parameter
    /// </summary>
    /// <param name="pid">Parameter ID (e.g., "0C" for RPM)</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>OBD2 reading with parsed value</returns>
    Task<OBD2Reading> ReadPIDAsync(string pid, CancellationToken cancellationToken = default);

    /// <summary>
    /// Read multiple OBD2 parameters
    /// </summary>
    /// <param name="pids">List of PIDs to read</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of OBD2 readings</returns>
    Task<List<OBD2Reading>> ReadPIDsAsync(IEnumerable<string> pids, CancellationToken cancellationToken = default);

    /// <summary>
    /// Stream OBD2 data continuously
    /// </summary>
    /// <param name="pids">PIDs to stream</param>
    /// <param name="intervalMs">Polling interval in milliseconds</param>
    /// <param name="cancellationToken">Cancellation token to stop streaming</param>
    /// <returns>Observable stream of OBD2 readings</returns>
    IObservable<OBD2Reading> StreamPIDs(IEnumerable<string> pids, int intervalMs = 200, CancellationToken cancellationToken = default);

    /// <summary>
    /// Read Diagnostic Trouble Codes (DTCs)
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>List of DTCs</returns>
    Task<List<DiagnosticTroubleCode>> ReadDTCsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Clear Diagnostic Trouble Codes
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>True if successful</returns>
    Task<bool> ClearDTCsAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Get VIN (Vehicle Identification Number)
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>17-character VIN</returns>
    Task<string?> GetVINAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Get ECU information
    /// </summary>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>ECU info string</returns>
    Task<string?> GetECUInfoAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Event raised when connection status changes
    /// </summary>
    event EventHandler<bool>? ConnectionStatusChanged;

    /// <summary>
    /// Event raised when data is received
    /// </summary>
    event EventHandler<OBD2Reading>? DataReceived;

    /// <summary>
    /// Event raised when an error occurs
    /// </summary>
    event EventHandler<OBD2ErrorEventArgs>? ErrorOccurred;
}

/// <summary>
/// Event args for OBD2 errors
/// </summary>
public class OBD2ErrorEventArgs : EventArgs
{
    public string ErrorMessage { get; set; } = string.Empty;
    public Exception? Exception { get; set; }
    public OBD2ErrorType ErrorType { get; set; }

    public OBD2ErrorEventArgs(string message, OBD2ErrorType errorType, Exception? exception = null)
    {
        ErrorMessage = message;
        ErrorType = errorType;
        Exception = exception;
    }
}

/// <summary>
/// Types of OBD2 errors
/// </summary>
public enum OBD2ErrorType
{
    ConnectionLost,
    TimeoutError,
    InvalidResponse,
    UnsupportedPID,
    AdapterError,
    VehicleNotResponding,
    ProtocolError,
    CommunicationError,
    Unknown
}

/// <summary>
/// OBD2 adapter information
/// </summary>
public class OBD2AdapterInfo
{
    public string AdapterType { get; set; } = string.Empty;
    public string FirmwareVersion { get; set; } = string.Empty;
    public string ProtocolVersion { get; set; } = string.Empty;
    public List<string> SupportedProtocols { get; set; } = new();
    public string DeviceIdentifier { get; set; } = string.Empty;
}
