namespace HOPE.Core.Protocols;

/// <summary>
/// Interface for vehicle diagnostic protocols (UDS, KWP2000).
/// </summary>
public interface IDiagnosticProtocol
{
    string Name { get; }
    
    /// <summary>
    /// Initialize the diagnostic session
    /// </summary>
    Task<bool> StartSessionAsync(byte sessionType);

    /// <summary>
    /// Perform security access (unlocking ECU for reading/writing)
    /// </summary>
    Task<bool> SecurityAccessAsync(byte accessType, byte[]? keyOverride = null);

    /// <summary>
    /// Read a specific memory address range
    /// </summary>
    Task<byte[]> ReadMemoryAsync(long address, int length);

    /// <summary>
    /// Send a low-level diagnostic request and receive response
    /// </summary>
    Task<byte[]> SendRequestAsync(byte serviceId, byte[] data);
}
