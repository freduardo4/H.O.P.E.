using HOPE.Core.Services.OBD;
using HOPE.Core.Security;

namespace HOPE.Core.Protocols;

public class UDSProtocol : IDiagnosticProtocol
{
    private readonly IOBD2Service _obdService;
    private readonly SecurityAccessService _securityService;
    
    public string Name => "UDS (ISO 14229)";

    public UDSProtocol(IOBD2Service obdService, SecurityAccessService securityService)
    {
        _obdService = obdService;
        _securityService = securityService;
    }

    public async Task<bool> StartSessionAsync(byte sessionType)
    {
        // UDS Service 0x10: DiagnosticSessionControl
        byte[] request = new byte[] { sessionType };
        var response = await SendRequestAsync(0x10, request);
        
        // Positive response is ServiceID + 0x40
        return response.Length > 0 && response[0] == 0x50;
    }

    public async Task<bool> SecurityAccessAsync(byte accessType, byte[]? keyOverride = null)
    {
        // UDS Service 0x27: SecurityAccess
        // Step 1: Request Seed
        byte[] seedRequest = new byte[] { accessType };
        var seedResponse = await SendRequestAsync(0x27, seedRequest);
        
        if (seedResponse.Length < 2 || seedResponse[0] != 0x67) return false;

        // Extract seed (skip service ID + subfunction)
        // Response format: [0x67] [SecurityLevel] [Seed...]
        // Actually usually [0x67] [AccessType] [Seed...]
        byte[] seed = seedResponse.Skip(2).ToArray();

        // Step 2: Calculate Key
        byte[] key;
        if (keyOverride != null)
        {
            key = keyOverride;
        }
        else
        {
            try 
            {
                // We don't have ECU name context here yet, defaulting to "Generic" or passed via constructor later
                key = _securityService.GenerateKey("Generic", accessType, seed);
            }
            catch
            {
                return false;
            }
        }

        // Step 3: Send Key
        byte[] keyRequest = new byte[key.Length + 1];
        keyRequest[0] = (byte)(accessType + 1); // SendKey subfunction
        Array.Copy(key, 0, keyRequest, 1, key.Length);
        
        var unlockResponse = await SendRequestAsync(0x27, keyRequest);
        return unlockResponse.Length > 0 && unlockResponse[0] == 0x67;
    }

    public async Task<byte[]> ReadMemoryAsync(long address, int length)
    {
        // UDS Service 0x23: ReadMemoryByAddress
        // Simplified implementation: assuming 4-byte address and 2-byte length
        byte[] data = new byte[7];
        data[0] = 0x24; // AddressAndLengthFormatIdentifier (4-byte addr, 2-byte len)
        
        // Address
        data[1] = (byte)((address >> 24) & 0xFF);
        data[2] = (byte)((address >> 16) & 0xFF);
        data[3] = (byte)((address >> 8) & 0xFF);
        data[4] = (byte)(address & 0xFF);
        
        // Length
        data[5] = (byte)((length >> 8) & 0xFF);
        data[6] = (byte)(length & 0xFF);

        var response = await SendRequestAsync(0x23, data);
        
        if (response.Length > 0 && response[0] == 0x63)
        {
            return response.Skip(1).ToArray();
        }
        
        return Array.Empty<byte>();
    }

    public async Task<byte[]> SendRequestAsync(byte serviceId, byte[] data)
    {
        // Convert to hex string for ELM327
        string hexReq = serviceId.ToString("X2") + string.Concat(data.Select(b => b.ToString("X2")));
        
        // Send via OBD2 service
        string responseHex = await _obdService.SendCommandAsync(hexReq);
        
        // Basic parsing - very simplified for skeleton
        try
        {
            return responseHex.Split(' ')
                .Where(s => !string.IsNullOrWhiteSpace(s))
                .Select(s => Convert.ToByte(s, 16))
                .ToArray();
        }
        catch
        {
            return Array.Empty<byte>();
        }
    }
}
