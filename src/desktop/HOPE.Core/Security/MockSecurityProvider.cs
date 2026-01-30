using System;

namespace HOPE.Core.Security;

public class MockSecurityProvider : ISecurityAccessProvider
{
    public string Name => "Mock Provider";

    public bool CanCalculateKey(string ecuName, int securityLevel)
    {
        // For testing, always say yes to "Generic" or "MockECU"
        return ecuName.Equals("Generic", StringComparison.OrdinalIgnoreCase) || 
               ecuName.Equals("MockECU", StringComparison.OrdinalIgnoreCase);
    }

    public byte[] CalculateKey(string ecuName, int securityLevel, byte[] seed)
    {
        // Simple mock algorithm: XOR with 0xFF (Bitwise NOT)
        // This is deterministic and easy to verify
        byte[] key = new byte[seed.Length];
        for (int i = 0; i < seed.Length; i++)
        {
            key[i] = (byte)(seed[i] ^ 0xFF);
        }
        return key;
    }
}
