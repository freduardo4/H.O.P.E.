using System;
using System.Collections.Generic;
using System.Linq;

namespace HOPE.Core.Security;

public class SecurityAccessService
{
    private readonly List<ISecurityAccessProvider> _providers = new();

    public SecurityAccessService()
    {
        // Providers will be registered externally or added here
    }

    public void RegisterProvider(ISecurityAccessProvider provider)
    {
        if (!_providers.Contains(provider))
        {
            _providers.Add(provider);
        }
    }

    public byte[] GenerateKey(string ecuName, int securityLevel, byte[] seed)
    {
        var provider = _providers.FirstOrDefault(p => p.CanCalculateKey(ecuName, securityLevel));
        if (provider == null)
        {
            // Fallback or error
            // As a fallback for development, if the seed is mockable, return a mock key
            if (seed.Length > 0)
            {
                // Simple XOR fallback for testing/simulation
                return seed.Select(b => (byte)(b ^ 0xFF)).ToArray();
            }
            throw new InvalidOperationException($"No security provider found for ECU: {ecuName}, Level: {securityLevel}");
        }

        return provider.CalculateKey(ecuName, securityLevel, seed);
    }
}
