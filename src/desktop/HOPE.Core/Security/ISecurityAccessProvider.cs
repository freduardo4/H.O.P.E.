namespace HOPE.Core.Security;

public interface ISecurityAccessProvider
{
    string Name { get; }
    bool CanCalculateKey(string ecuName, int securityLevel);
    byte[] CalculateKey(string ecuName, int securityLevel, byte[] seed);
}
