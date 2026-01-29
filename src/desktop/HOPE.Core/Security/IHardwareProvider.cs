using System;

namespace HOPE.Core.Security
{
    /// <summary>
    /// Provides hardware identification for cryptographic binding.
    /// </summary>
    public interface IHardwareProvider
    {
        /// <summary>
        /// Gets a stable unique hardware identifier for the current machine.
        /// </summary>
        string GetHardwareId();

        /// <summary>
        /// Validates if the current hardware matches the expected ID.
        /// </summary>
        bool IsHardwareMatch(string expectedId);
    }
}
