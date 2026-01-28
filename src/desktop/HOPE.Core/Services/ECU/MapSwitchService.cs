using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using HOPE.Core.Models;

namespace HOPE.Core.Services.ECU
{
    public class MapSwitchService
    {
        private readonly IHardwareAdapter _hardware;
        
        // Define standard profiles available for this platform
        private readonly List<MapSwitchProfile> _availableProfiles = new List<MapSwitchProfile>
        {
            new MapSwitchProfile { Id = 1, Name = "Economy", Description = "Fuel sipping mode. Reduced boost, lean cruise.", RevLimit = 4500, BoostTarget = 8.0 },
            new MapSwitchProfile { Id = 2, Name = "Normal", Description = "Standard daily driving.", RevLimit = 6500, BoostTarget = 15.0 },
            new MapSwitchProfile { Id = 3, Name = "Sport", Description = "Aggressive timing, higher boost.", RevLimit = 7200, BoostTarget = 22.0 },
            new MapSwitchProfile { Id = 4, Name = "Valet", Description = "Speed limited, minimal power.", RevLimit = 3000, BoostTarget = 0.0 },
            new MapSwitchProfile { Id = 5, Name = "Anti-Theft", Description = "No start / limp mode.", RevLimit = 0, BoostTarget = 0.0 }
        };

        // Custom UDS Identifier for Map Switching
        // Using a proprietary Data Identifier (DID) in the F1xx range typically used for ECU ID
        // Or a routine control. Let's use WriteDataByIdentifier 0xF1A0 for simplicity.
        private const ushort DID_ACTIVE_MAP = 0xF1A0;

        public MapSwitchService(IHardwareAdapter hardware)
        {
            _hardware = hardware;
        }

        public List<MapSwitchProfile> GetAvailableProfiles()
        {
            return _availableProfiles;
        }

        public async Task<MapSwitchProfile> GetCurrentProfileAsync(CancellationToken ct = default)
        {
            // 0x22 ReadDataByIdentifier
            byte[] request = new byte[] { 0x22, (byte)(DID_ACTIVE_MAP >> 8), (byte)(DID_ACTIVE_MAP & 0xFF) };
            
            // Standard timeout 1000ms
            byte[] response = await _hardware.SendMessageAsync(request, 1000, ct);

            if (response == null || response.Length < 4 || response[0] != 0x62) // 0x62 is positive response to 0x22
            {
                // Fallback or error
                return MapSwitchProfile.Default;
            }

            // Expected response: 62 F1 A0 [MapID]
            byte mapId = response[3];
            
            var profile = _availableProfiles.FirstOrDefault(p => p.Id == mapId);
            return profile ?? MapSwitchProfile.Default;
        }

        public async Task<bool> SwitchProfileAsync(byte profileId, CancellationToken ct = default)
        {
            var target = _availableProfiles.FirstOrDefault(p => p.Id == profileId);
            if (target == null) throw new ArgumentException("Invalid Profile ID");

            // 0x2E WriteDataByIdentifier
            byte[] request = new byte[] { 0x2E, (byte)(DID_ACTIVE_MAP >> 8), (byte)(DID_ACTIVE_MAP & 0xFF), profileId };

            byte[] response = await _hardware.SendMessageAsync(request, 1000, ct);

            // Positive response: 6E F1 A0
            if (response != null && response.Length >= 3 && response[0] == 0x6E)
            {
                return true;
            }

            return false;
        }
    }
}
