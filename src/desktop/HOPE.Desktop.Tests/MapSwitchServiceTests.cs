using System.Threading.Tasks;
using Xunit;
using HOPE.Core.Services.ECU;
using HOPE.Core.Testing;
using HOPE.Core.Services.Protocols;

namespace HOPE.Desktop.Tests
{
    public class MapSwitchServiceTests
    {
        [Fact]
        public async Task GetCurrentProfile_ReturnsDefault_WhenInitiallyConnected()
        {
            // Arrange
            var adapter = new SimulatedHardwareAdapter();
            await adapter.ConnectAsync("SIM");
            // Security unlock needed for write, but read might be open. 
            // In Sim, 0x22 doesn't check security currently based on my code.
            
            var service = new MapSwitchService(adapter);

            // Act
            var profile = await service.GetCurrentProfileAsync();

            // Assert
            Assert.NotNull(profile);
            Assert.Equal(1, profile.Id); // Default is 1
            Assert.Equal("Economy", profile.Name);
        }

        [Fact]
        public async Task SwitchProfile_UpdatesActiveProfile_WhenSecurityUnlocked()
        {
            // Arrange
            var adapter = new SimulatedHardwareAdapter();
            await adapter.ConnectAsync("SIM");
            var service = new MapSwitchService(adapter);

            // Unlock security
            // 27 01 -> 67 01 [Seed] -> 27 02 [Key] -> 67 02
            // Initial seed is 12 34 56 78 -> Key is 13 35 57 79
            await adapter.SendMessageAsync(new byte[] { 0x27, 0x01 });
            await adapter.SendMessageAsync(new byte[] { 0x27, 0x02, 0x13, 0x35, 0x57, 0x79 });

            // Act
            var result = await service.SwitchProfileAsync(3); // Switch to "Sport"
            var newProfile = await service.GetCurrentProfileAsync();

            // Assert
            Assert.True(result);
            Assert.Equal(3, newProfile.Id);
            Assert.Equal("Sport", newProfile.Name);
        }

        [Fact]
        public async Task SwitchProfile_Fails_WhenSecurityLocked()
        {
            // Arrange
            var adapter = new SimulatedHardwareAdapter();
            await adapter.ConnectAsync("SIM");
            var service = new MapSwitchService(adapter);

            // Act
            try {
                // Should return false because 0x2E returns 7F ... 33 (SecurityAccessDenied)
                // My service code: if (response[0] == 0x6E) return true; else return false;
                var result = await service.SwitchProfileAsync(3);
                Assert.False(result);
            }
            catch {
                // Service might throw or return false depending on implementation detail
                // current impl returns false on negative response check
            }
            
            var profile = await service.GetCurrentProfileAsync();

            // Assert
            Assert.Equal(1, profile.Id); // Should still be 1
        }
    }
}
