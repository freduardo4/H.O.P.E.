using System;
using System.Net.Http;
using System.Threading.Tasks;
using HOPE.Core.Services.Safety;
using HOPE.Core.Testing;
using HOPE.Core.Services.Protocols;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Services
{
    public class PreFlightServiceIntegrationTests
    {
        private readonly SimulatedHardwareAdapter _simulatedHardware;
        private readonly Mock<CloudSafetyService> _mockCloudSafety;
        private readonly PreFlightService _service;

        public PreFlightServiceIntegrationTests()
        {
            _simulatedHardware = new SimulatedHardwareAdapter();
            
            // Mock cloud safety to always approve for these tests
            _mockCloudSafety = new Mock<CloudSafetyService>(new HttpClient());
            _mockCloudSafety.Setup(c => c.ValidateFlashOperationAsync(It.IsAny<string>(), It.IsAny<double>(), It.IsAny<System.Threading.CancellationToken>()))
                .ReturnsAsync(true);

            _service = new PreFlightService(_mockCloudSafety.Object, _simulatedHardware);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenDisconneted()
        {
            // Adapter is disconnected by default
            var result = await _service.RunFullCheckAsync("ECU123");
            Assert.False(result.Success);
            Assert.Contains("not connected", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenVoltageLow()
        {
            await _simulatedHardware.ConnectAsync("COM1");
            _simulatedHardware.SimulatedVoltage = 11.0;

            var result = await _service.RunFullCheckAsync("ECU123");
            Assert.False(result.Success);
            Assert.Contains("Low battery voltage", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Succeeds_WhenConnectedAndVoltageGood()
        {
            await _simulatedHardware.ConnectAsync("COM1");
            _simulatedHardware.SimulatedVoltage = 12.8;

            var result = await _service.RunFullCheckAsync("ECU123");
            Assert.True(result.Success);
            Assert.Contains("Safe to proceed", result.Message);
        }
    }
}
