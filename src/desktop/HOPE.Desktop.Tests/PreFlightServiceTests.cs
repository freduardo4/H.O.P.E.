using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.Safety;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests
{
    public class PreFlightServiceTests
    {
        private readonly Mock<IHardwareAdapter> _mockHardware;
        private readonly Mock<CloudSafetyService> _mockCloudSafety;
        private readonly PreFlightService _service;

        public PreFlightServiceTests()
        {
            _mockHardware = new Mock<IHardwareAdapter>();
            // Mocking a concrete class requires it to have a public constructor (it does)
            // and we rely on virtual methods for behavior override.
            _mockCloudSafety = new Mock<CloudSafetyService>(new HttpClient()); 
            
            _service = new PreFlightService(_mockCloudSafety.Object, _mockHardware.Object);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenHardwareDisconnected()
        {
            _mockHardware.Setup(h => h.IsConnected).Returns(false);

            var result = await _service.RunFullCheckAsync("ECU123");

            Assert.False(result.Success);
            Assert.Contains("not connected", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenVoltageUnknown()
        {
            _mockHardware.Setup(h => h.IsConnected).Returns(true);
            _mockHardware.Setup(h => h.ReadBatteryVoltageAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync((double?)null);

            var result = await _service.RunFullCheckAsync("ECU123");

            Assert.False(result.Success);
            Assert.Contains("Could not read battery voltage", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenVoltageTooLow()
        {
            _mockHardware.Setup(h => h.IsConnected).Returns(true);
            _mockHardware.Setup(h => h.ReadBatteryVoltageAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync(11.5); // Low voltage

            var result = await _service.RunFullCheckAsync("ECU123");

            Assert.False(result.Success);
            Assert.Contains("Low battery voltage", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Fails_WhenCloudRejects()
        {
            _mockHardware.Setup(h => h.IsConnected).Returns(true);
            _mockHardware.Setup(h => h.ReadBatteryVoltageAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync(13.5); // Good voltage

            _mockCloudSafety.Setup(c => c.ValidateFlashOperationAsync(It.IsAny<string>(), It.IsAny<double>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(false);

            var result = await _service.RunFullCheckAsync("ECU123");

            Assert.False(result.Success);
            Assert.Contains("Cloud safety policy rejected", result.Message);
        }

        [Fact]
        public async Task RunFullCheck_Succeeds_WhenAllChecksPass()
        {
            _mockHardware.Setup(h => h.IsConnected).Returns(true);
            _mockHardware.Setup(h => h.ReadBatteryVoltageAsync(It.IsAny<CancellationToken>()))
                .ReturnsAsync(13.5); // Good voltage

            _mockCloudSafety.Setup(c => c.ValidateFlashOperationAsync(It.IsAny<string>(), It.IsAny<double>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(true);

            var result = await _service.RunFullCheckAsync("ECU123");

            Assert.True(result.Success);
            Assert.Contains("Safe to proceed", result.Message);
        }
    }
}
