using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.AI;
using HOPE.Core.Services.Database;
using HOPE.Core.Services.OBD;
using HOPE.Desktop.ViewModels;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.ViewModels;

public class DashboardViewModelTests
{
    private readonly Mock<IOBD2Service> _mockObdService;
    private readonly Mock<IDatabaseService> _mockDbService;
    private readonly Mock<IAnomalyService> _mockAnomalyService;
    private readonly Subject<OBD2Reading> _obdStreamSubject;
    private readonly DashboardViewModel _viewModel;

    public DashboardViewModelTests()
    {
        _mockObdService = new Mock<IOBD2Service>();
        _mockDbService = new Mock<IDatabaseService>();
        _mockAnomalyService = new Mock<IAnomalyService>();
        _obdStreamSubject = new Subject<OBD2Reading>();

        // Setup OBD service mock
        _mockObdService.Setup(s => s.StreamPIDs(
                It.IsAny<IEnumerable<string>>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .Returns(_obdStreamSubject);
        
        _mockObdService.Setup(s => s.ConnectAsync(
                It.IsAny<string>(),
                It.IsAny<int>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(true);
        _mockObdService.SetupGet(s => s.IsConnected).Returns(false);

        // Setup DB service mock
        _mockDbService.Setup(s => s.StartSessionAsync(It.IsAny<Guid>()))
            .ReturnsAsync(Guid.NewGuid());

        _viewModel = new DashboardViewModel(
            _mockObdService.Object,
            _mockDbService.Object,
            _mockAnomalyService.Object);
    }

    [Fact]
    public void Constructor_InitializesPropertiesWithDefaults()
    {
        Assert.False(_viewModel.IsStreaming);
        Assert.Equal("Disconnected", _viewModel.ConnectionStatus);
        Assert.True(_viewModel.IsChartVisible);
        Assert.Empty(_viewModel.ContributingParameters);
        Assert.Empty(_viewModel.AnomalyHistory);
    }

    [Fact]
    public void UpdateConnectionStatus_UpdatesBasedOnServiceStatus()
    {
        // Arrange
        _mockObdService.SetupGet(s => s.IsConnected).Returns(true);
        _mockObdService.SetupGet(s => s.AdapterType).Returns("TestAdapter");

        // Act - Simulate event
        _mockObdService.Raise(s => s.ConnectionStatusChanged += null, null, true);

        // Assert
        Assert.Contains("Connected", _viewModel.ConnectionStatus);
        Assert.Contains("TestAdapter", _viewModel.ConnectionStatus);
    }

    [Fact]
    public async Task ToggleStreamingAsync_StartsStreaming_WhenNotStreaming()
    {
        // Act
        await _viewModel.ToggleStreamingCommand.ExecuteAsync(null);

        // Assert
        Assert.True(_viewModel.IsStreaming);
        _mockObdService.Verify(s => s.ConnectAsync(
            It.IsAny<string>(), 
            It.IsAny<int>(), 
            It.IsAny<CancellationToken>()), Times.Once);
        _mockDbService.Verify(s => s.StartSessionAsync(It.IsAny<Guid>()), Times.Once);
        _mockObdService.Verify(s => s.StreamPIDs(
            It.IsAny<IEnumerable<string>>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ToggleStreamingAsync_StopsStreaming_WhenStreaming()
    {
        // Arrange - Start streaming first
        await _viewModel.ToggleStreamingCommand.ExecuteAsync(null);
        Assert.True(_viewModel.IsStreaming);

        // Act - Stop streaming
        await _viewModel.ToggleStreamingCommand.ExecuteAsync(null);

        // Assert
        Assert.False(_viewModel.IsStreaming);
        _mockDbService.Verify(s => s.EndSessionAsync(It.IsAny<Guid>()), Times.Once);
    }

    [Fact]
    public async Task OnReadingRecieved_UpdatesPropertiesAndCharts()
    {
        // Set sync context to force synchronous execution for ObserveOn
        SynchronizationContext.SetSynchronizationContext(new SynchronousSynchronizationContext());

        // Arrange
        await _viewModel.ToggleStreamingCommand.ExecuteAsync(null);
        var reading = new OBD2Reading
        {
            PID = OBD2PIDs.EngineRPM,
            Value = 3000,
            Timestamp = DateTime.UtcNow
        };

        // Act
        _obdStreamSubject.OnNext(reading);
        
        // Assert
        Assert.Equal(3000, _viewModel.EngineRpm);
        _mockDbService.Verify(s => s.LogReadingAsync(It.Is<OBD2Reading>(r => r.Value == 3000)), Times.Once);
    }

    private class SynchronousSynchronizationContext : SynchronizationContext
    {
        public override void Post(SendOrPostCallback d, object? state)
        {
            d(state);
        }
    }
}
