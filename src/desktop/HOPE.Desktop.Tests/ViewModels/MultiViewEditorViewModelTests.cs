using Xunit;
using Moq;
using HOPE.Desktop.ViewModels;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.ECU;
using Microsoft.Extensions.Logging;
using System.Threading.Tasks;
using System.Collections.Generic;
using System;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;

namespace HOPE.Desktop.Tests.ViewModels;

public class MultiViewEditorViewModelTests
{
    private readonly Mock<ICalibrationRepository> _mockRepo;
    private readonly Mock<ILogger<MultiViewEditorViewModel>> _mockLogger;

    public MultiViewEditorViewModelTests()
    {
        _mockRepo = new Mock<ICalibrationRepository>();
        _mockLogger = new Mock<ILogger<MultiViewEditorViewModel>>();
    }

    [Fact]
    public async Task LoadAsync_ShouldCallGetCalibration_WhenHashProvided()
    {
        // Arrange
        var commitHash = "abcdef";
        var mockCal = new CalibrationFile 
        { 
            EcuId = "ECU123", 
            Blocks = new List<CalibrationBlock>() 
        };

        _mockRepo.Setup(r => r.GetCalibrationAsync(commitHash, It.IsAny<System.Threading.CancellationToken>()))
                 .ReturnsAsync(mockCal);

        var viewModel = new MultiViewEditorViewModel(_mockRepo.Object, _mockLogger.Object);

        // Act
        await viewModel.LoadAsync(commitHash);

        // Assert
        _mockRepo.Verify(r => r.GetCalibrationAsync(commitHash, It.IsAny<System.Threading.CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task Save_ShouldCallStageAndCommit_WhenCalibrationLoaded()
    {
        // Arrange
        var viewModel = new MultiViewEditorViewModel(_mockRepo.Object, _mockLogger.Object);
        // By default Save assumes demo calibration if none loaded

        // Act
        viewModel.SaveCommand.Execute(null);
        await Task.Delay(50); // Small wait for async command

        // Assert
        _mockRepo.Verify(r => r.StageAsync(It.IsAny<CalibrationFile>(), It.IsAny<System.Threading.CancellationToken>()), Times.Once);
        _mockRepo.Verify(r => r.CommitAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<System.Threading.CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task LoadAsync_ShouldLogerror_WhenRepoFails()
    {
        // Arrange
        _mockRepo.Setup(r => r.GetCalibrationAsync(It.IsAny<string>(), It.IsAny<System.Threading.CancellationToken>()))
                 .ThrowsAsync(new Exception("Repo failure"));
        
        var viewModel = new MultiViewEditorViewModel(_mockRepo.Object, _mockLogger.Object);

        // Act
        await viewModel.LoadAsync("hash");

        // Assert
        // Verify logger error was called
        _mockLogger.Verify(
            x => x.Log(
                LogLevel.Error,
                It.IsAny<EventId>(),
                It.Is<It.IsAnyType>((v, t) => true),
                It.IsAny<Exception>(),
                It.Is<Func<It.IsAnyType, Exception?, string>>((v, t) => true)),
            Times.Once);
    }

    [Fact]
    public async Task OpenAxisEditorCommand_ShouldRescaleMapAndCleanStatus()
    {
        // Arrange
        var viewModel = new MultiViewEditorViewModel(_mockRepo.Object, _mockLogger.Object);
        await viewModel.LoadAsync("dummy"); // Initialize mock data

        // Act
        viewModel.OpenAxisEditorCommand.Execute(null);

        // Assert
        Assert.Equal("Axis rescaled successfully", viewModel.StatusMessage);
        // Verify some views were updated (e.g. TabularData)
        Assert.NotNull(viewModel.TabularData);
    }

    [Fact]
    public async Task OnSelectedRowChanged_ShouldUpdateChartSeries()
    {
        // Arrange
        var viewModel = new MultiViewEditorViewModel(_mockRepo.Object, _mockLogger.Object);
        await viewModel.LoadAsync("dummy");

        // Act
        viewModel.SelectedRow = viewModel.AxisRows[1];

        // Assert
        Assert.NotEmpty(viewModel.ChartSeries);
        var series = viewModel.ChartSeries[0] as LineSeries<double>;
        Assert.NotNull(series);
        Assert.Equal(16, series.Values.Count());
    }
}
