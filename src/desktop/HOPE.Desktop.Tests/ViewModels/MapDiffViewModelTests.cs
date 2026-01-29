using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.Export;
using HOPE.Core.Interfaces;
using HOPE.Desktop.ViewModels;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.ViewModels
{
    public class MapDiffViewModelTests
    {
        private readonly Mock<ICalibrationRepository> _mockRepo;
        private readonly Mock<IExportService> _mockExport;
        private readonly MapDiffViewModel _viewModel;

        public MapDiffViewModelTests()
        {
            _mockRepo = new Mock<ICalibrationRepository>();
            _mockExport = new Mock<IExportService>();
            _viewModel = new MapDiffViewModel(_mockRepo.Object, _mockExport.Object);
        }

        [Fact]
        public async Task LoadHistoryAsync_PopulatesAvailableCommits()
        {
            // Arrange
            var history = new List<CalibrationCommit>
            {
                new CalibrationCommit { Hash = "hash1", Message = "Commit 1", Timestamp = DateTime.UtcNow },
                new CalibrationCommit { Hash = "hash2", Message = "Commit 2", Timestamp = DateTime.UtcNow }
            };
            _mockRepo.Setup(r => r.GetHistoryAsync(It.IsAny<int>(), It.IsAny<CancellationToken>())).ReturnsAsync(history);

            // Act
            _viewModel.LoadHistoryCommand.Execute(null);

            // Assert
            Assert.Equal(2, _viewModel.AvailableCommits.Count);
            Assert.Equal("hash1", _viewModel.AvailableCommits[0].Hash);
        }

        [Fact]
        public async Task GenerateDiffAsync_SetsStatusMessageOnError()
        {
            // Arrange - No commits selected
            _viewModel.SelectedBaseCommit = null;

            // Act
            _viewModel.GenerateDiffCommand.Execute(null);

            // Assert
            Assert.Contains("Please select two commits", _viewModel.StatusMessage);
        }

        [Fact]
        public async Task GenerateDiffAsync_CalculatesDiffAndGeneratesSurfaces()
        {
            // Arrange
            _viewModel.SelectedBaseCommit = new CalibrationCommitItem { Hash = "base" };
            _viewModel.SelectedCompareCommit = new CalibrationCommitItem { Hash = "compare" };
            var mockDiff = new CalibrationDiff { BaseEcuId = "BCU", CompareEcuId = "CCU", Changes = new List<BlockChange>() };
            _mockRepo.Setup(r => r.DiffAsync("base", "compare", It.IsAny<CancellationToken>())).ReturnsAsync(mockDiff);

            // Act
            _viewModel.GenerateDiffCommand.Execute(null);

            // Assert
            Assert.Equal("Diff generated.", _viewModel.StatusMessage);
            Assert.NotEmpty(_viewModel.BaseSurfacePoints);
            Assert.NotEmpty(_viewModel.CompareSurfacePoints);
        }

        [Fact]
        public async Task ExportReportAsync_CallsExportService()
        {
            // Arrange - Generate diff first
            _viewModel.SelectedBaseCommit = new CalibrationCommitItem { Hash = "base" };
            _viewModel.SelectedCompareCommit = new CalibrationCommitItem { Hash = "compare" };
            _mockRepo.Setup(r => r.DiffAsync(It.IsAny<string>(), It.IsAny<string>(), It.IsAny<CancellationToken>())).ReturnsAsync(new CalibrationDiff());
            _mockExport.Setup(e => e.GetDefaultExportDirectory()).Returns("C:\\Temp");
            
            _viewModel.GenerateDiffCommand.Execute(null);

            // Act
            _viewModel.ExportReportCommand.Execute(null);

            // Assert
            _mockExport.Verify(e => e.ExportDiffReportAsync(It.IsAny<CalibrationDiff>(), It.IsAny<string>()), Times.Once);
            Assert.Contains("Report exported", _viewModel.StatusMessage);
        }
    }
}
