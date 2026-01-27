using HOPE.Core.Services.ECU;
using HOPE.Core.Services.AI;
using HOPE.Desktop.ViewModels;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.ViewModels;

public class MapVisualizationViewModelTests
{
    private readonly Mock<IECUService> _mockEcuService;
    private readonly Mock<ITuningOptimizerService> _mockTuningOptimizer;
    private readonly MapVisualizationViewModel _viewModel;

    public MapVisualizationViewModelTests()
    {
        _mockEcuService = new Mock<IECUService>();
        _mockTuningOptimizer = new Mock<ITuningOptimizerService>();

        // Setup default ECU service behavior
        _mockEcuService.Setup(s => s.ReadMapAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(CreateSampleMap());

        // Setup optimizer as available
        _mockTuningOptimizer.SetupGet(s => s.IsAvailable).Returns(true);
        _mockTuningOptimizer.SetupGet(s => s.OptimizerScriptPath).Returns("genetic_optimizer.py");

        _viewModel = new MapVisualizationViewModel(
            _mockEcuService.Object,
            null,
            _mockTuningOptimizer.Object);
    }

    [Fact]
    public void Constructor_InitializesPropertiesWithDefaults()
    {
        // Arrange & Act - constructor called in setup

        // Assert
        Assert.Equal("Ready to Read ECU", _viewModel.StatusMessage);
        Assert.False(_viewModel.IsBusy);
        Assert.False(_viewModel.IsDiffMode);
        Assert.False(_viewModel.IsOptimizing);
        Assert.True(_viewModel.IsOptimizerAvailable);
        Assert.Equal(50, _viewModel.Generations);
        Assert.Equal(50, _viewModel.PopulationSize);
        Assert.Equal(0.1, _viewModel.MutationRate);
        Assert.Equal(OptimizationObjective.AfrAccuracy, _viewModel.SelectedObjective);
    }

    [Fact]
    public void Constructor_WithNoOptimizer_SetsOptimizerNotAvailable()
    {
        // Arrange & Act
        var viewModel = new MapVisualizationViewModel(_mockEcuService.Object);

        // Assert
        Assert.False(viewModel.IsOptimizerAvailable);
    }

    [Fact]
    public async Task ReadEcuMapCommand_LoadsMapData()
    {
        // Act
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        // Assert
        Assert.Equal("Map Loaded Successfully", _viewModel.StatusMessage);
        Assert.False(_viewModel.IsBusy);
        Assert.False(_viewModel.IsDiffMode);
        Assert.Equal(8, _viewModel.MapData.Count); // 8 rows
        Assert.Equal(8, _viewModel.MapData[0].Values.Count); // 8 columns
        _mockEcuService.Verify(s => s.ReadMapAsync("FuelMap", It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ReadEcuMapCommand_HandlesError()
    {
        // Arrange
        _mockEcuService.Setup(s => s.ReadMapAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Connection failed"));

        // Act
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        // Assert
        Assert.Contains("Error:", _viewModel.StatusMessage);
        Assert.False(_viewModel.IsBusy);
    }

    [Fact]
    public async Task LoadCompareMapCommand_LoadsAndComputesDiff()
    {
        // Arrange - First load base map
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        // Act
        await _viewModel.LoadCompareMapCommand.ExecuteAsync(null);

        // Assert
        Assert.True(_viewModel.IsDiffMode);
        Assert.Contains("Diff View", _viewModel.StatusMessage);
        Assert.True(_viewModel.DiffData.Count > 0);
        Assert.True(_viewModel.TotalCells > 0);
    }

    [Fact]
    public async Task LoadCompareMapCommand_RequiresBaseMap()
    {
        // Act - Try to load compare map without base map
        await _viewModel.LoadCompareMapCommand.ExecuteAsync(null);

        // Assert
        Assert.Equal("Load base map first", _viewModel.StatusMessage);
        Assert.False(_viewModel.IsDiffMode);
    }

    [Fact]
    public async Task ToggleDiffModeCommand_TogglesBetweenModes()
    {
        // Arrange - Load both maps
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);
        await _viewModel.LoadCompareMapCommand.ExecuteAsync(null);
        Assert.True(_viewModel.IsDiffMode);

        // Act
        _viewModel.ToggleDiffModeCommand.Execute(null);

        // Assert
        Assert.False(_viewModel.IsDiffMode);
        Assert.Equal("Single Map View", _viewModel.StatusMessage);
    }

    [Fact]
    public async Task ToggleDiffModeCommand_RequiresBaseMap()
    {
        // Act - Toggle without any maps
        _viewModel.ToggleDiffModeCommand.Execute(null);

        // Assert - Should do nothing
        Assert.False(_viewModel.IsDiffMode);
    }

    [Fact]
    public void ExitDiffModeCommand_ExitsDiffMode()
    {
        // Arrange
        var viewModel = new MapVisualizationViewModel(_mockEcuService.Object);

        // Act
        viewModel.ExitDiffModeCommand.Execute(null);

        // Assert
        Assert.False(viewModel.IsDiffMode);
        Assert.Equal("Single Map View", viewModel.StatusMessage);
    }

    [Fact]
    public async Task OptimizeMapCommand_RunsOptimization()
    {
        // Arrange
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        var optimizedMap = new CalibrationMap
        {
            Name = "Optimized Map",
            RpmAxis = new double[] { 800, 1300, 1800, 2300, 2800, 3300, 3800, 4300 },
            LoadAxis = new double[] { 0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5 },
            Values = CreateSampleMap()
        };

        _mockTuningOptimizer.Setup(s => s.OptimizeAsync(
                It.IsAny<CalibrationMap>(),
                It.IsAny<IEnumerable<TelemetryDataPoint>>(),
                It.IsAny<OptimizationOptions>(),
                It.IsAny<IProgress<OptimizationProgress>>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new OptimizationResult
            {
                Success = true,
                OptimizedMap = optimizedMap,
                FinalFitness = 0.95,
                FinalAfrError = 0.1,
                CellsChanged = 15,
                GenerationsCompleted = 50,
                Duration = TimeSpan.FromSeconds(5)
            });

        // Act
        await _viewModel.OptimizeMapCommand.ExecuteAsync(CancellationToken.None);

        // Assert
        Assert.True(_viewModel.IsDiffMode);
        Assert.True(_viewModel.ShowOptimizationResults);
        Assert.Equal(0.95, _viewModel.LastFitness);
        Assert.Equal(0.1, _viewModel.LastAfrError);
        Assert.Equal(15, _viewModel.LastCellsChanged);
        Assert.Contains("Optimization complete", _viewModel.StatusMessage);
    }

    [Fact]
    public async Task OptimizeMapCommand_HandlesCancellation()
    {
        // Arrange
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        _mockTuningOptimizer.Setup(s => s.OptimizeAsync(
                It.IsAny<CalibrationMap>(),
                It.IsAny<IEnumerable<TelemetryDataPoint>>(),
                It.IsAny<OptimizationOptions>(),
                It.IsAny<IProgress<OptimizationProgress>>(),
                It.IsAny<CancellationToken>()))
            .ThrowsAsync(new OperationCanceledException());

        var cts = new CancellationTokenSource();
        cts.Cancel();

        // Act
        await _viewModel.OptimizeMapCommand.ExecuteAsync(cts.Token);

        // Assert
        Assert.Equal("Optimization cancelled", _viewModel.StatusMessage);
        Assert.False(_viewModel.IsOptimizing);
    }

    [Fact]
    public async Task OptimizeMapCommand_HandlesFailure()
    {
        // Arrange
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        _mockTuningOptimizer.Setup(s => s.OptimizeAsync(
                It.IsAny<CalibrationMap>(),
                It.IsAny<IEnumerable<TelemetryDataPoint>>(),
                It.IsAny<OptimizationOptions>(),
                It.IsAny<IProgress<OptimizationProgress>>(),
                It.IsAny<CancellationToken>()))
            .ReturnsAsync(new OptimizationResult
            {
                Success = false,
                ErrorMessage = "Python not found"
            });

        // Act
        await _viewModel.OptimizeMapCommand.ExecuteAsync(CancellationToken.None);

        // Assert
        Assert.Contains("failed", _viewModel.StatusMessage);
        Assert.False(_viewModel.ShowOptimizationResults);
    }

    [Fact]
    public void AddTelemetryPoint_AddsTelemetryData()
    {
        // Act
        _viewModel.AddTelemetryPoint(2500, 50, 14.2, 14.7, 25.0);
        _viewModel.AddTelemetryPoint(3000, 60, 14.0, 14.7, 30.0);

        // Assert - Can't directly access telemetry data, but we can verify it doesn't throw
        Assert.Equal("Ready to Read ECU", _viewModel.StatusMessage);
    }

    [Fact]
    public void ClearTelemetryDataCommand_ClearsTelemetry()
    {
        // Arrange
        _viewModel.AddTelemetryPoint(2500, 50, 14.2, 14.7, 25.0);

        // Act
        _viewModel.ClearTelemetryDataCommand.Execute(null);

        // Assert
        Assert.Contains("cleared", _viewModel.StatusMessage);
    }

    [Fact]
    public async Task DiffComputation_CalculatesStatisticsCorrectly()
    {
        // Arrange - Load base map
        await _viewModel.ReadEcuMapCommand.ExecuteAsync(null);

        // Act - Load compare map (which applies simulated changes)
        await _viewModel.LoadCompareMapCommand.ExecuteAsync(null);

        // Assert
        Assert.True(_viewModel.TotalCells > 0);
        Assert.True(_viewModel.TotalCellsChanged >= 0);
        Assert.True(_viewModel.DiffData.Count > 0);
    }

    [Fact]
    public void OptimizationObjectives_ContainsAllValues()
    {
        // Assert
        Assert.Contains(OptimizationObjective.AfrAccuracy, _viewModel.OptimizationObjectives);
        Assert.Contains(OptimizationObjective.FuelEconomy, _viewModel.OptimizationObjectives);
        Assert.Contains(OptimizationObjective.PowerOutput, _viewModel.OptimizationObjectives);
        Assert.Contains(OptimizationObjective.Emissions, _viewModel.OptimizationObjectives);
        Assert.Contains(OptimizationObjective.Balanced, _viewModel.OptimizationObjectives);
    }

    [Fact]
    public async Task OptimizeMapCommand_RequiresBaseMap()
    {
        // Act - Try to optimize without base map
        await _viewModel.OptimizeMapCommand.ExecuteAsync(CancellationToken.None);

        // Assert
        Assert.Equal("Load a map first before optimizing", _viewModel.StatusMessage);
    }

    private static double[,] CreateSampleMap()
    {
        var map = new double[8, 8];
        for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                // Simulate a typical AFR map: richer at high load/RPM, leaner at cruise
                double baseAfr = 14.7; // Stoichiometric
                double loadEffect = j * 0.05; // Richer with load
                double rpmEffect = i > 4 ? 0.3 : 0; // Richer at high RPM
                map[i, j] = baseAfr - loadEffect - rpmEffect;
            }
        }
        return map;
    }
}
