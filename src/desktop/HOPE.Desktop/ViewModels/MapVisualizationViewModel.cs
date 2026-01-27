using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.AI;
using HOPE.Desktop.Converters;
using System.Collections.ObjectModel;
using HOPE.Core.Models;

namespace HOPE.Desktop.ViewModels;

public partial class MapVisualizationViewModel : ObservableObject
{
    private readonly IECUService _ecuService;
    private readonly CalibrationRepository? _calibrationRepository;
    private readonly ITuningOptimizerService? _tuningOptimizer;

    [ObservableProperty]
    private string _statusMessage = "Ready to Read ECU";

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private bool _isDiffMode;

    // Optimization properties
    [ObservableProperty]
    private bool _isOptimizing;

    [ObservableProperty]
    private int _optimizationProgress;

    [ObservableProperty]
    private string _optimizationStatus = string.Empty;

    [ObservableProperty]
    private bool _isOptimizerAvailable;

    [ObservableProperty]
    private int _generations = 50;

    [ObservableProperty]
    private int _populationSize = 50;

    [ObservableProperty]
    private double _mutationRate = 0.1;

    [ObservableProperty]
    private OptimizationObjective _selectedObjective = OptimizationObjective.AfrAccuracy;

    [ObservableProperty]
    private double _lastFitness;

    [ObservableProperty]
    private double _lastAfrError;

    [ObservableProperty]
    private int _lastCellsChanged;

    [ObservableProperty]
    private bool _showOptimizationResults;

    /// <summary>
    /// Available optimization objectives
    /// </summary>
    public IReadOnlyList<OptimizationObjective> OptimizationObjectives { get; } =
        Enum.GetValues<OptimizationObjective>().ToList();

    [ObservableProperty]
    private string _baseMapLabel = "Current Map";

    [ObservableProperty]
    private string _compareMapLabel = "Compare Map";

    [ObservableProperty]
    private int _totalCellsChanged;

    [ObservableProperty]
    private int _totalCells;

    [ObservableProperty]
    private double _averagePercentChange;

    [ObservableProperty]
    private double _maxIncrease;

    [ObservableProperty]
    private double _maxDecrease;

    [ObservableProperty]
    private string _selectedCommitA = string.Empty;

    [ObservableProperty]
    private string _selectedCommitB = string.Empty;

    /// <summary>
    /// Standard map data for single view mode
    /// </summary>
    public ObservableCollection<MapRow> MapData { get; } = new();

    /// <summary>
    /// Diff data for comparison mode
    /// </summary>
    public ObservableCollection<DiffMapRow> DiffData { get; } = new();

    /// <summary>
    /// Available commits for comparison
    /// </summary>
    public ObservableCollection<CalibrationCommitItem> AvailableCommits { get; } = new();

    private double[,]? _baseMapData;
    private double[,]? _compareMapData;
    private CalibrationMap? _currentCalibrationMap;
    private List<TelemetryDataPoint> _telemetryData = new();

    public MapVisualizationViewModel(
        IECUService ecuService,
        CalibrationRepository? calibrationRepository = null,
        ITuningOptimizerService? tuningOptimizer = null)
    {
        _ecuService = ecuService;
        _calibrationRepository = calibrationRepository;
        _tuningOptimizer = tuningOptimizer;
        IsOptimizerAvailable = _tuningOptimizer?.IsAvailable ?? false;
    }

    [RelayCommand]
    private async Task ReadEcuMapAsync()
    {
        IsBusy = true;
        StatusMessage = "Reading ECU Map...";

        try
        {
            var data = await _ecuService.ReadMapAsync("FuelMap");
            _baseMapData = data;

            LoadMapDataToView(data, MapData);

            IsDiffMode = false;
            StatusMessage = "Map Loaded Successfully";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task LoadCompareMapAsync()
    {
        IsBusy = true;
        StatusMessage = "Loading Compare Map...";

        try
        {
            // Load a second map for comparison (simulated variation for demo)
            var data = await _ecuService.ReadMapAsync("FuelMap");
            _compareMapData = ApplySimulatedTuneChanges(data);

            if (_baseMapData != null)
            {
                ComputeDiff(_baseMapData, _compareMapData);
                IsDiffMode = true;
                StatusMessage = $"Diff View - {TotalCellsChanged} cells changed";
            }
            else
            {
                StatusMessage = "Load base map first";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void ToggleDiffMode()
    {
        if (_baseMapData == null) return;

        if (IsDiffMode && _compareMapData == null)
        {
            StatusMessage = "Load a compare map first";
            return;
        }

        IsDiffMode = !IsDiffMode;

        if (IsDiffMode && _baseMapData != null && _compareMapData != null)
        {
            ComputeDiff(_baseMapData, _compareMapData);
            StatusMessage = $"Diff View - {TotalCellsChanged} cells changed";
        }
        else
        {
            StatusMessage = "Single Map View";
        }
    }

    [RelayCommand]
    private void ExitDiffMode()
    {
        IsDiffMode = false;
        StatusMessage = "Single Map View";
    }

    [RelayCommand(CanExecute = nameof(CanOptimize))]
    private async Task OptimizeMapAsync(CancellationToken ct)
    {
        if (_tuningOptimizer == null || _baseMapData == null)
        {
            StatusMessage = "Load a map first before optimizing";
            return;
        }

        IsOptimizing = true;
        IsBusy = true;
        ShowOptimizationResults = false;
        OptimizationProgress = 0;
        OptimizationStatus = "Preparing optimization...";

        try
        {
            // Convert the base map data to CalibrationMap format
            var calibrationMap = CreateCalibrationMap(_baseMapData);
            _currentCalibrationMap = calibrationMap;

            // Generate or use existing telemetry data
            if (_telemetryData.Count == 0)
            {
                _telemetryData = GenerateSyntheticTelemetry(calibrationMap);
                StatusMessage = "Using synthetic telemetry data for optimization";
            }

            var options = new OptimizationOptions
            {
                Generations = Generations,
                PopulationSize = PopulationSize,
                MutationRate = MutationRate,
                Objective = SelectedObjective,
                CrossoverRate = 0.7,
                TargetFitness = 0.95
            };

            var progress = new Progress<OptimizationProgress>(p =>
            {
                OptimizationProgress = p.PercentComplete;
                OptimizationStatus = $"Gen {p.CurrentGeneration}/{p.TotalGenerations} - Fitness: {p.CurrentFitness:F3}, AFR Error: {p.CurrentAfrError:F3}";
            });

            OptimizationStatus = "Running genetic algorithm optimization...";

            var result = await _tuningOptimizer.OptimizeAsync(
                calibrationMap,
                _telemetryData,
                options,
                progress,
                ct);

            if (result.Success)
            {
                // Convert optimized map back to 2D array for display
                _compareMapData = ConvertCalibrationMapToArray(result.OptimizedMap);

                // Update statistics
                LastFitness = result.FinalFitness;
                LastAfrError = result.FinalAfrError;
                LastCellsChanged = result.CellsChanged;

                // Compute and display diff
                ComputeDiff(_baseMapData, _compareMapData);
                IsDiffMode = true;
                ShowOptimizationResults = true;

                StatusMessage = $"Optimization complete - {result.CellsChanged} cells changed, Fitness: {result.FinalFitness:F3}";
                OptimizationStatus = $"Completed in {result.Duration.TotalSeconds:F1}s - {result.GenerationsCompleted} generations";
            }
            else
            {
                StatusMessage = $"Optimization failed: {result.ErrorMessage}";
                OptimizationStatus = "Failed";
            }
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "Optimization cancelled";
            OptimizationStatus = "Cancelled";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
            OptimizationStatus = "Error occurred";
        }
        finally
        {
            IsOptimizing = false;
            IsBusy = false;
            OptimizationProgress = 100;
        }
    }

    private bool CanOptimize()
    {
        return IsOptimizerAvailable && _baseMapData != null && !IsOptimizing;
    }

    [RelayCommand]
    private void ClearTelemetryData()
    {
        _telemetryData.Clear();
        StatusMessage = "Telemetry data cleared - will use synthetic data for next optimization";
    }

    /// <summary>
    /// Add telemetry data point from live OBD2 readings
    /// </summary>
    public void AddTelemetryPoint(double rpm, double load, double actualAfr, double targetAfr, double maf)
    {
        _telemetryData.Add(new TelemetryDataPoint
        {
            Rpm = rpm,
            Load = load,
            ActualAfr = actualAfr,
            TargetAfr = targetAfr,
            Maf = maf,
            CoolantTemp = 90.0,
            IntakeTemp = 30.0,
            Timestamp = DateTime.UtcNow
        });

        // Keep buffer manageable
        if (_telemetryData.Count > 10000)
        {
            _telemetryData.RemoveRange(0, 1000);
        }
    }

    private CalibrationMap CreateCalibrationMap(double[,] data)
    {
        int rows = data.GetLength(0);
        int cols = data.GetLength(1);

        // Create standard RPM and load axes
        var rpmAxis = new double[rows];
        var loadAxis = new double[cols];

        for (int i = 0; i < rows; i++)
        {
            rpmAxis[i] = 800 + i * 500; // 800, 1300, 1800, etc.
        }

        for (int j = 0; j < cols; j++)
        {
            loadAxis[j] = j * 12.5; // 0%, 12.5%, 25%, etc.
        }

        var map = new CalibrationMap
        {
            Name = "Target AFR Map",
            RpmAxis = rpmAxis,
            LoadAxis = loadAxis,
            Values = (double[,])data.Clone(),
            MinValue = 10.0,
            MaxValue = 18.0
        };

        return map;
    }

    private double[,] ConvertCalibrationMapToArray(CalibrationMap map)
    {
        return (double[,])map.Values.Clone();
    }

    private List<TelemetryDataPoint> GenerateSyntheticTelemetry(CalibrationMap map)
    {
        var telemetry = new List<TelemetryDataPoint>();
        var random = new Random(42);

        // Generate telemetry data points across the map surface
        for (int i = 0; i < map.RpmAxis.Length; i++)
        {
            for (int j = 0; j < map.LoadAxis.Length; j++)
            {
                // Generate multiple samples per cell with realistic variation
                int samplesPerCell = 3;
                for (int s = 0; s < samplesPerCell; s++)
                {
                    double rpm = map.RpmAxis[i] + (random.NextDouble() - 0.5) * 200;
                    double load = map.LoadAxis[j] + (random.NextDouble() - 0.5) * 5;
                    double targetAfr = map.Values[i, j];

                    // Simulate actual AFR with some deviation
                    double deviation = (random.NextDouble() - 0.5) * 0.8;
                    double actualAfr = targetAfr + deviation;

                    telemetry.Add(new TelemetryDataPoint
                    {
                        Rpm = Math.Max(600, rpm),
                        Load = Math.Clamp(load, 0, 100),
                        ActualAfr = actualAfr,
                        TargetAfr = targetAfr,
                        Maf = 10 + load * 0.4 + rpm * 0.005,
                        CoolantTemp = 85 + random.NextDouble() * 10,
                        IntakeTemp = 25 + random.NextDouble() * 15,
                        Timestamp = DateTime.UtcNow.AddSeconds(-random.Next(0, 3600))
                    });
                }
            }
        }

        return telemetry;
    }

    [RelayCommand]
    private async Task LoadFromCommitHistoryAsync()
    {
        if (_calibrationRepository == null)
        {
            StatusMessage = "Calibration repository not available";
            return;
        }

        IsBusy = true;
        try
        {
            var history = await _calibrationRepository.GetHistoryAsync(20);
            AvailableCommits.Clear();

            foreach (var commit in history)
            {
                AvailableCommits.Add(new CalibrationCommitItem
                {
                    Hash = commit.Hash,
                    ShortHash = commit.ShortHash,
                    Message = commit.Message,
                    Author = commit.Author,
                    Timestamp = commit.Timestamp,
                    DisplayText = $"{commit.ShortHash} - {commit.Message} ({commit.Timestamp:g})"
                });
            }

            StatusMessage = $"Loaded {history.Count} commits from history";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error loading history: {ex.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void LoadMapDataToView(double[,] data, ObservableCollection<MapRow> target)
    {
        target.Clear();
        TotalCells = data.GetLength(0) * data.GetLength(1);

        for (int i = 0; i < data.GetLength(0); i++)
        {
            var row = new MapRow { Index = i };
            for (int j = 0; j < data.GetLength(1); j++)
            {
                row.Values.Add(data[i, j]);
            }
            target.Add(row);
        }
    }

    private void ComputeDiff(double[,] baseData, double[,] compareData)
    {
        DiffData.Clear();

        int rows = Math.Max(baseData.GetLength(0), compareData.GetLength(0));
        int cols = Math.Max(baseData.GetLength(1), compareData.GetLength(1));

        TotalCells = rows * cols;
        TotalCellsChanged = 0;
        double totalPercentChange = 0;
        MaxIncrease = 0;
        MaxDecrease = 0;

        for (int i = 0; i < rows; i++)
        {
            var row = new DiffMapRow { Index = i };

            for (int j = 0; j < cols; j++)
            {
                double baseVal = i < baseData.GetLength(0) && j < baseData.GetLength(1)
                    ? baseData[i, j] : 0;
                double compareVal = i < compareData.GetLength(0) && j < compareData.GetLength(1)
                    ? compareData[i, j] : 0;

                var cell = new DiffCell
                {
                    BaseValue = baseVal,
                    CompareValue = compareVal
                };

                if (cell.HasChanged)
                {
                    TotalCellsChanged++;
                    totalPercentChange += Math.Abs(cell.PercentChange);

                    if (cell.Difference > MaxIncrease)
                        MaxIncrease = cell.Difference;
                    if (cell.Difference < MaxDecrease)
                        MaxDecrease = cell.Difference;
                }

                row.Cells.Add(cell);
            }

            DiffData.Add(row);
        }

        AveragePercentChange = TotalCellsChanged > 0
            ? totalPercentChange / TotalCellsChanged
            : 0;
    }

    /// <summary>
    /// Simulates tune changes for demonstration purposes.
    /// In production, this would load from actual calibration files.
    /// </summary>
    private double[,] ApplySimulatedTuneChanges(double[,] original)
    {
        var modified = (double[,])original.Clone();
        var random = new Random(42); // Fixed seed for reproducibility

        // Simulate a "performance tune" - richer mixture at high RPM/load
        for (int i = 0; i < modified.GetLength(0); i++)
        {
            for (int j = 0; j < modified.GetLength(1); j++)
            {
                // Higher rows = higher RPM, higher columns = higher load
                bool isHighRpmHighLoad = i >= modified.GetLength(0) / 2 && j >= modified.GetLength(1) / 2;
                bool isMidRange = i >= modified.GetLength(0) / 3 && i <= 2 * modified.GetLength(0) / 3;

                if (isHighRpmHighLoad)
                {
                    // Richer mixture for power (lower AFR)
                    modified[i, j] -= 0.3 + random.NextDouble() * 0.4;
                }
                else if (isMidRange && random.NextDouble() > 0.7)
                {
                    // Random small adjustments in mid-range
                    modified[i, j] += (random.NextDouble() - 0.5) * 0.2;
                }
            }
        }

        return modified;
    }
}



public class CalibrationCommitItem
{
    public string Hash { get; set; } = string.Empty;
    public string ShortHash { get; set; } = string.Empty;
    public string Message { get; set; } = string.Empty;
    public string Author { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string DisplayText { get; set; } = string.Empty;
}


