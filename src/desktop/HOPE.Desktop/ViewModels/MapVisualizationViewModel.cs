using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.ECU;
using HOPE.Desktop.Converters;
using System.Collections.ObjectModel;

namespace HOPE.Desktop.ViewModels;

public partial class MapVisualizationViewModel : ObservableObject
{
    private readonly IECUService _ecuService;
    private readonly CalibrationRepository? _calibrationRepository;

    [ObservableProperty]
    private string _statusMessage = "Ready to Read ECU";

    [ObservableProperty]
    private bool _isBusy;

    [ObservableProperty]
    private bool _isDiffMode;

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

    public MapVisualizationViewModel(IECUService ecuService, CalibrationRepository? calibrationRepository = null)
    {
        _ecuService = ecuService;
        _calibrationRepository = calibrationRepository;
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

public class MapRow
{
    public int Index { get; set; }
    public ObservableCollection<double> Values { get; } = new();
}

public class DiffMapRow
{
    public int Index { get; set; }
    public ObservableCollection<DiffCell> Cells { get; } = new();
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
