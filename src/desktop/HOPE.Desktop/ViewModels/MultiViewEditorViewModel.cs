using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using System.Collections.ObjectModel;
using System.Data;
using System.Windows.Media.Media3D;
using System.Windows.Media;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.SkiaSharpView.Painting;
using SkiaSharp;
using HOPE.Core.Services.ECU;

using HOPE.Core.Services.ECU;
using Microsoft.Extensions.Logging;
using HOPE.Core.Interfaces;

namespace HOPE.Desktop.ViewModels;

public partial class MultiViewEditorViewModel : ObservableObject
{
    private readonly ICalibrationRepository _repository;
    private readonly ILogger<MultiViewEditorViewModel> _logger;
    private CalibrationFile? _currentCalibration;

    [ObservableProperty]
    private string _mapName = "Fuel Map (High Octane)";

    [ObservableProperty]
    private string _mapAddress = "0x8000";

    [ObservableProperty]
    private string _statusMessage = "Ready";

    // --- 3D View Properties ---
    [ObservableProperty]
    private Point3DCollection _surfacePoints = new();

    [ObservableProperty]
    private Int32Collection _surfaceIndices = new();

    // --- Tabular View Properties ---
    [ObservableProperty]
    private DataView? _tabularData;

    // --- 2D Chart Properties ---
    [ObservableProperty]
    private ISeries[] _chartSeries = Array.Empty<ISeries>();

    [ObservableProperty]
    private Axis[] _xAxes = { new Axis { Name = "RPM" } };

    [ObservableProperty]
    private Axis[] _yAxes = { new Axis { Name = "Load/VE" } };

    [ObservableProperty]
    private ObservableCollection<string> _axisRows = new();

    [ObservableProperty]
    private string? _selectedRow;

    partial void OnSelectedRowChanged(string? value)
    {
        UpdateChartForSelectedRow(value);
    }

    // --- Hex View Properties ---
    [ObservableProperty]
    private string _hexContent = string.Empty;

    // --- Axis Rescaling Logic ---
    private double[] _currentXAxis = new double[] { 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000, 4400, 4800, 5200, 5600, 6000, 6400, 6800 };
    private double[] _currentYAxis = new double[] { 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160 };
    private double[,] _mapData = new double[16, 16]; // [Row, Col]

    public MultiViewEditorViewModel(ICalibrationRepository repository, ILogger<MultiViewEditorViewModel> logger)
    {
        _repository = repository;
        _logger = logger;
        // InitializeMockData(); // Mocks removed, use LoadAsync
    }

    public async Task LoadAsync(string commitHash)
    {
        try 
        {
            StatusMessage = $"Loading commit {commitHash}...";
            _logger.LogInformation("Loading calibration commit {CommitHash}", commitHash);
            _currentCalibration = await _repository.GetCalibrationAsync(commitHash);
            
            // TODO: Parse calibration blocks into _mapData
            InitializeMockData(); 
            StatusMessage = "Calibration loaded successfully";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load calibration {CommitHash}", commitHash);
            StatusMessage = $"Error loading calibration: {ex.Message}";
        }
    }

    private void InitializeMockData()
    {
        // Generate Mock Map Data (VE Table-ish)
        for (int r = 0; r < 16; r++)
        {
            for (int c = 0; c < 16; c++)
            {
                // Typical VE shape: increases with Load (Y), peaks at mid-high RPM (X)
                double rpmFactor = Math.Sin(c * 0.2); // curve
                double loadFactor = (r + 5.0) / 21.0;
                _mapData[r, c] = 14.7 + (loadFactor * 10) + (rpmFactor * 2);
            }
        }

        RefreshViews();
    }

    private void RefreshViews()
    {
        Update3DView();
        UpdateTabularView();
        Update2DChartView();
        UpdateHexView();
    }

    private void Update3DView()
    {
        var points = new Point3DCollection();
        var indices = new Int32Collection();

        int rows = _mapData.GetLength(0);
        int cols = _mapData.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                points.Add(new Point3D(c, r, _mapData[r, c]));
            }
        }

        for (int r = 0; r < rows - 1; r++)
        {
            for (int c = 0; c < cols - 1; c++)
            {
                int p1 = r * cols + c;
                int p2 = r * cols + (c + 1);
                int p3 = (r + 1) * cols + c;
                int p4 = (r + 1) * cols + (c + 1);

                indices.Add(p1); indices.Add(p3); indices.Add(p2);
                indices.Add(p2); indices.Add(p3); indices.Add(p4);
            }
        }

        SurfacePoints = points;
        SurfaceIndices = indices;
    }

    private void UpdateTabularView()
    {
        var dt = new DataTable();
        dt.Columns.Add("Load \\ RPM"); // Corner header
        
        foreach (var xVal in _currentXAxis)
        {
            dt.Columns.Add(xVal.ToString("F0"));
        }

        int rows = _mapData.GetLength(0);
        int cols = _mapData.GetLength(1);

        for (int r = 0; r < rows; r++)
        {
            var row = dt.NewRow();
            row[0] = _currentYAxis[r].ToString("F1");
            for (int c = 0; c < cols; c++)
            {
                row[c + 1] = _mapData[r, c].ToString("F2");
            }
            dt.Rows.Add(row);
        }

        TabularData = dt.DefaultView;
    }

    private void Update2DChartView()
    {
        AxisRows.Clear();
        int rows = _mapData.GetLength(0);
        for(int r=0; r<rows; r++)
        {
            AxisRows.Add($"Row {r} ({_currentYAxis[r]:F1})");
        }
        
        if (SelectedRow == null && AxisRows.Count > 0)
        {
            SelectedRow = AxisRows[0];
        }
        else
        {
            // Force redraw logic
            UpdateChartForSelectedRow(SelectedRow);
        }
    }

    private void UpdateChartForSelectedRow(string? rowSelection)
    {
        if (string.IsNullOrEmpty(rowSelection)) return;

        int rowIndex = AxisRows.IndexOf(rowSelection);
        if (rowIndex < 0) return;

        int cols = _mapData.GetLength(1);
        var values = new double[cols];
        for (int c = 0; c < cols; c++)
        {
            values[c] = _mapData[rowIndex, c];
        }

        ChartSeries = new ISeries[]
        {
            new LineSeries<double>
            {
                Values = values,
                Fill = null,
                GeometrySize = 5,
                Stroke = new SolidColorPaint(SKColors.DodgerBlue) { StrokeThickness = 3 }
            }
        };
    }

    private void UpdateHexView()
    {
        // Mock Hex Dump
        var sb = new System.Text.StringBuilder();
        int rows = _mapData.GetLength(0);
        int cols = _mapData.GetLength(1);
        
        // Assume floats (4 bytes)
        for (int r = 0; r < rows; r++)
        {
            sb.Append($"{0x8000 + (r * cols * 4):X8}: ");
            for (int c = 0; c < cols; c++)
            {
                byte[] bytes = BitConverter.GetBytes((float)_mapData[r, c]);
                foreach (var b in bytes) sb.Append($"{b:X2} ");
            }
            sb.AppendLine();
        }
        HexContent = sb.ToString(); 
    }

    [RelayCommand]
    private async Task Save()
    {
        try
        {
            if (_currentCalibration == null) 
            {
                 // Create dummy if missing (for demo)
                 _currentCalibration = new CalibrationFile { EcuId = "DEMO_ECU", Blocks = new List<CalibrationBlock>() };
            }

            // Update _currentCalibration with _mapData (would need serialization logic)
            _logger.LogInformation("Saving calibration...");
            
            await _repository.StageAsync(_currentCalibration);
            await _repository.CommitAsync("Updated map from MultiViewEditor");
            
            _logger.LogInformation("Calibration saved successfully.");
            StatusMessage = "Calibration saved successfully";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to save calibration");
            StatusMessage = $"Error saving: {ex.Message}";
        }
    }

    [RelayCommand]
    private void OpenAxisEditor()
    {
        // In a real app, this would pass the current axes to a dialog service.
        // For simulation, we'll just demonstrate the logic:
        
        try
        {
            // Simulating user changing X-Axis (rescaling RPMs)
            var oldAxis = _currentXAxis.ToArray();
            
            // New axis: Shifted + Compressed (e.g. user wants more resolution at high RPM)
            var newAxis = new double[16];
            for(int i=0; i<16; i++) newAxis[i] = 1000 + (i * 300); // Different spacing

            // Re-interpolate every row
            int rows = _mapData.GetLength(0);
            
            for (int r = 0; r < rows; r++)
            {
                // Extract row
                double[] rowVals = new double[16];
                for(int c=0; c<16; c++) rowVals[c] = _mapData[r,c];

                // Interpolate
                var newVals = CalibrationRepository.InterpolateMap(oldAxis, rowVals, newAxis);

                // Update Map
                for(int c=0; c<16; c++) _mapData[r,c] = newVals[c];
            }
            
            _currentXAxis = newAxis;
            
            RefreshViews();
            StatusMessage = "Axis rescaled successfully";
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to rescale axis");
            StatusMessage = $"Error rescaling axis: {ex.Message}";
        }
    }
}
