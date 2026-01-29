using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.ECU;
using HOPE.Core.Models;
using System.Collections.ObjectModel;
using System.Windows.Media.Media3D;
using System.Windows.Media;
using System.IO;
using HOPE.Core.Services.Export;

using HOPE.Core.Interfaces;
using HOPE.Desktop.ViewModels;

public partial class MapDiffViewModel : ObservableObject
{
    private readonly ICalibrationRepository? _calibrationRepository;

    [ObservableProperty]
    private Point3DCollection _baseSurfacePoints = new();

    [ObservableProperty]
    private Int32Collection _baseSurfaceIndices = new();
    
    [ObservableProperty]
    private Point3DCollection _compareSurfacePoints = new();

    [ObservableProperty]
    private Int32Collection _compareSurfaceIndices = new();

    [ObservableProperty]
    private CalibrationCommitItem? _selectedBaseCommit;

    [ObservableProperty]
    private CalibrationCommitItem? _selectedCompareCommit;

    public ObservableCollection<CalibrationCommitItem> AvailableCommits { get; } = new();

    [ObservableProperty]
    private string _statusMessage = "Select commits to compare";

    private readonly IExportService? _exportService;
    private CalibrationDiff? _currentDiff;

    public MapDiffViewModel(ICalibrationRepository? calibrationRepository = null, IExportService? exportService = null)
    {
        _calibrationRepository = calibrationRepository;
        _exportService = exportService;
    }

    [RelayCommand]
    private async Task LoadHistoryAsync()
    {
        if (_calibrationRepository == null) return;
        
        var history = await _calibrationRepository.GetHistoryAsync(50);
        AvailableCommits.Clear();
        foreach (var c in history)
        {
            AvailableCommits.Add(new CalibrationCommitItem 
            { 
               Hash = c.Hash,
               ShortHash = c.ShortHash,
               Message = c.Message,
               Timestamp = c.Timestamp,
               Author = c.Author,
               DisplayText = $"{c.ShortHash} - {c.Message}" 
            });
        }
    }

    [RelayCommand]
    private async Task GenerateDiffAsync()
    {
        if (SelectedBaseCommit == null || SelectedCompareCommit == null)
        {
            StatusMessage = "Please select two commits.";
            return;
        }

        StatusMessage = "Calculating diff...";
        try 
        {
             if (_calibrationRepository != null)
             {
                 _currentDiff = await _calibrationRepository.DiffAsync(SelectedBaseCommit.Hash, SelectedCompareCommit.Hash);
             }
             else
             {
                 // Mock Diff for testing/preview without repo
                 _currentDiff = new CalibrationDiff
                 {
                     BaseEcuId = "ECU_BASE",
                     CompareEcuId = "ECU_COMPARE",
                     BaseTimestamp = DateTime.UtcNow.AddDays(-1),
                     CompareTimestamp = DateTime.UtcNow,
                     TotalBytesChanged = 128,
                     Changes = new List<BlockChange>
                     {
                         new BlockChange { BlockName = "FuelMap", Address = 0x8000, ChangeType = ChangeType.Modified, ByteChanges = new List<ByteChange>(new ByteChange[128]) }
                     }
                 };
             }
             
             // Viz Generation (Mock since we don't parse real map structure here yet)
             GenerateMockSurfaces();
             
             StatusMessage = "Diff generated.";
        }
        catch(Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
    }

    [RelayCommand]
    private async Task ExportReportAsync()
    {
        if (_currentDiff == null)
        {
             StatusMessage = "No diff to export.";
             return;
        }

        if (_exportService == null)
        {
             StatusMessage = "Export service unavailable.";
             return;
        }

        try
        {
            string dir = _exportService.GetDefaultExportDirectory();
            string filename = $"Diff_Report_{DateTime.Now:yyyyMMdd_HHmmss}.pdf";
            string path = Path.Combine(dir, filename);

            await _exportService.ExportDiffReportAsync(_currentDiff, path);
            StatusMessage = $"Report exported to {filename}";
        }
        catch(Exception ex)
        {
            StatusMessage = $"Export Error: {ex.Message}";
        }
    }

    private void GenerateMockSurfaces()
    {
        // Generate two surfaces (Base and Compare)
        // 16x16 grid
        int rows = 16;
        int cols = 16;
        
        var basePoints = new Point3DCollection();
        var comparePoints = new Point3DCollection();
        var indices = new Int32Collection();

        for(int x=0; x<rows; x++)
        {
            for(int y=0; y<cols; y++)
            {
                // Fake data
                double z1 = Math.Sin(x * 0.3) * Math.Cos(y * 0.3) * 5 + 14.7; 
                double z2 = z1 + (Math.Sin(x * 1.0) * 0.5); // Add some difference

                basePoints.Add(new Point3D(x, y, z1));
                comparePoints.Add(new Point3D(x, y, z2));
            }
        }

        // Generate Triangles
        for(int x=0; x<rows-1; x++)
        {
            for(int y=0; y<cols-1; y++)
            {
                int topLeft = x * cols + y;
                int topRight = x * cols + (y+1);
                int bottomLeft = (x+1) * cols + y;
                int bottomRight = (x+1) * cols + (y+1);

                // Triangle 1
                indices.Add(topLeft);
                indices.Add(bottomLeft);
                indices.Add(topRight);

                // Triangle 2
                indices.Add(topRight);
                indices.Add(bottomLeft);
                indices.Add(bottomRight);
            }
        }

        BaseSurfacePoints = basePoints;
        BaseSurfaceIndices = indices;
        
        CompareSurfacePoints = comparePoints;
        CompareSurfaceIndices = indices;
    }
}
