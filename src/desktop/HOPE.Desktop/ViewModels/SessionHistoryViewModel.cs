using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models;
using HOPE.Core.Services.Database;
using HOPE.Core.Services.Export;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;

namespace HOPE.Desktop.ViewModels;

public partial class SessionHistoryViewModel : ObservableObject
{
    private readonly IDatabaseService _dbService;
    private readonly IExportService _exportService;

    [ObservableProperty]
    private ObservableCollection<SessionItem> _sessions = new();

    [ObservableProperty]
    private SessionItem? _selectedSession;

    [ObservableProperty]
    private ObservableCollection<OBD2Reading> _sessionReadings = new();

    [ObservableProperty]
    private bool _isLoading;

    [ObservableProperty]
    private string _statusMessage = "Loading sessions...";
    
    [ObservableProperty]
    private bool _isExporting;

    public SessionHistoryViewModel(IDatabaseService dbService, IExportService exportService)
    {
        _dbService = dbService;
        _exportService = exportService;
        _ = LoadSessionsAsync();
    }

    private async Task LoadSessionsAsync()
    {
        IsLoading = true;
        Sessions.Clear();

        try
        {
            var sessions = await _dbService.GetSessionsAsync();
            
            foreach (var session in sessions)
            {
                Sessions.Add(new SessionItem
                {
                    Id = session.Id,
                    StartTime = session.StartTime,
                    EndTime = session.EndTime,
                    Duration = session.EndTime.HasValue 
                        ? session.EndTime.Value - session.StartTime 
                        : TimeSpan.Zero,
                    Notes = session.Notes
                });
            }

            StatusMessage = Sessions.Count == 0 
                ? "No sessions recorded yet" 
                : $"{Sessions.Count} session(s) found";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error loading sessions: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task RefreshSessionsAsync()
    {
        await LoadSessionsAsync();
    }

    [RelayCommand]
    private async Task ViewSessionDetailsAsync()
    {
        if (SelectedSession == null) return;

        IsLoading = true;
        SessionReadings.Clear();

        try
        {
            var readings = await _dbService.GetSessionDataAsync(SelectedSession.Id);
            foreach (var reading in readings.Take(100)) // Limit for display
            {
                SessionReadings.Add(reading);
            }

            StatusMessage = $"Loaded {readings.Count} readings from session";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error loading session data: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }
    
    [RelayCommand]
    private async Task ExportToCsvAsync()
    {
        if (SelectedSession == null)
        {
            StatusMessage = "Please select a session to export";
            return;
        }

        IsExporting = true;
        StatusMessage = "Exporting to CSV...";

        try
        {
            var exportDir = _exportService.GetDefaultExportDirectory();
            var fileName = $"HOPE_Session_{SelectedSession.StartTime:yyyyMMdd_HHmmss}.csv";
            var outputPath = Path.Combine(exportDir, fileName);

            await _exportService.ExportToCsvAsync(SelectedSession.Id, outputPath);
            
            StatusMessage = $"Exported to: {fileName}";
            
            // Open the export folder
            Process.Start("explorer.exe", $"/select,\"{outputPath}\"");
        }
        catch (Exception ex)
        {
            StatusMessage = $"Export failed: {ex.Message}";
        }
        finally
        {
            IsExporting = false;
        }
    }
    
    [RelayCommand]
    private async Task ExportToPdfAsync()
    {
        if (SelectedSession == null)
        {
            StatusMessage = "Please select a session to export";
            return;
        }

        IsExporting = true;
        StatusMessage = "Generating PDF report...";

        try
        {
            var exportDir = _exportService.GetDefaultExportDirectory();
            var fileName = $"HOPE_Report_{SelectedSession.StartTime:yyyyMMdd_HHmmss}.pdf";
            var outputPath = Path.Combine(exportDir, fileName);

            await _exportService.ExportToPdfAsync(SelectedSession.Id, outputPath);
            
            StatusMessage = $"Report generated: {fileName}";
            
            // Open the PDF
            Process.Start(new ProcessStartInfo
            {
                FileName = outputPath,
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            StatusMessage = $"Export failed: {ex.Message}";
        }
        finally
        {
            IsExporting = false;
        }
    }
}

public class SessionItem
{
    public Guid Id { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public TimeSpan Duration { get; set; }
    public string Notes { get; set; } = string.Empty;

    public string FormattedDate => StartTime.ToString("MMM dd, yyyy HH:mm");
    public string FormattedDuration => Duration.TotalSeconds > 0 
        ? $"{Duration.Minutes}m {Duration.Seconds}s" 
        : "In Progress";
}

