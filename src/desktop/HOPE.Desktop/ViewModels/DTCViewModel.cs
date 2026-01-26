using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Data;
using HOPE.Core.Models;
using HOPE.Core.Services.OBD;
using System.Collections.ObjectModel;

namespace HOPE.Desktop.ViewModels;

public partial class DTCViewModel : ObservableObject
{
    private readonly IOBD2Service _obdService;

    [ObservableProperty]
    private ObservableCollection<DTCItem> _diagnosticCodes = new();

    [ObservableProperty]
    private bool _isLoading;

    [ObservableProperty]
    private string _statusMessage = "Click 'Read DTCs' to scan for codes";

    public DTCViewModel(IOBD2Service obdService)
    {
        _obdService = obdService;
    }

    [RelayCommand]
    private async Task ReadDTCsAsync()
    {
        IsLoading = true;
        StatusMessage = "Scanning for diagnostic codes...";
        DiagnosticCodes.Clear();

        try
        {
            if (!_obdService.IsConnected)
            {
                await _obdService.ConnectAsync("MOCK_PORT");
            }

            var dtcs = await _obdService.ReadDTCsAsync();
            
            foreach (var dtc in dtcs)
            {
                var dtcInfo = DTCDatabase.GetInfo(dtc.Code);
                DiagnosticCodes.Add(new DTCItem
                {
                    Code = dtc.Code,
                    Description = dtcInfo?.Description ?? GetDTCDescription(dtc.Code),
                    Severity = dtcInfo != null ? GetSeverityString(dtcInfo.Severity) : GetDTCSeverity(dtc.Code),
                    Category = dtcInfo != null ? dtcInfo.Category.ToString() : GetDTCCategory(dtc.Code),
                    PossibleCauses = dtcInfo?.PossibleCauses ?? []
                });
            }

            StatusMessage = DiagnosticCodes.Count == 0 
                ? "No diagnostic codes found - vehicle is healthy!" 
                : $"Found {DiagnosticCodes.Count} diagnostic code(s)";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error reading codes: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    private async Task ClearDTCsAsync()
    {
        IsLoading = true;
        StatusMessage = "Clearing diagnostic codes...";

        try
        {
            if (!_obdService.IsConnected)
            {
                await _obdService.ConnectAsync("MOCK_PORT");
            }

            await _obdService.ClearDTCsAsync();
            DiagnosticCodes.Clear();
            StatusMessage = "Diagnostic codes cleared successfully!";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error clearing codes: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    private static string GetSeverityString(DTCSeverity severity)
    {
        return severity switch
        {
            DTCSeverity.Critical => "Critical",
            DTCSeverity.High => "High",
            DTCSeverity.Medium => "Medium",
            DTCSeverity.Low => "Low",
            _ => "Medium"
        };
    }

    private static string GetDTCDescription(string code)
    {
        return DTCDatabase.GetDescription(code);
    }

    private static string GetDTCSeverity(string code)
    {
        // Fallback for codes not in database
        if (code.StartsWith("P0") || code.StartsWith("P2"))
            return "Medium";
        if (code.StartsWith("P1"))
            return "Low";
        if (code.StartsWith("P3") || code.StartsWith("U"))
            return "High";
        return "Medium";
    }

    private static string GetDTCCategory(string code)
    {
        if (code.StartsWith("P"))
            return "Powertrain";
        if (code.StartsWith("B"))
            return "Body";
        if (code.StartsWith("C"))
            return "Chassis";
        if (code.StartsWith("U"))
            return "Network";
        return "Generic";
    }
}

public class DTCItem
{
    public string Code { get; set; } = string.Empty;
    public string Description { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public string Category { get; set; } = string.Empty;
    public string[] PossibleCauses { get; set; } = [];

    public string SeverityColor => Severity switch
    {
        "Critical" => "#D32F2F",
        "High" => "#F44336",
        "Medium" => "#FF9800",
        "Low" => "#4CAF50",
        _ => "#9E9E9E"
    };

    public string CausesText => PossibleCauses.Length > 0
        ? string.Join("\n• ", ["Possible causes:", .. PossibleCauses])
        : "No diagnostic information available";

    public bool HasCauses => PossibleCauses.Length > 0;
}
