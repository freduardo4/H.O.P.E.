using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
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
                DiagnosticCodes.Add(new DTCItem
                {
                    Code = dtc.Code,
                    Description = GetDTCDescription(dtc.Code),
                    Severity = GetDTCSeverity(dtc.Code),
                    Category = GetDTCCategory(dtc.Code)
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

    private string GetDTCDescription(string code)
    {
        // Common DTC descriptions lookup
        var descriptions = new Dictionary<string, string>
        {
            ["P0300"] = "Random/Multiple Cylinder Misfire Detected",
            ["P0301"] = "Cylinder 1 Misfire Detected",
            ["P0302"] = "Cylinder 2 Misfire Detected",
            ["P0420"] = "Catalyst System Efficiency Below Threshold",
            ["P0171"] = "System Too Lean (Bank 1)",
            ["P0172"] = "System Too Rich (Bank 1)",
            ["P0440"] = "Evaporative Emission Control System Malfunction",
            ["P0442"] = "Evaporative Emission System Leak Detected (Small Leak)",
            ["P0455"] = "Evaporative Emission System Leak Detected (Large Leak)",
            ["P0500"] = "Vehicle Speed Sensor Malfunction",
            ["P0113"] = "Intake Air Temperature Sensor High Input",
            ["P0128"] = "Coolant Thermostat Below Thermostat Regulating Temperature"
        };

        return descriptions.TryGetValue(code, out var desc) ? desc : "Unknown code - check service manual";
    }

    private string GetDTCSeverity(string code)
    {
        if (code.StartsWith("P0") || code.StartsWith("P2"))
            return "Medium";
        if (code.StartsWith("P1"))
            return "Low";
        return "High";
    }

    private string GetDTCCategory(string code)
    {
        if (code.StartsWith("P0"))
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
    
    public string SeverityColor => Severity switch
    {
        "Low" => "#4CAF50",
        "Medium" => "#FF9800",
        "High" => "#F44336",
        _ => "#9E9E9E"
    };
}
