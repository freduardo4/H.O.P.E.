using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.OBD;
using System.Collections.ObjectModel;
using System.IO.Ports;

namespace HOPE.Desktop.ViewModels;

public partial class SettingsViewModel : ObservableObject
{
    private readonly IOBD2Service _obdService;

    [ObservableProperty]
    private ObservableCollection<string> _availablePorts = new();

    [ObservableProperty]
    private string? _selectedPort;

    [ObservableProperty]
    private ObservableCollection<int> _baudRates = new() { 9600, 19200, 38400, 57600, 115200 };

    [ObservableProperty]
    private int _selectedBaudRate = 38400;

    [ObservableProperty]
    private bool _isConnected;

    [ObservableProperty]
    private string _connectionStatus = "Disconnected";

    [ObservableProperty]
    private bool _isTesting;

    [ObservableProperty]
    private string _adapterInfo = "";

    public SettingsViewModel(IOBD2Service obdService)
    {
        _obdService = obdService;
        RefreshPorts();
        UpdateConnectionStatus();
    }

    [RelayCommand]
    private void RefreshPorts()
    {
        AvailablePorts.Clear();
        
        // Add mock port for testing
        AvailablePorts.Add("MOCK_PORT");
        
        // Add real COM ports
        foreach (var port in _obdService.GetAvailablePorts())
        {
            AvailablePorts.Add(port);
        }

        // Add J2534 devices
        try
        {
            var j2534Devices = HOPE.Core.Hardware.J2534Adapter.GetInstalledDevices();
            foreach (var device in j2534Devices)
            {
                AvailablePorts.Add($"J2534:{device.Name}");
            }
        }
        catch { /* Fallback if J2534 not available on system */ }

        if (AvailablePorts.Count > 0 && SelectedPort == null)
        {
            SelectedPort = AvailablePorts[0];
        }
    }

    [RelayCommand]
    private async Task TestConnectionAsync()
    {
        if (string.IsNullOrEmpty(SelectedPort))
        {
            ConnectionStatus = "Please select a port";
            return;
        }

        IsTesting = true;
        ConnectionStatus = "Testing connection...";
        AdapterInfo = "";

        try
        {
            // Disconnect if already connected
            if (_obdService.IsConnected)
            {
                await _obdService.DisconnectAsync();
            }

            bool connected = await _obdService.ConnectAsync(SelectedPort);
            
            if (connected)
            {
                IsConnected = true;
                ConnectionStatus = "Connected successfully!";
                
                // Get adapter info
                var ecuInfo = await _obdService.GetECUInfoAsync();
                var vin = await _obdService.GetVINAsync();
                AdapterInfo = $"Adapter: {_obdService.AdapterType}\n" +
                             $"Protocol: {_obdService.DetectedProtocol ?? "Auto"}\n" +
                             $"ECU: {ecuInfo ?? "N/A"}\n" +
                             $"VIN: {vin ?? "N/A"}";
            }
            else
            {
                IsConnected = false;
                ConnectionStatus = "Connection failed - check adapter";
            }
        }
        catch (Exception ex)
        {
            IsConnected = false;
            ConnectionStatus = $"Error: {ex.Message}";
        }
        finally
        {
            IsTesting = false;
        }
    }

    [RelayCommand]
    private async Task DisconnectAsync()
    {
        try
        {
            await _obdService.DisconnectAsync();
            IsConnected = false;
            ConnectionStatus = "Disconnected";
            AdapterInfo = "";
        }
        catch (Exception ex)
        {
            ConnectionStatus = $"Error disconnecting: {ex.Message}";
        }
    }

    private void UpdateConnectionStatus()
    {
        IsConnected = _obdService.IsConnected;
        ConnectionStatus = IsConnected ? "Connected" : "Disconnected";
    }
}
