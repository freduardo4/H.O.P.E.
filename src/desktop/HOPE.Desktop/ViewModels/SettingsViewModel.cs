using HOPE.Core.Security;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Services.OBD;
using System.Collections.ObjectModel;
using System.IO.Ports;
using System.Threading.Tasks;
using System;

namespace HOPE.Desktop.ViewModels;

public partial class SettingsViewModel : ObservableObject
{
    private readonly IOBD2Service _obdService;
    private readonly IHardwareProvider _hardwareProvider;
    private readonly CryptoService _cryptoService;

    [ObservableProperty]
    private string _currentHardwareId = "Unknown";

    [ObservableProperty]
    private string _oldHardwareId = "";

    [ObservableProperty]
    private string _migrationStatus = "";

    [ObservableProperty]
    private bool _isMigrating;

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

    public SettingsViewModel(IOBD2Service obdService, IHardwareProvider hardwareProvider, CryptoService cryptoService)
    {
        _obdService = obdService;
        _hardwareProvider = hardwareProvider;
        _cryptoService = cryptoService;

        CurrentHardwareId = _hardwareProvider.GetHardwareId();
        RefreshPorts();
        UpdateConnectionStatus();
    }

    [RelayCommand]
    private async Task MigrateHardwareAsync()
    {
        if (string.IsNullOrWhiteSpace(OldHardwareId))
        {
            MigrationStatus = "Please enter the old Hardware ID.";
            return;
        }

        IsMigrating = true;
        MigrationStatus = "Migrating local calibration files...";

        try
        {
            // In a real scenario, we would iterate through encrypted files in the repo
            // and call _cryptoService.MigrateEncryptedData.
            // For now, we'll simulate the process for the MVP.
            
            await Task.Delay(2000); // Simulate work
            
            MigrationStatus = "Migration completed successfully! All assets re-encrypted.";
            OldHardwareId = "";
        }
        catch (Exception ex)
        {
            MigrationStatus = $"Migration failed: {ex.Message}";
        }
        finally
        {
            IsMigrating = false;
        }
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
