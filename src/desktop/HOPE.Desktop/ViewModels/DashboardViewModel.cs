using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using System.Collections.ObjectModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;

namespace HOPE.Desktop.ViewModels;

public partial class DashboardViewModel : ObservableObject, IDisposable
{
    private readonly IOBD2Service _obdService;
    private readonly IDatabaseService _dbService;
    private CompositeDisposable _disposables = new();
    private Guid _currentSessionId;

    [ObservableProperty]
    private double _engineRpm;

    [ObservableProperty]
    private double _vehicleSpeed;

    [ObservableProperty]
    private double _engineLoad;

    [ObservableProperty]
    private double _coolantTemp;

    [ObservableProperty]
    private bool _isStreaming;

    [ObservableProperty]
    private string _connectionStatus = "Disconnected";

    public DashboardViewModel(IOBD2Service obdService, IDatabaseService dbService)
    {
        _obdService = obdService;
        _dbService = dbService;
        _obdService.ConnectionStatusChanged += OnConnectionStatusChanged;
        
        UpdateConnectionStatus();
    }

    private void OnConnectionStatusChanged(object? sender, bool isConnected)
    {
        UpdateConnectionStatus();
    }

    private void UpdateConnectionStatus()
    {
        ConnectionStatus = _obdService.IsConnected ? "Connected (" + _obdService.AdapterType + ")" : "Disconnected";
    }

    [RelayCommand]
    private async Task ToggleStreamingAsync()
    {
        if (IsStreaming)
        {
            await StopStreamingAsync();
        }
        else
        {
            await StartStreamingAsync();
        }
    }

    private async Task StartStreamingAsync()
    {
        if (!_obdService.IsConnected)
        {
            bool connected = await _obdService.ConnectAsync("MOCK_PORT");
            if (!connected) return;
        }

        // Start database session
        _currentSessionId = await _dbService.StartSessionAsync(Guid.Empty); // Mock vehicle ID

        IsStreaming = true;
        
        var pids = new[] { OBD2PIDs.EngineRPM, OBD2PIDs.VehicleSpeed, OBD2PIDs.EngineLoad, OBD2PIDs.CoolantTemp };
        
        _obdService.StreamPIDs(pids)
            .ObserveOn(SynchronizationContext.Current!)
            .Subscribe(async reading =>
            {
                // Assign session ID to reading for logging
                var readingWithSession = reading with { SessionId = _currentSessionId };
                
                await _dbService.LogReadingAsync(readingWithSession);

                switch (reading.PID)
                {
                    case OBD2PIDs.EngineRPM:
                        EngineRpm = reading.Value;
                        break;
                    case OBD2PIDs.VehicleSpeed:
                        VehicleSpeed = reading.Value;
                        break;
                    case OBD2PIDs.EngineLoad:
                        EngineLoad = reading.Value;
                        break;
                    case OBD2PIDs.CoolantTemp:
                        CoolantTemp = reading.Value;
                        break;
                }
            })
            .DisposeWith(_disposables);
    }

    private async Task StopStreamingAsync()
    {
        _disposables.Dispose();
        _disposables = new CompositeDisposable();
        
        if (_currentSessionId != Guid.Empty)
        {
            await _dbService.EndSessionAsync(_currentSessionId);
            _currentSessionId = Guid.Empty;
        }

        IsStreaming = false;
    }

    public void Dispose()
    {
        _obdService.ConnectionStatusChanged -= OnConnectionStatusChanged;
        _disposables.Dispose();
    }
}
