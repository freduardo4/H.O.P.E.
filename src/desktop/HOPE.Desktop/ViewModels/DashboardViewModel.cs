using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Services.AI;
using System.Collections.ObjectModel;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using LiveChartsCore;
using LiveChartsCore.SkiaSharpView;
using LiveChartsCore.Defaults;

namespace HOPE.Desktop.ViewModels;

public partial class DashboardViewModel : ObservableObject, IDisposable
{
    private readonly IOBD2Service _obdService;
    private readonly IDatabaseService _dbService;
    private readonly IAnomalyService _anomalyService;
    private CompositeDisposable _disposables = new();
    private Guid _currentSessionId;
    
    // Chart data buffers (rolling 60-second window)
    private const int MaxChartPoints = 60;
    private readonly ObservableCollection<ObservableValue> _rpmValues = new();
    private readonly ObservableCollection<ObservableValue> _speedValues = new();
    private readonly ObservableCollection<ObservableValue> _loadValues = new();
    
    // Reading buffer for anomaly detection
    private readonly List<OBD2Reading> _readingBuffer = new();
    private int _readingCount = 0;

    [ObservableProperty]
    private double _engineRpm;

    [ObservableProperty]
    private double _vehicleSpeed;

    [ObservableProperty]
    private double _engineLoad;

    [ObservableProperty]
    private double _coolantTemp;

    [ObservableProperty]
    private double _mafAirFlow;

    [ObservableProperty]
    private double _throttlePosition;

    [ObservableProperty]
    private double _intakeAirTemp;

    [ObservableProperty]
    private double _manifoldPressure;

    [ObservableProperty]
    private bool _isStreaming;

    [ObservableProperty]
    private string _connectionStatus = "Disconnected";
    
    [ObservableProperty]
    private bool _isChartVisible = true;
    
    [ObservableProperty]
    private double _anomalyScore;
    
    [ObservableProperty]
    private bool _isAnomaly;
    
    [ObservableProperty]
    private string _anomalyDescription = "No anomalies detected";
    
    // Chart series for real-time visualization
    public ISeries[] ChartSeries { get; }
    
    public Axis[] XAxes { get; } = new Axis[]
    {
        new Axis
        {
            Name = "Time (s)",
            NamePaint = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.White),
            LabelsPaint = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.LightGray),
            MinLimit = 0,
            MaxLimit = 60
        }
    };
    
    public Axis[] YAxesRpm { get; } = new Axis[]
    {
        new Axis
        {
            Name = "RPM",
            NamePaint = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.White),
            LabelsPaint = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.LightGray),
            MinLimit = 0,
            MaxLimit = 8000
        }
    };

    public DashboardViewModel(IOBD2Service obdService, IDatabaseService dbService, IAnomalyService anomalyService)
    {
        _obdService = obdService;
        _dbService = dbService;
        _anomalyService = anomalyService;
        _obdService.ConnectionStatusChanged += OnConnectionStatusChanged;
        
        // Initialize chart series
        ChartSeries = new ISeries[]
        {
            new LineSeries<ObservableValue>
            {
                Name = "RPM",
                Values = _rpmValues,
                Stroke = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.LimeGreen, 2),
                GeometrySize = 0,
                Fill = null
            },
            new LineSeries<ObservableValue>
            {
                Name = "Speed (km/h × 50)",
                Values = _speedValues,
                Stroke = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.DodgerBlue, 2),
                GeometrySize = 0,
                Fill = null
            },
            new LineSeries<ObservableValue>
            {
                Name = "Load (% × 50)",
                Values = _loadValues,
                Stroke = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.Orange, 2),
                GeometrySize = 0,
                Fill = null
            }
        };
        
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
    private void ToggleChartVisibility()
    {
        IsChartVisible = !IsChartVisible;
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

        // Clear chart data
        _rpmValues.Clear();
        _speedValues.Clear();
        _loadValues.Clear();
        _readingBuffer.Clear();
        _readingCount = 0;
        
        // Reset anomaly state
        AnomalyScore = 0;
        IsAnomaly = false;
        AnomalyDescription = "Collecting data...";

        // Start database session
        _currentSessionId = await _dbService.StartSessionAsync(Guid.Empty); // Mock vehicle ID

        IsStreaming = true;
        
        var pids = new[] { 
            OBD2PIDs.EngineRPM, 
            OBD2PIDs.VehicleSpeed, 
            OBD2PIDs.EngineLoad, 
            OBD2PIDs.CoolantTemp,
            OBD2PIDs.MAFSensor,
            OBD2PIDs.ThrottlePosition,
            OBD2PIDs.IntakeAirTemp,
            OBD2PIDs.IntakeManifoldPressure
        };
        
        _obdService.StreamPIDs(pids)
            .ObserveOn(SynchronizationContext.Current!)
            .Subscribe(async reading =>
            {
                // Assign session ID to reading for logging
                var readingWithSession = reading with { SessionId = _currentSessionId };
                
                await _dbService.LogReadingAsync(readingWithSession);
                
                // Buffer readings for anomaly detection
                _readingBuffer.Add(readingWithSession);
                if (_readingBuffer.Count > MaxChartPoints * 8) // 8 PIDs × 60 seconds
                {
                    _readingBuffer.RemoveAt(0);
                }

                switch (reading.PID)
                {
                    case OBD2PIDs.EngineRPM:
                        EngineRpm = reading.Value;
                        AddChartPoint(_rpmValues, reading.Value);
                        break;
                    case OBD2PIDs.VehicleSpeed:
                        VehicleSpeed = reading.Value;
                        AddChartPoint(_speedValues, reading.Value * 50); // Scale for visibility
                        break;
                    case OBD2PIDs.EngineLoad:
                        EngineLoad = reading.Value;
                        AddChartPoint(_loadValues, reading.Value * 50); // Scale for visibility
                        break;
                    case OBD2PIDs.CoolantTemp:
                        CoolantTemp = reading.Value;
                        break;
                    case OBD2PIDs.MAFSensor:
                        MafAirFlow = reading.Value;
                        break;
                    case OBD2PIDs.ThrottlePosition:
                        ThrottlePosition = reading.Value;
                        break;
                    case OBD2PIDs.IntakeAirTemp:
                        IntakeAirTemp = reading.Value;
                        break;
                    case OBD2PIDs.IntakeManifoldPressure:
                        ManifoldPressure = reading.Value;
                        break;
                }
                
                // Run anomaly detection every 60 readings (~7.5 seconds with 8 PIDs)
                _readingCount++;
                if (_readingCount >= 60 && _readingBuffer.Count >= 60)
                {
                    _readingCount = 0;
                    await RunAnomalyDetectionAsync();
                }
            })
            .DisposeWith(_disposables);
    }
    
    private void AddChartPoint(ObservableCollection<ObservableValue> collection, double value)
    {
        collection.Add(new ObservableValue(value));
        if (collection.Count > MaxChartPoints)
        {
            collection.RemoveAt(0);
        }
    }
    
    private async Task RunAnomalyDetectionAsync()
    {
        try
        {
            var result = await _anomalyService.AnalyzeAsync(_readingBuffer);
            AnomalyScore = result.Score;
            IsAnomaly = result.IsAnomaly;
            AnomalyDescription = result.IsAnomaly 
                ? $"⚠️ {result.Description}" 
                : "✓ No anomalies detected";
        }
        catch
        {
            AnomalyDescription = "Analysis unavailable";
        }
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
        AnomalyDescription = "Streaming stopped";
    }

    public void Dispose()
    {
        _obdService.ConnectionStatusChanged -= OnConnectionStatusChanged;
        _disposables.Dispose();
    }
}
