using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Services.AI;
using HOPE.Core.Services.Reports;
using HOPE.Core.Services.Security;
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
    private readonly IReportService _reportService;
    private readonly ISyncService _syncService;
    private readonly IAuditService _auditService;
    private readonly IRULPredictorService? _rulPredictorService;
    private readonly ExplainableAnomalyService? _explainableService;
    private CompositeDisposable _disposables = new();
    private Guid _currentSessionId;

    // Chart data buffers (rolling 60-second window)
    private const int MaxChartPoints = 60;
    private readonly ObservableCollection<ObservableValue> _rpmValues = new();
    private readonly ObservableCollection<ObservableValue> _speedValues = new();
    private readonly ObservableCollection<ObservableValue> _loadValues = new();

    // Anomaly score history for chart
    private readonly ObservableCollection<ObservableValue> _anomalyScoreValues = new();

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

    [ObservableProperty]
    private bool _isAnomalyPanelExpanded;

    [ObservableProperty]
    private double _anomalyConfidence;

    [ObservableProperty]
    private int _totalAnomaliesDetected;

    [ObservableProperty]
    private string _anomalySeverity = "Normal";

    // RUL Prediction Properties
    [ObservableProperty]
    private double _overallVehicleHealth = 1.0;

    [ObservableProperty]
    private bool _isRulPredictionAvailable;

    [ObservableProperty]
    private string _nextServiceDate = "N/A";

    [ObservableProperty]
    private double _estimatedMaintenanceCost;

    [ObservableProperty]
    private bool _isComponentHealthPanelExpanded;

    [ObservableProperty]
    private FocusMode _currentFocusMode = FocusMode.Standard;

    [ObservableProperty]
    private string _accentColor = "#00FF00"; // Default LimeGreen

    [ObservableProperty]
    private string _dashboardTitle = "H.O.P.E. DASHBOARD";

    [ObservableProperty]
    private bool _isPerformanceVisible = true;

    [ObservableProperty]
    private bool _isEconomyVisible = true;

    [ObservableProperty]
    private bool _isDiagnosticVisible = true;

    [ObservableProperty]
    private bool _isGeneratingReport;

    [ObservableProperty]
    private string _reportStatusText = string.Empty;

    [ObservableProperty]
    private string _syncStatusText = "Up to date";

    [ObservableProperty]
    private bool _isSyncing;

    public ObservableCollection<FocusMode> AvailableFocusModes { get; } = new()
    {
        FocusMode.Standard,
        FocusMode.WOT,
        FocusMode.Economy,
        FocusMode.Diagnostic
    };

    /// <summary>
    /// Component health predictions
    /// </summary>
    public ObservableCollection<ComponentHealthItem> ComponentHealthItems { get; } = new();

    /// <summary>
    /// Urgent maintenance items
    /// </summary>
    public ObservableCollection<string> UrgentMaintenanceItems { get; } = new();

    /// <summary>
    /// Parameters contributing to the current anomaly
    /// </summary>
    public ObservableCollection<ContributingParameter> ContributingParameters { get; } = new();

    /// <summary>
    /// Repair suggestions for the current anomaly
    /// </summary>
    public ObservableCollection<RepairSuggestion> RepairSuggestions { get; } = new();

    /// <summary>
    /// Recent anomaly history
    /// </summary>
    public ObservableCollection<AnomalyHistoryItem> AnomalyHistory { get; } = new();

    /// <summary>
    /// Anomaly score trend chart series
    /// </summary>
    public ISeries[] AnomalyChartSeries { get; }
    
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

    public DashboardViewModel(
        IOBD2Service obdService,
        IDatabaseService dbService,
        IAnomalyService anomalyService,
        IReportService reportService,
        ISyncService syncService,
        IAuditService auditService,
        IRULPredictorService? rulPredictorService = null,
        ExplainableAnomalyService? explainableService = null)
    {
        _obdService = obdService;
        _dbService = dbService;
        _anomalyService = anomalyService;
        _reportService = reportService;
        _syncService = syncService;
        _auditService = auditService;
        _rulPredictorService = rulPredictorService;
        _explainableService = explainableService;

        _obdService.ConnectionStatusChanged += OnConnectionStatusChanged;
        _syncService.SyncStatusChanged += (s, e) => 
        {
            IsSyncing = e.IsSyncing;
            SyncStatusText = e.Message;
        };

        UpdateConnectionStatus();
        IsRulPredictionAvailable = _rulPredictorService != null;

        // Initial sync pull
        _ = _syncService.PullChangesAsync();

        // Initialize main chart series
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

        // Initialize anomaly score chart series
        AnomalyChartSeries = new ISeries[]
        {
            new LineSeries<ObservableValue>
            {
                Name = "Anomaly Score",
                Values = _anomalyScoreValues,
                Stroke = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColors.OrangeRed, 2),
                GeometrySize = 0,
                Fill = new LiveChartsCore.SkiaSharpView.Painting.SolidColorPaint(SkiaSharp.SKColor.Parse("#33FF4500"))
            }
        };

        UpdateConnectionStatus();
        UpdateUIForFocusMode();
    }

    partial void OnCurrentFocusModeChanged(FocusMode value)
    {
        UpdateUIForFocusMode();
        
        // If streaming, we need to restart to apply new frequency/PID set
        if (IsStreaming)
        {
            _ = RestartStreamingAsync();
        }
    }

    private void UpdateUIForFocusMode()
    {
        switch (CurrentFocusMode)
        {
            case FocusMode.WOT:
                AccentColor = "#FF0000"; // Aggressive Red
                DashboardTitle = "PERFORMANCE MODE (WOT)";
                IsPerformanceVisible = true;
                IsEconomyVisible = false;
                IsDiagnosticVisible = false;
                break;
            case FocusMode.Economy:
                AccentColor = "#00AAFF"; // Calm Blue
                DashboardTitle = "ECO MONITORING";
                IsPerformanceVisible = true;
                IsEconomyVisible = true;
                IsDiagnosticVisible = false;
                break;
            case FocusMode.Diagnostic:
                AccentColor = "#FFAA00"; // Warning Orange
                DashboardTitle = "DEEP DIAGNOSTICS";
                IsPerformanceVisible = false;
                IsEconomyVisible = false;
                IsDiagnosticVisible = true;
                break;
            case FocusMode.Panic:
                AccentColor = "#FF0000"; // Pure Red
                DashboardTitle = "CRITICAL ALERT: ENGINE SAFETY";
                IsPerformanceVisible = true;
                IsEconomyVisible = false;
                IsDiagnosticVisible = true;
                break;
            default:
                AccentColor = "#00FF00"; // Standard Green
                DashboardTitle = "H.O.P.E. DASHBOARD";
                IsPerformanceVisible = true;
                IsEconomyVisible = true;
                IsDiagnosticVisible = true;
                break;
        }
        
        _obdService.SetFocusModeAsync(CurrentFocusMode);
    }

    [RelayCommand]
    private async Task GenerateReportAsync()
    {
        if (_currentSessionId == Guid.Empty) return;

        IsGeneratingReport = true;
        ReportStatusText = "Generating Report...";

        try
        {
            var session = await _dbService.GetSessionAsync(_currentSessionId);
            if (session != null)
            {
                // Populate session with currently mapped anomalies and insights
                session.AIInsights.AddRange(AnomalyHistory.Select(h => new AIInsight
                {
                    Description = h.Description,
                    Confidence = h.Score,
                    DetectedAt = DateTime.Now // Simplified
                }));

                string reportsDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "HOPE", "Reports");
                if (!Directory.Exists(reportsDir)) Directory.CreateDirectory(reportsDir);
                
                string fileName = $"HealthReport_{DateTime.Now:yyyyMMdd_HHmmss}.pdf";
                string fullPath = Path.Combine(reportsDir, fileName);

                await _reportService.GenerateVehicleHealthReportAsync(session, fullPath);
                
                // Log audit event
                await _auditService.LogActivityAsync("ReportGenerated", session.Id, $"Path: {fullPath}");

                ReportStatusText = $"Report saved: {fileName}";
                
                // Show file in explorer
                System.Diagnostics.Process.Start("explorer.exe", $"/select,\"{fullPath}\"");
            }
        }
        catch (Exception ex)
        {
            ReportStatusText = "Failed to generate report";
            // Log error...
        }
        finally
        {
            await Task.Delay(3000); // Show status for a bit
            IsGeneratingReport = false;
            ReportStatusText = string.Empty;
        }
    }

    private async Task RestartStreamingAsync()
    {
        await StopStreamingAsync();
        await StartStreamingAsync();
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

        // Log audit event
        await _auditService.LogActivityAsync("DiagnosticSessionStart", _currentSessionId, "VehicleId: " + Guid.Empty);

        IsStreaming = true;
        
        var pids = GetPIDsForCurrentMode();
        
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

    private string[] GetPIDsForCurrentMode()
    {
        return CurrentFocusMode switch
        {
            FocusMode.WOT => new[]
            {
                OBD2PIDs.EngineRPM,
                OBD2PIDs.ThrottlePosition,
                OBD2PIDs.EngineLoad,
                OBD2PIDs.MAFSensor,
                OBD2PIDs.TimingAdvance,
                OBD2PIDs.IntakeManifoldPressure
            },
            FocusMode.Economy => new[]
            {
                OBD2PIDs.EngineRPM,
                OBD2PIDs.VehicleSpeed,
                OBD2PIDs.EngineLoad,
                OBD2PIDs.MAFSensor,
                OBD2PIDs.ShortTermFuelTrim,
                OBD2PIDs.LongTermFuelTrim
            },
            FocusMode.Diagnostic => new[]
            {
                OBD2PIDs.CoolantTemp,
                OBD2PIDs.IntakeAirTemp,
                OBD2PIDs.FuelPressure,
                OBD2PIDs.O2Sensor1Voltage,
                OBD2PIDs.BarometricPressure,
                OBD2PIDs.EngineRuntime
            },
            _ => new[]
            {
                OBD2PIDs.EngineRPM,
                OBD2PIDs.VehicleSpeed,
                OBD2PIDs.EngineLoad,
                OBD2PIDs.CoolantTemp,
                OBD2PIDs.MAFSensor,
                OBD2PIDs.ThrottlePosition,
                OBD2PIDs.IntakeAirTemp,
                OBD2PIDs.IntakeManifoldPressure
            }
        };
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
            AnomalyConfidence = result.Confidence;

            // Add to anomaly score history chart
            AddChartPoint(_anomalyScoreValues, result.Score * 100); // Scale to percentage

            // Update contributing parameters
            ContributingParameters.Clear();
            foreach (var param in result.ContributingParameters.Take(5))
            {
                ContributingParameters.Add(new ContributingParameter
                {
                    Name = param,
                    Contribution = GetParameterContribution(param)
                });
            }

            if (result.IsAnomaly)
            {
                TotalAnomaliesDetected++;
                AnomalySeverity = ClassifyAnomalySeverity(result.Score);
                AnomalyDescription = result.Description;

                // Add to history
                AnomalyHistory.Insert(0, new AnomalyHistoryItem
                {
                    Timestamp = result.Timestamp,
                    Score = result.Score,
                    Description = result.Description,
                    Severity = AnomalySeverity
                });

                // Keep only last 10 items
                while (AnomalyHistory.Count > 10)
                    AnomalyHistory.RemoveAt(AnomalyHistory.Count - 1);

                // Generate repair suggestions based on contributing parameters
                await UpdateRepairSuggestionsAsync(result);

                // Auto-expand panel on anomaly
                IsAnomalyPanelExpanded = true;

                // Auto-trigger Panic Mode for critical anomalies
                if (result.Score > 0.98 && CurrentFocusMode != FocusMode.Panic)
                {
                    CurrentFocusMode = FocusMode.Panic;
                }
            }
            else
            {
                AnomalySeverity = "Normal";
                AnomalyDescription = "No anomalies detected";
            }
        }
        catch
        {
            AnomalyDescription = "Analysis unavailable";
        }
    }

    private string ClassifyAnomalySeverity(double score)
    {
        return score switch
        {
            >= 0.9 => "Critical",
            >= 0.7 => "High",
            >= 0.5 => "Medium",
            >= 0.3 => "Low",
            _ => "Normal"
        };
    }

    private double GetParameterContribution(string paramName)
    {
        // Calculate relative contribution based on deviation from baseline
        // This is a simplified calculation - in production, use proper feature attribution
        return paramName switch
        {
            "Engine RPM" or "0C" => Math.Abs(EngineRpm - 2500) / 5500,
            "Vehicle Speed" or "0D" => Math.Abs(VehicleSpeed - 60) / 140,
            "Engine Load" or "04" => Math.Abs(EngineLoad - 30) / 70,
            "Coolant Temp" or "05" => Math.Abs(CoolantTemp - 90) / 40,
            "MAF" or "10" => Math.Abs(MafAirFlow - 15) / 35,
            "Throttle" or "11" => Math.Abs(ThrottlePosition - 15) / 85,
            _ => 0.5
        };
    }

    private async Task UpdateRepairSuggestionsAsync(AnomalyResult result)
    {
        RepairSuggestions.Clear();

        // Generate suggestions based on contributing parameters
        foreach (var param in result.ContributingParameters.Take(3))
        {
            var suggestions = GetSuggestionsForParameter(param);
            foreach (var suggestion in suggestions)
            {
                RepairSuggestions.Add(suggestion);
            }
        }

        await Task.CompletedTask; // Placeholder for async LLM-enhanced suggestions
    }

    private IEnumerable<RepairSuggestion> GetSuggestionsForParameter(string param)
    {
        return param.ToUpperInvariant() switch
        {
            "ENGINE RPM" or "0C" => new[]
            {
                new RepairSuggestion { Priority = 1, Component = "Idle Control", Action = "Check idle air control valve and throttle body", EstimatedCost = "$50-200" },
                new RepairSuggestion { Priority = 2, Component = "Vacuum Leak", Action = "Inspect intake manifold gaskets and vacuum hoses", EstimatedCost = "$30-150" }
            },
            "COOLANT TEMP" or "05" => new[]
            {
                new RepairSuggestion { Priority = 1, Component = "Thermostat", Action = "Replace thermostat if stuck open/closed", EstimatedCost = "$80-150" },
                new RepairSuggestion { Priority = 2, Component = "Coolant Sensor", Action = "Test and replace ECT sensor if faulty", EstimatedCost = "$30-80" }
            },
            "MAF" or "10" => new[]
            {
                new RepairSuggestion { Priority = 1, Component = "MAF Sensor", Action = "Clean MAF sensor with appropriate cleaner", EstimatedCost = "$10-20" },
                new RepairSuggestion { Priority = 2, Component = "Air Filter", Action = "Replace air filter if dirty/clogged", EstimatedCost = "$20-50" }
            },
            "ENGINE LOAD" or "04" => new[]
            {
                new RepairSuggestion { Priority = 1, Component = "Fuel System", Action = "Check fuel pressure and injector performance", EstimatedCost = "$100-400" },
                new RepairSuggestion { Priority = 2, Component = "Ignition System", Action = "Inspect spark plugs and ignition coils", EstimatedCost = "$50-200" }
            },
            _ => new[]
            {
                new RepairSuggestion { Priority = 1, Component = "General", Action = "Perform comprehensive diagnostic scan", EstimatedCost = "$50-100" }
            }
        };
    }

    [RelayCommand]
    private void ToggleAnomalyPanel()
    {
        IsAnomalyPanelExpanded = !IsAnomalyPanelExpanded;
    }

    [RelayCommand]
    private void ClearAnomalyHistory()
    {
        AnomalyHistory.Clear();
        TotalAnomaliesDetected = 0;
    }

    [RelayCommand]
    private async Task RunRulPredictionAsync()
    {
        if (_rulPredictorService == null) return;

        try
        {
            // Generate telemetry data from recent readings
            var telemetryData = GenerateComponentTelemetry();

            var progress = new Progress<RULPredictionProgress>(p =>
            {
                // Could update a progress indicator here
            });

            var prediction = await _rulPredictorService.PredictMaintenanceAsync(
                "CURRENT_VEHICLE",
                50000, // TODO: Get actual odometer from vehicle
                telemetryData,
                50.0,
                progress);

            if (prediction.Success)
            {
                OverallVehicleHealth = prediction.OverallHealth;
                EstimatedMaintenanceCost = prediction.EstimatedMaintenanceCost;
                NextServiceDate = prediction.NextRecommendedService.ToString("MMM dd, yyyy");

                // Update component health items
                ComponentHealthItems.Clear();
                foreach (var comp in prediction.Components.OrderBy(c => c.HealthScore))
                {
                    ComponentHealthItems.Add(new ComponentHealthItem
                    {
                        ComponentName = FormatComponentName(comp.Component),
                        HealthScore = comp.HealthScore,
                        WarningLevel = comp.WarningLevel.ToString(),
                        RemainingLifeKm = comp.EstimatedRulKm,
                        RemainingLifeDays = comp.EstimatedRulDays,
                        DegradationRate = comp.DegradationRate,
                        ContributingFactors = comp.ContributingFactors
                    });
                }

                // Update urgent items
                UrgentMaintenanceItems.Clear();
                foreach (var item in prediction.UrgentItems)
                {
                    UrgentMaintenanceItems.Add(item);
                }

                // Auto-expand panel if there are urgent items
                if (prediction.UrgentItems.Count > 0)
                {
                    IsComponentHealthPanelExpanded = true;
                }
            }
        }
        catch (Exception ex)
        {
            // Log error but don't crash
            System.Diagnostics.Debug.WriteLine($"RUL prediction failed: {ex.Message}");
        }
    }

    private List<ComponentTelemetry> GenerateComponentTelemetry()
    {
        // Generate synthetic telemetry based on current sensor readings
        // In a real implementation, this would come from historical data
        var telemetry = new List<ComponentTelemetry>();

        // Battery health based on voltage patterns
        if (MafAirFlow > 0)
        {
            telemetry.Add(new ComponentTelemetry
            {
                Component = VehicleComponentType.Battery,
                SensorData = GenerateSyntheticHealthData(0.85, 0.02)
            });
        }

        // Spark plugs based on engine performance
        if (EngineRpm > 0)
        {
            var sparkHealth = CalculateSparkPlugHealth();
            telemetry.Add(new ComponentTelemetry
            {
                Component = VehicleComponentType.SparkPlugs,
                SensorData = GenerateSyntheticHealthData(sparkHealth, 0.01)
            });
        }

        // O2 sensor based on fuel mixture
        telemetry.Add(new ComponentTelemetry
        {
            Component = VehicleComponentType.O2Sensor,
            SensorData = GenerateSyntheticHealthData(0.82, 0.015)
        });

        // Catalytic converter
        telemetry.Add(new ComponentTelemetry
        {
            Component = VehicleComponentType.CatalyticConverter,
            SensorData = GenerateSyntheticHealthData(0.90, 0.005)
        });

        // Brake pads (simulated)
        telemetry.Add(new ComponentTelemetry
        {
            Component = VehicleComponentType.BrakePads,
            SensorData = GenerateSyntheticHealthData(0.65, 0.02)
        });

        return telemetry;
    }

    private double CalculateSparkPlugHealth()
    {
        // Estimate spark plug health based on engine smoothness
        var rpmVariance = Math.Abs(EngineRpm - 2000) / 6000;
        var loadFactor = EngineLoad / 100;
        return Math.Max(0.5, 1.0 - (rpmVariance * 0.2 + loadFactor * 0.1));
    }

    private double[] GenerateSyntheticHealthData(double currentHealth, double degradationRate)
    {
        var data = new double[30];
        for (int i = 0; i < 30; i++)
        {
            // Simulate gradual degradation with small noise
            var dayOffset = 30 - i;
            var historicalHealth = Math.Min(1.0, currentHealth + (dayOffset * degradationRate));
            var noise = (Random.Shared.NextDouble() - 0.5) * 0.02;
            data[i] = Math.Clamp(historicalHealth + noise, 0, 1);
        }
        return data;
    }

    private static string FormatComponentName(VehicleComponentType component)
    {
        return component switch
        {
            VehicleComponentType.CatalyticConverter => "Catalytic Converter",
            VehicleComponentType.O2Sensor => "O2 Sensor",
            VehicleComponentType.SparkPlugs => "Spark Plugs",
            VehicleComponentType.Battery => "Battery",
            VehicleComponentType.BrakePads => "Brake Pads",
            VehicleComponentType.AirFilter => "Air Filter",
            VehicleComponentType.FuelFilter => "Fuel Filter",
            VehicleComponentType.TimingBelt => "Timing Belt",
            VehicleComponentType.Coolant => "Coolant",
            VehicleComponentType.TransmissionFluid => "Transmission Fluid",
            _ => component.ToString()
        };
    }

    [RelayCommand]
    private void ToggleComponentHealthPanel()
    {
        IsComponentHealthPanelExpanded = !IsComponentHealthPanelExpanded;
    }

    private async Task StopStreamingAsync()
    {
        _disposables.Dispose();
        _disposables = new CompositeDisposable();
        
        if (_currentSessionId != Guid.Empty)
        {
            await _dbService.EndSessionAsync(_currentSessionId);
            
            // Log audit event
            await _auditService.LogActivityAsync("DiagnosticSessionEnd", _currentSessionId);
            
            _currentSessionId = Guid.Empty;
        }

        IsStreaming = false;
        AnomalyDescription = "Streaming stopped";

        // Push session data to cloud sync
        _ = _syncService.PushChangesAsync();
    }

    public void Dispose()
    {
        _obdService.ConnectionStatusChanged -= OnConnectionStatusChanged;
        _disposables.Dispose();
    }
}

/// <summary>
/// Represents a parameter contributing to an anomaly detection
/// </summary>
public class ContributingParameter
{
    public string Name { get; set; } = string.Empty;
    public double Contribution { get; set; }
    public string ContributionPercent => $"{Contribution * 100:F0}%";
}

/// <summary>
/// Repair suggestion for a detected anomaly
/// </summary>
public class RepairSuggestion
{
    public int Priority { get; set; }
    public string Component { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public string EstimatedCost { get; set; } = string.Empty;
    public string PriorityLabel => Priority switch
    {
        1 => "HIGH",
        2 => "MEDIUM",
        _ => "LOW"
    };
}

/// <summary>
/// Historical anomaly record
/// </summary>
public class AnomalyHistoryItem
{
    public DateTime Timestamp { get; set; }
    public double Score { get; set; }
    public string Description { get; set; } = string.Empty;
    public string Severity { get; set; } = string.Empty;
    public string TimeAgo => GetTimeAgo();
    public string ScorePercent => $"{Score * 100:F0}%";

    private string GetTimeAgo()
    {
        var elapsed = DateTime.UtcNow - Timestamp;
        if (elapsed.TotalMinutes < 1) return "Just now";
        if (elapsed.TotalMinutes < 60) return $"{elapsed.Minutes}m ago";
        if (elapsed.TotalHours < 24) return $"{elapsed.Hours}h ago";
        return Timestamp.ToString("g");
    }
}

/// <summary>
/// Component health item for RUL prediction display
/// </summary>
public class ComponentHealthItem
{
    public string ComponentName { get; set; } = string.Empty;
    public double HealthScore { get; set; }
    public string WarningLevel { get; set; } = "Normal";
    public double RemainingLifeKm { get; set; }
    public int RemainingLifeDays { get; set; }
    public double DegradationRate { get; set; }
    public List<string> ContributingFactors { get; set; } = new();

    public string HealthPercent => $"{HealthScore * 100:F0}%";
    public string RemainingLifeFormatted => RemainingLifeKm >= 1000
        ? $"{RemainingLifeKm / 1000:F1}k km"
        : $"{RemainingLifeKm:F0} km";
    public string RemainingDaysFormatted => RemainingLifeDays >= 365
        ? $"{RemainingLifeDays / 365:F1} years"
        : RemainingLifeDays >= 30
            ? $"{RemainingLifeDays / 30:F0} months"
            : $"{RemainingLifeDays} days";
    public string StatusColor => WarningLevel switch
    {
        "Critical" => "#FF4444",
        "Warning" => "#FFAA00",
        _ => "#44FF44"
    };
}
