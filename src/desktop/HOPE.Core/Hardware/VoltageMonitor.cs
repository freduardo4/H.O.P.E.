using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Interfaces;

namespace HOPE.Core.Hardware;

/// <summary>
/// Monitors vehicle battery voltage via J2534 adapter.
/// Critical for safe ECU operations - prevents writes during low voltage conditions.
/// </summary>
public class VoltageMonitor : IDisposable
{
    /// <summary>Minimum voltage for safe write operations (engine running or external charger)</summary>
    public const double SAFE_WRITE_THRESHOLD = 13.0;

    /// <summary>Minimum voltage for any diagnostic operations</summary>
    public const double SAFE_DIAGNOSTIC_THRESHOLD = 12.5;

    /// <summary>Critical voltage threshold - should abort all operations</summary>
    public const double CRITICAL_THRESHOLD = 11.5;

    /// <summary>Warning threshold - may indicate failing battery or high parasitic draw</summary>
    public const double WARNING_THRESHOLD = 12.0;

    private readonly IHardwareAdapter _adapter;
    private readonly Subject<VoltageReading> _voltageSubject = new();
    private readonly Subject<VoltageWarning> _warningSubject = new();
    private CancellationTokenSource? _monitorCts;
    private bool _isMonitoring;
    private bool _disposed;
    private VoltageReading? _lastReading;
    private readonly object _lock = new();

    /// <summary>
    /// Gets the current voltage reading
    /// </summary>
    public double? CurrentVoltage => _lastReading?.Voltage;

    /// <summary>
    /// Gets the current voltage status
    /// </summary>
    public VoltageStatus CurrentStatus => _lastReading?.Status ?? VoltageStatus.Unknown;

    /// <summary>
    /// Gets whether write operations are currently safe
    /// </summary>
    public bool IsWriteOperationSafe => CurrentVoltage >= SAFE_WRITE_THRESHOLD;

    /// <summary>
    /// Gets whether diagnostic operations are currently safe
    /// </summary>
    public bool IsDiagnosticOperationSafe => CurrentVoltage >= SAFE_DIAGNOSTIC_THRESHOLD;

    /// <summary>
    /// Gets whether the voltage is in critical range
    /// </summary>
    public bool IsCritical => CurrentVoltage < CRITICAL_THRESHOLD;

    /// <summary>
    /// Gets whether monitoring is active
    /// </summary>
    public bool IsMonitoring => _isMonitoring;

    /// <summary>
    /// Observable stream of voltage readings
    /// </summary>
    public IObservable<VoltageReading> VoltageReadings => _voltageSubject.AsObservable();

    /// <summary>
    /// Observable stream of voltage warnings
    /// </summary>
    public IObservable<VoltageWarning> VoltageAlerts => _warningSubject.AsObservable();

    /// <summary>
    /// Event raised when voltage status changes
    /// </summary>
    public event EventHandler<VoltageStatusChangedEventArgs>? StatusChanged;

    public VoltageMonitor(IHardwareAdapter adapter)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));

        if (!adapter.SupportsVoltageMonitoring)
        {
            throw new NotSupportedException($"Adapter {adapter.AdapterName} does not support voltage monitoring");
        }
    }

    /// <summary>
    /// Read battery voltage once
    /// </summary>
    public async Task<VoltageReading> ReadBatteryVoltageAsync(CancellationToken cancellationToken = default)
    {
        if (!_adapter.IsConnected)
        {
            return new VoltageReading
            {
                Timestamp = DateTime.UtcNow,
                Voltage = null,
                Status = VoltageStatus.Unknown,
                Message = "Adapter not connected"
            };
        }

        var voltage = await _adapter.ReadBatteryVoltageAsync(cancellationToken);
        var isQuantized = _adapter.HasQuantizedVoltageReporting;

        var reading = new VoltageReading
        {
            Timestamp = DateTime.UtcNow,
            Voltage = voltage,
            Status = DetermineStatus(voltage, isQuantized),
            Message = GetStatusMessage(voltage, isQuantized)
        };

        UpdateLastReading(reading);

        return reading;
    }

    /// <summary>
    /// Start continuous voltage monitoring
    /// </summary>
    /// <param name="intervalMs">Monitoring interval in milliseconds (default 1000ms)</param>
    public void StartMonitoring(int intervalMs = 1000)
    {
        if (_isMonitoring) return;

        _monitorCts = new CancellationTokenSource();
        _isMonitoring = true;

        Task.Run(async () =>
        {
            var previousStatus = VoltageStatus.Unknown;

            while (!_monitorCts.Token.IsCancellationRequested)
            {
                try
                {
                    var reading = await ReadBatteryVoltageAsync(_monitorCts.Token);
                    _voltageSubject.OnNext(reading);

                    // Check for status change
                    if (reading.Status != previousStatus)
                    {
                        StatusChanged?.Invoke(this, new VoltageStatusChangedEventArgs(
                            previousStatus, reading.Status, reading.Voltage));

                        // Emit warning if status degraded
                        if (ShouldEmitWarning(previousStatus, reading.Status))
                        {
                            _warningSubject.OnNext(new VoltageWarning
                            {
                                Timestamp = DateTime.UtcNow,
                                Voltage = reading.Voltage,
                                Status = reading.Status,
                                WarningLevel = GetWarningLevel(reading.Status),
                                Message = reading.Message,
                                RecommendedAction = GetRecommendedAction(reading.Status)
                            });
                        }

                        previousStatus = reading.Status;
                    }

                    await Task.Delay(intervalMs, _monitorCts.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    _warningSubject.OnNext(new VoltageWarning
                    {
                        Timestamp = DateTime.UtcNow,
                        Voltage = null,
                        Status = VoltageStatus.Unknown,
                        WarningLevel = WarningLevel.Error,
                        Message = $"Voltage monitoring error: {ex.Message}",
                        RecommendedAction = "Check adapter connection"
                    });

                    await Task.Delay(5000, _monitorCts.Token); // Wait longer after error
                }
            }

            _isMonitoring = false;
        }, _monitorCts.Token);
    }

    /// <summary>
    /// Stop continuous voltage monitoring
    /// </summary>
    public void StopMonitoring()
    {
        _monitorCts?.Cancel();
        _isMonitoring = false;
    }

    /// <summary>
    /// Validate voltage is safe for a specific operation
    /// </summary>
    /// <param name="operationType">Type of operation to validate</param>
    /// <returns>Validation result with details</returns>
    public async Task<VoltageValidationResult> ValidateForOperationAsync(
        OperationType operationType,
        CancellationToken cancellationToken = default)
    {
        var reading = await ReadBatteryVoltageAsync(cancellationToken);

        double requiredVoltage = operationType switch
        {
            OperationType.ECUFlash => SAFE_WRITE_THRESHOLD,
            OperationType.ECUWrite => SAFE_WRITE_THRESHOLD,
            OperationType.BiDirectionalControl => SAFE_DIAGNOSTIC_THRESHOLD,
            OperationType.DTCClear => SAFE_DIAGNOSTIC_THRESHOLD,
            OperationType.Diagnostic => SAFE_DIAGNOSTIC_THRESHOLD,
            OperationType.Read => CRITICAL_THRESHOLD,
            _ => SAFE_DIAGNOSTIC_THRESHOLD
        };

        bool isSafe = reading.Voltage.HasValue && reading.Voltage.Value >= requiredVoltage;

        return new VoltageValidationResult
        {
            IsValid = isSafe,
            CurrentVoltage = reading.Voltage,
            RequiredVoltage = requiredVoltage,
            OperationType = operationType,
            Status = reading.Status,
            Message = isSafe
                ? $"Voltage OK for {operationType}: {reading.Voltage:F2}V >= {requiredVoltage:F1}V"
                : $"Voltage too low for {operationType}: {reading.Voltage?.ToString("F2") ?? "unknown"}V < {requiredVoltage:F1}V required",
            Recommendation = isSafe ? null : GetRecommendedAction(reading.Status)
        };
    }

    private void UpdateLastReading(VoltageReading reading)
    {
        lock (_lock)
        {
            _lastReading = reading;
        }
    }

    private static VoltageStatus DetermineStatus(double? voltage, bool isQuantized = false)
    {
        if (!voltage.HasValue) return VoltageStatus.Unknown;

        if (isQuantized)
        {
            // Scanmatik specific: 7V or 13.7V
            if (voltage.Value >= 13.7) return VoltageStatus.DiagnosticSafe; // We can't be sure it's 13.0+ precisely
            if (voltage.Value <= 7.0) return VoltageStatus.Critical;
            return VoltageStatus.Low;
        }

        return voltage.Value switch
        {
            < CRITICAL_THRESHOLD => VoltageStatus.Critical,
            < WARNING_THRESHOLD => VoltageStatus.Warning,
            < SAFE_DIAGNOSTIC_THRESHOLD => VoltageStatus.Low,
            < SAFE_WRITE_THRESHOLD => VoltageStatus.DiagnosticSafe,
            _ => VoltageStatus.WriteSafe
        };
    }

    private static string GetStatusMessage(double? voltage, bool isQuantized = false)
    {
        if (!voltage.HasValue) return "Unable to read battery voltage";

        if (isQuantized && voltage.Value >= 13.7)
        {
            return "OK (Quantized): Battery reported as 13.7V. Precise voltage unknown. Manual verification recommended.";
        }

        return voltage.Value switch
        {
            < CRITICAL_THRESHOLD => $"CRITICAL: Battery voltage {voltage:F2}V is dangerously low! Abort all operations.",
            < WARNING_THRESHOLD => $"WARNING: Battery voltage {voltage:F2}V is low. Check battery health.",
            < SAFE_DIAGNOSTIC_THRESHOLD => $"LOW: Battery voltage {voltage:F2}V. Diagnostic operations may be unreliable.",
            < SAFE_WRITE_THRESHOLD => $"OK: Battery voltage {voltage:F2}V is safe for diagnostics. Start engine for ECU writes.",
            _ => $"OPTIMAL: Battery voltage {voltage:F2}V is safe for all operations."
        };
    }

    private static bool ShouldEmitWarning(VoltageStatus previous, VoltageStatus current)
    {
        // Emit warning when status degrades
        return current switch
        {
            VoltageStatus.Critical => true,
            VoltageStatus.Warning when previous != VoltageStatus.Critical && previous != VoltageStatus.Warning => true,
            VoltageStatus.Low when previous == VoltageStatus.DiagnosticSafe || previous == VoltageStatus.WriteSafe => true,
            _ => false
        };
    }

    private static WarningLevel GetWarningLevel(VoltageStatus status)
    {
        return status switch
        {
            VoltageStatus.Critical => WarningLevel.Critical,
            VoltageStatus.Warning => WarningLevel.Warning,
            VoltageStatus.Low => WarningLevel.Info,
            _ => WarningLevel.Info
        };
    }

    private static string GetRecommendedAction(VoltageStatus status)
    {
        return status switch
        {
            VoltageStatus.Critical => "STOP all operations immediately. Connect a battery charger or start the engine.",
            VoltageStatus.Warning => "Check battery condition. Do not perform ECU writes. Connect a charger if possible.",
            VoltageStatus.Low => "Start the engine or connect a battery charger before performing write operations.",
            VoltageStatus.DiagnosticSafe => "Start the engine or connect an external charger for ECU write operations.",
            _ => "Ensure the vehicle is connected and the adapter is functioning."
        };
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        StopMonitoring();
        _voltageSubject.Dispose();
        _warningSubject.Dispose();
        _monitorCts?.Dispose();
    }
}

/// <summary>
/// Battery voltage reading result
/// </summary>
public class VoltageReading
{
    public DateTime Timestamp { get; set; }
    public double? Voltage { get; set; }
    public VoltageStatus Status { get; set; }
    public string Message { get; set; } = string.Empty;
}

/// <summary>
/// Battery voltage status levels
/// </summary>
public enum VoltageStatus
{
    /// <summary>Unable to read voltage</summary>
    Unknown,

    /// <summary>Below 11.5V - abort all operations</summary>
    Critical,

    /// <summary>Below 12.0V - battery health concern</summary>
    Warning,

    /// <summary>Below 12.5V - may affect reliability</summary>
    Low,

    /// <summary>12.5V+ - safe for diagnostics</summary>
    DiagnosticSafe,

    /// <summary>13.0V+ - safe for ECU writes (engine running)</summary>
    WriteSafe
}

/// <summary>
/// Voltage warning notification
/// </summary>
public class VoltageWarning
{
    public DateTime Timestamp { get; set; }
    public double? Voltage { get; set; }
    public VoltageStatus Status { get; set; }
    public WarningLevel WarningLevel { get; set; }
    public string Message { get; set; } = string.Empty;
    public string RecommendedAction { get; set; } = string.Empty;
}

/// <summary>
/// Warning severity levels
/// </summary>
public enum WarningLevel
{
    Info,
    Warning,
    Critical,
    Error
}

/// <summary>
/// Types of operations that may require voltage validation
/// </summary>
public enum OperationType
{
    Read,
    Diagnostic,
    DTCClear,
    BiDirectionalControl,
    ECUWrite,
    ECUFlash
}

/// <summary>
/// Result of voltage validation for an operation
/// </summary>
public class VoltageValidationResult
{
    public bool IsValid { get; set; }
    public double? CurrentVoltage { get; set; }
    public double RequiredVoltage { get; set; }
    public OperationType OperationType { get; set; }
    public VoltageStatus Status { get; set; }
    public string Message { get; set; } = string.Empty;
    public string? Recommendation { get; set; }
}

/// <summary>
/// Event args for voltage status changes
/// </summary>
public class VoltageStatusChangedEventArgs : EventArgs
{
    public VoltageStatus PreviousStatus { get; }
    public VoltageStatus NewStatus { get; }
    public double? CurrentVoltage { get; }

    public VoltageStatusChangedEventArgs(VoltageStatus previous, VoltageStatus current, double? voltage)
    {
        PreviousStatus = previous;
        NewStatus = current;
        CurrentVoltage = voltage;
    }
}
