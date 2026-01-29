using HOPE.Core.Hardware;

namespace HOPE.Core.Interfaces;

public interface IVoltageMonitor : IDisposable
{
    double? CurrentVoltage { get; }
    VoltageStatus CurrentStatus { get; }
    bool IsWriteOperationSafe { get; }
    bool IsDiagnosticOperationSafe { get; }
    bool IsCritical { get; }
    bool IsMonitoring { get; }
    IObservable<VoltageReading> VoltageReadings { get; }
    IObservable<VoltageWarning> VoltageAlerts { get; }
    event EventHandler<VoltageStatusChangedEventArgs>? StatusChanged;

    Task<VoltageReading> ReadBatteryVoltageAsync(CancellationToken cancellationToken = default);
    void StartMonitoring(int intervalMs = 1000);
    void StopMonitoring();
    Task<VoltageValidationResult> ValidateForOperationAsync(OperationType operationType, CancellationToken cancellationToken = default);
}
