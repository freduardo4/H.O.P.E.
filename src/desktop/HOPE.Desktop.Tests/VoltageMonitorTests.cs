using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Testing;
using Xunit;

namespace HOPE.Desktop.Tests;

public class VoltageMonitorTests
{
    private readonly SimulatedHardwareAdapter _adapter;
    private readonly VoltageMonitor _monitor;

    public VoltageMonitorTests()
    {
        _adapter = new SimulatedHardwareAdapter();
        _adapter.ConnectAsync("test").Wait();
        _monitor = new VoltageMonitor(_adapter);
    }

    [Fact]
    public async Task ReadBatteryVoltage_ReturnsCorrectValues()
    {
        // Arrange
        _adapter.SetVoltage(14.2);

        // Act
        var reading = await _monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(14.2, reading.Voltage);
        Assert.Equal(VoltageStatus.WriteSafe, reading.Status);
    }

    [Theory]
    [InlineData(14.0, VoltageStatus.WriteSafe)]
    [InlineData(12.7, VoltageStatus.DiagnosticSafe)]
    [InlineData(12.2, VoltageStatus.Low)]
    [InlineData(11.8, VoltageStatus.Warning)]
    [InlineData(11.0, VoltageStatus.Critical)]
    public async Task VoltageThresholds_MapToCorrectStatus(double voltage, VoltageStatus expectedStatus)
    {
        // Arrange
        _adapter.SetVoltage(voltage);

        // Act
        var reading = await _monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(expectedStatus, reading.Status);
    }

    [Fact]
    public async Task StartMonitoring_EmitsReadings()
    {
        // Arrange
        var readings = new List<VoltageReading>();
        _monitor.VoltageReadings.Subscribe(r => readings.Add(r));
        _adapter.SetVoltage(13.5);

        // Act
        _monitor.StartMonitoring(intervalMs: 10);
        await Task.Delay(100);
        _monitor.StopMonitoring();

        // Assert
        Assert.NotEmpty(readings);
        Assert.All(readings, r => Assert.Equal(13.5, r.Voltage));
    }

    [Fact]
    public async Task StatusChanged_FiresOnThresholdCross()
    {
        // Arrange
        VoltageStatus? detectedNewStatus = null;
        _monitor.StatusChanged += (s, e) => detectedNewStatus = e.NewStatus;
        
        // Act - Start high
        _adapter.SetVoltage(14.0);
        await _monitor.ReadBatteryVoltageAsync();
        
        // Act - Drop to critical
        _adapter.SetVoltage(11.0);
        await _monitor.ReadBatteryVoltageAsync(); // This triggers the manual update check if not monitoring, 
                                                  // but VoltageMonitor's manual ReadBatteryVoltageAsync also updates _lastReading and triggers logic?
                                                  // Actually ReadBatteryVoltageAsync calls UpdateLastReading but doesn't fire events.
                                                  // Events are fired in the monitoring loop.

        _monitor.StartMonitoring(10);
        await Task.Delay(50);
        _monitor.StopMonitoring();

        // Assert
        Assert.Equal(VoltageStatus.Critical, detectedNewStatus);
    }

    [Theory]
    [InlineData(OperationType.ECUFlash, 13.5, true)]
    [InlineData(OperationType.ECUFlash, 12.0, false)]
    [InlineData(OperationType.Read, 12.0, true)]
    [InlineData(OperationType.Read, 10.0, false)]
    public async Task ValidateForOperation_EnforcesSafeThresholds(OperationType op, double voltage, bool expectedSafe)
    {
        // Arrange
        _adapter.SetVoltage(voltage);

        // Act
        var result = await _monitor.ValidateForOperationAsync(op);

        // Assert
        Assert.Equal(expectedSafe, result.IsValid);
    }

    [Fact]
    public async Task DisconnectedAdapter_ReturnsUnknownStatus()
    {
        // Arrange
        await _adapter.DisconnectAsync();

        // Act
        var reading = await _monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(VoltageStatus.Unknown, reading.Status);
        Assert.Null(reading.Voltage);
    }
}
