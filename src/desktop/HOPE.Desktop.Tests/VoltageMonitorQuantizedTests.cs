using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Testing;
using Xunit;

namespace HOPE.Desktop.Tests;

public class VoltageMonitorQuantizedTests
{
    [Fact]
    public async Task ReadBatteryVoltageAsync_StandardAdapter_ReturnsCorrectStatus()
    {
        // Arrange
        var adapter = new SimulatedHardwareAdapter();
        await adapter.ConnectAsync("MOCK");
        adapter.SetVoltage(13.5);
        var monitor = new VoltageMonitor(adapter);

        // Act
        var reading = await monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(VoltageStatus.WriteSafe, reading.Status);
        Assert.Equal(13.5, reading.Voltage);
    }

    [Fact]
    public async Task ReadBatteryVoltageAsync_ScanmatikQuantizedHigh_ReturnsDiagnosticSafeWithWarning()
    {
        // Arrange
        var mockAdapter = new MockQuantizedAdapter("Scanmatik 2 PRO");
        await mockAdapter.ConnectAsync("MOCK");
        mockAdapter.SetVoltage(13.7);
        var monitor = new VoltageMonitor(mockAdapter);

        // Act
        var reading = await monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(VoltageStatus.DiagnosticSafe, reading.Status);
        Assert.Contains("Quantized", reading.Message);
    }

    [Fact]
    public async Task ReadBatteryVoltageAsync_ScanmatikQuantizedLow_ReturnsCritical()
    {
        // Arrange
        var mockAdapter = new MockQuantizedAdapter("Scanmatik 2 PRO");
        await mockAdapter.ConnectAsync("MOCK");
        mockAdapter.SetVoltage(7.0);
        var monitor = new VoltageMonitor(mockAdapter);

        // Act
        var reading = await monitor.ReadBatteryVoltageAsync();

        // Assert
        Assert.Equal(VoltageStatus.Critical, reading.Status);
    }

    private class MockQuantizedAdapter : SimulatedHardwareAdapter
    {
        private readonly string _name;
        public MockQuantizedAdapter(string name) => _name = name;
        public override string AdapterName => _name;
        public override bool HasQuantizedVoltageReporting => _name.Contains("Scanmatik");
    }
}
