using HOPE.Core.Interfaces;
using HOPE.Core.Services.Protocols;
using HOPE.Core.Testing;
using Xunit;

namespace HOPE.Desktop.Tests;

public class SimulatedHardwareAdapterTests
{
    private readonly SimulatedHardwareAdapter _adapter;

    public SimulatedHardwareAdapterTests()
    {
        _adapter = new SimulatedHardwareAdapter();
    }

    [Fact]
    public async Task ConnectAsync_SetsIsConnected()
    {
        var result = await _adapter.ConnectAsync("COM1");
        Assert.True(result);
        Assert.True(_adapter.IsConnected);
    }

    [Fact]
    public async Task SendMessage_DiagnosticSessionControl_ReturnsPositive()
    {
        await _adapter.ConnectAsync("COM1");
        var request = new byte[] { UdsServiceId.DiagnosticSessionControl, (byte)UdsSession.ExtendedDiagnostic };
        var response = await _adapter.SendMessageAsync(request);

        Assert.Equal(2, response.Length);
        Assert.Equal((byte)(UdsServiceId.DiagnosticSessionControl + 0x40), response[0]);
        Assert.Equal((byte)UdsSession.ExtendedDiagnostic, response[1]);
    }

    [Fact]
    public async Task SendMessage_SecurityAccess_RequestSeed_ReturnsSeed()
    {
        await _adapter.ConnectAsync("COM1");
        var request = new byte[] { UdsServiceId.SecurityAccess, 0x01 };
        var response = await _adapter.SendMessageAsync(request);

        Assert.Equal(6, response.Length);
        Assert.Equal((byte)(UdsServiceId.SecurityAccess + 0x40), response[0]);
        Assert.Equal(0x01, response[1]);
        // Seed is 0x12, 0x34, 0x56, 0x78
        Assert.Equal(0x12, response[2]);
    }

    [Fact]
    public async Task SendMessage_SecurityAccess_SendValidKey_ReturnsPositive()
    {
        await _adapter.ConnectAsync("COM1");
        // Trigger seed first
        await _adapter.SendMessageAsync(new byte[] { UdsServiceId.SecurityAccess, 0x01 });
        
        // Key is seed + 1
        var keyRequest = new byte[] { UdsServiceId.SecurityAccess, 0x02, 0x13, 0x35, 0x57, 0x79 };
        var response = await _adapter.SendMessageAsync(keyRequest);

        Assert.Equal(2, response.Length);
        Assert.Equal((byte)(UdsServiceId.SecurityAccess + 0x40), response[0]);
        Assert.Equal(0x02, response[1]);
    }

    [Fact]
    public async Task SendMessage_SecurityAccess_SendInvalidKey_ReturnsNegative()
    {
        await _adapter.ConnectAsync("COM1");
        await _adapter.SendMessageAsync(new byte[] { UdsServiceId.SecurityAccess, 0x01 });
        
        var keyRequest = new byte[] { UdsServiceId.SecurityAccess, 0x02, 0x00, 0x00, 0x00, 0x00 };
        var response = await _adapter.SendMessageAsync(keyRequest);

        Assert.Equal(3, response.Length);
        Assert.Equal(0x7F, response[0]);
        Assert.Equal(UdsServiceId.SecurityAccess, response[1]);
        Assert.Equal((byte)UdsNrc.InvalidKey, response[2]);
    }

    [Fact]
    public async Task SendMessage_InjectError_ReturnsEmptyAndTriggersEvent()
    {
        await _adapter.ConnectAsync("COM1");
        _adapter.InjectError = true;
        _adapter.InjectedErrorType = HardwareErrorType.Timeout;

        HardwareErrorEventArgs? raisedEvent = null;
        _adapter.ErrorOccurred += (sender, args) => raisedEvent = args;

        var response = await _adapter.SendMessageAsync(new byte[] { 0x10, 0x01 });

        Assert.Empty(response);
        Assert.NotNull(raisedEvent);
        Assert.Equal(HardwareErrorType.Timeout, raisedEvent.ErrorType);
    }

    [Fact]
    public async Task ReadBatteryVoltage_ReturnsSimulatedValue()
    {
        _adapter.SimulatedVoltage = 14.2;
        var voltage = await _adapter.ReadBatteryVoltageAsync();
        Assert.Equal(14.2, voltage);
    }
}
