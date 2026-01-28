using System.Net;
using System.Net.Sockets;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using HOPE.Core.Models.Simulation;
using HOPE.Core.Services.Simulation;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Simulation;

public class BeamNgServiceTests
{
    private readonly Mock<ILogger<BeamNgService>> _loggerMock = new();

    [Fact]
    public async Task StartAsync_ShouldListenOnPortAndReceiveTelemetry()
    {
        // Arrange
        var service = new BeamNgService(_loggerMock.Object);
        int testPort = 5555;
        var telemetryReceived = new TaskCompletionSource<BeamNgTelemetry>();

        service.TelemetryStream.Subscribe(t => telemetryReceived.TrySetResult(t));

        await service.StartAsync(testPort);

        // Act: Send a dummy OutGauge packet
        var testTelemetry = new BeamNgTelemetry
        {
            Time = 1234,
            Car = "ESC\0".ToCharArray(),
            Rpm = 3000,
            Speed = 25.5f,
            Gear = 3
        };

        byte[] packet = SerializeTelemetry(testTelemetry);
        using var client = new UdpClient();
        await client.SendAsync(packet, packet.Length, "127.0.0.1", testPort);

        // Assert
        var received = await telemetryReceived.Task.WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(1234u, received.Time);
        Assert.Equal(3000f, received.Rpm);
        Assert.Equal(25.5f, received.Speed);
        Assert.Equal(3, received.Gear);

        await service.StopAsync();
    }

    private byte[] SerializeTelemetry(BeamNgTelemetry telemetry)
    {
        int size = Marshal.SizeOf<BeamNgTelemetry>();
        byte[] arr = new byte[size];
        IntPtr ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.StructureToPtr(telemetry, ptr, true);
            Marshal.Copy(ptr, arr, 0, size);
            return arr;
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }
}
