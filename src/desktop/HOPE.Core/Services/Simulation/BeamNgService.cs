using System.Net;
using System.Net.Sockets;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Runtime.InteropServices;
using HOPE.Core.Models.Simulation;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.Simulation;

public class BeamNgService : IBeamNgService
{
    private readonly ILogger<BeamNgService> _logger;
    private readonly Subject<BeamNgTelemetry> _telemetrySubject = new();
    private UdpClient? _udpClient;
    private CancellationTokenSource? _cts;
    private bool _isConnected;

    public BeamNgService(ILogger<BeamNgService> logger)
    {
        _logger = logger;
    }

    public IObservable<BeamNgTelemetry> TelemetryStream => _telemetrySubject.AsObservable();

    public bool IsConnected => _isConnected;

    public async Task StartAsync(int port = 4444, CancellationToken ct = default)
    {
        if (_isConnected) return;

        try
        {
            _udpClient = new UdpClient(port);
            _cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            _isConnected = true;

            _ = Task.Run(() => ListenAsync(_cts.Token), _cts.Token);
            _logger.LogInformation("BeamNG OutGauge listener started on port {Port}", port);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start BeamNG OutGauge listener on port {Port}", port);
            throw;
        }
    }

    private async Task ListenAsync(CancellationToken ct)
    {
        while (!ct.IsCancellationRequested)
        {
            try
            {
                var result = await _udpClient!.ReceiveAsync(ct);
                if (result.Buffer.Length > 0)
                {
                    var telemetry = ParseTelemetry(result.Buffer);
                    _telemetrySubject.OnNext(telemetry);
                }
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error receiving BeamNG telemetry");
                if (ct.IsCancellationRequested) break;
                await Task.Delay(1000, ct); // Retry after delay
            }
        }
    }

    private BeamNgTelemetry ParseTelemetry(byte[] buffer)
    {
        // OutGauge packets are typically 96 bytes with displays, or 48-64 without.
        // We'll use Marshal to parse the struct.
        int size = Marshal.SizeOf<BeamNgTelemetry>();
        
        // If buffer is smaller, we might need a smaller struct or padding.
        // BeamNG typically sends the full 96 bytes if configured.
        byte[] paddedBuffer = buffer;
        if (buffer.Length < size)
        {
            paddedBuffer = new byte[size];
            Array.Copy(buffer, paddedBuffer, buffer.Length);
        }

        IntPtr ptr = Marshal.AllocHGlobal(size);
        try
        {
            Marshal.Copy(paddedBuffer, 0, ptr, size);
            return Marshal.PtrToStructure<BeamNgTelemetry>(ptr);
        }
        finally
        {
            Marshal.FreeHGlobal(ptr);
        }
    }

    public Task StopAsync()
    {
        _cts?.Cancel();
        _udpClient?.Dispose();
        _udpClient = null;
        _isConnected = false;
        _logger.LogInformation("BeamNG OutGauge listener stopped");
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        StopAsync().GetAwaiter().GetResult();
        _telemetrySubject.Dispose();
        _cts?.Dispose();
    }
}
