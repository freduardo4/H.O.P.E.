using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.Diagnostics;

public class FlightRecorderService : IFlightRecorder, IDisposable
{
    private readonly IHardwareAdapter _hardware;
    private readonly ILogger<FlightRecorderService> _logger;
    private readonly ConcurrentQueue<RecordedFrame> _buffer;
    private readonly int _bufferSize;
    
    private IDisposable? _subscription;
    private bool _isRecording;

    public IObservable<HardwareErrorEventArgs> TriggerEvents => Observable.FromEventPattern<HardwareErrorEventArgs>(
            h => _hardware.ErrorOccurred += h, 
            h => _hardware.ErrorOccurred -= h).Select(x => x.EventArgs);

    public FlightRecorderService(IHardwareAdapter hardware, ILogger<FlightRecorderService> logger, int bufferSize = 10000)
    {
        _hardware = hardware;
        _logger = logger;
        _bufferSize = bufferSize;
        _buffer = new ConcurrentQueue<RecordedFrame>();
        
        // Auto-subscribe to errors to dump? 
        // For now, we just expose the TriggerEvents. The consumer (ViewModel or App) can decide to subscribe and call Dump.
        // OR we can internally wire it up. Let's wire it up internally for "Black Box" behavior.
        
        // We observe errors strictly for auto-dumping logic
        Observable.FromEventPattern<HardwareErrorEventArgs>(
             h => _hardware.ErrorOccurred += h,
             h => _hardware.ErrorOccurred -= h)
             .Subscribe(async evt => 
             {
                 if (_isRecording)
                 {
                    _logger.LogInformation("Flight Recorder Triggered by Hardware Error! Dumping log...");
                     var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                     var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Logs", $"FlightRecorder_Crash_{timestamp}.csv");
                     await DumpToFileAsync(path);
                 }
             });
    }

    public void StartRecording()
    {
        if (_isRecording) return;
        
        _isRecording = true;
        _buffer.Clear();
        _logger.LogInformation("Flight Recorder Started.");

        _subscription = _hardware.StreamMessages().Subscribe(data =>
        {
            var frame = new RecordedFrame(DateTime.UtcNow, data);
            _buffer.Enqueue(frame);
            
            while (_buffer.Count > _bufferSize)
            {
                _buffer.TryDequeue(out _);
            }
        });
    }

    public void StopRecording()
    {
        _isRecording = false;
        _subscription?.Dispose();
        _logger.LogInformation("Flight Recorder Stopped.");
    }

    public async Task DumpToFileAsync(string path)
    {
        try
        {
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            var snapshot = _buffer.ToArray();
            var sb = new StringBuilder();
            sb.AppendLine("Timestamp,Data_Hex");
            
            foreach (var frame in snapshot)
            {
                sb.AppendLine($"{frame.Timestamp:O},{BitConverter.ToString(frame.Data).Replace("-", "")}");
            }

            await File.WriteAllTextAsync(path, sb.ToString());
            _logger.LogInformation($"Flight log dumped to {path}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to dump flight recorder log.");
        }
    }

    public void Dispose()
    {
        StopRecording();
    }

    private record RecordedFrame(DateTime Timestamp, byte[] Data);
}
