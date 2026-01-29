using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace HOPE.Core.Services.Diagnostics;

public class LogDataPoint
{
    public DateTime Timestamp { get; set; }
    public double Rpm { get; set; }
    public double Load { get; set; }
    public double Afr { get; set; }
    public double TargetAfr { get; set; }
}

public class LogReplayService
{
    private List<LogDataPoint> _loadedLog = new();
    private CancellationTokenSource? _playbackCts;
    public event Action<LogDataPoint>? DataPointReplayed;
    public event Action? PlaybackFinished;

    public int TotalPoints => _loadedLog.Count;
    public bool IsPlaying => _playbackCts != null && !_playbackCts.IsCancellationRequested;

    public async Task LoadLogAsync(string filePath)
    {
        _loadedLog.Clear();
        // Simple CSV parser assuming columns: Time,RPM,Load,AFR,TargetAFR
        var lines = await File.ReadAllLinesAsync(filePath);
        foreach (var line in lines)
        {
            // Skip header or invalid lines checking here simplified
            var parts = line.Split(',');
            if (parts.Length >= 4 && double.TryParse(parts[1], out var rpm))
            {
                _loadedLog.Add(new LogDataPoint
                {
                    Timestamp = DateTime.UtcNow, // Placeholder
                    Rpm = rpm,
                    Load = double.Parse(parts[2]),
                    Afr = double.Parse(parts[3]),
                    TargetAfr = parts.Length > 4 ? double.Parse(parts[4]) : 14.7
                });
            }
        }
    }

    public void Play(int speedMultiplier = 1)
    {
        if (IsPlaying) return;
        _playbackCts = new CancellationTokenSource();
        
        Task.Run(async () => 
        {
            try 
            {
                foreach (var point in _loadedLog)
                {
                    if (_playbackCts.Token.IsCancellationRequested) break;
                    
                    ProcessPoint(point);
                    await Task.Delay(1000 / (10 * speedMultiplier)); // Simulate 10Hz log at variable speed
                }
            }
            finally
            {
                _playbackCts = null;
                PlaybackFinished?.Invoke();
            }
        }, _playbackCts.Token);
    }

    public void Stop()
    {
        _playbackCts?.Cancel();
    }

    /// <summary>
    /// Manually process a data point, firing the replayed event.
    /// Useful for mock injection or bridging live data.
    /// </summary>
    public void ProcessPoint(LogDataPoint point)
    {
        DataPointReplayed?.Invoke(point);
    }
}
