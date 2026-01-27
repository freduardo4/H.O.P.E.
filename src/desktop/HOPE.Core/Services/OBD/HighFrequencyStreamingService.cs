using System.Collections.Concurrent;
using System.Diagnostics;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using HOPE.Core.Interfaces;
using HOPE.Core.Models;

namespace HOPE.Core.Services.OBD;

/// <summary>
/// High-frequency OBD2 data streaming service supporting 10-50Hz sampling rates.
/// Optimized for J2534 hardware with hardware-level timing control.
/// </summary>
public class HighFrequencyStreamingService : IDisposable
{
    private readonly IHardwareAdapter _adapter;
    private readonly Subject<OBD2Frame> _frameSubject = new();
    private readonly Subject<StreamStatistics> _statsSubject = new();
    private readonly ConcurrentDictionary<string, OBD2Reading> _latestReadings = new();
    private readonly ConcurrentQueue<OBD2Frame> _frameBuffer = new();

    private CancellationTokenSource? _streamingCts;
    private Task? _streamingTask;
    private StreamingMode _currentMode = StreamingMode.Stopped;
    private readonly Stopwatch _sessionStopwatch = new();
    private long _totalFrames;
    private long _errorFrames;
    private readonly object _stateLock = new();

    /// <summary>
    /// Observable stream of OBD2 frames at high frequency
    /// </summary>
    public IObservable<OBD2Frame> FrameStream => _frameSubject.AsObservable();

    /// <summary>
    /// Observable stream of statistics updates
    /// </summary>
    public IObservable<StreamStatistics> StatisticsStream => _statsSubject.AsObservable();

    /// <summary>
    /// Gets the current streaming mode
    /// </summary>
    public StreamingMode CurrentMode => _currentMode;

    /// <summary>
    /// Gets whether high-frequency mode is supported by current adapter
    /// </summary>
    public bool SupportsHighFrequency => _adapter.SupportsHighFrequency;

    /// <summary>
    /// Gets the latest reading for a specific PID
    /// </summary>
    public OBD2Reading? GetLatestReading(string pid) =>
        _latestReadings.TryGetValue(pid.ToUpper(), out var reading) ? reading : null;

    /// <summary>
    /// Gets all latest readings as a snapshot
    /// </summary>
    public IReadOnlyDictionary<string, OBD2Reading> GetLatestReadings() =>
        new Dictionary<string, OBD2Reading>(_latestReadings);

    public HighFrequencyStreamingService(IHardwareAdapter adapter)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
    }

    /// <summary>
    /// Start high-frequency streaming at the specified rate
    /// </summary>
    /// <param name="config">Streaming configuration</param>
    /// <param name="cancellationToken">Cancellation token</param>
    public async Task StartStreamingAsync(HighFrequencyConfig config, CancellationToken cancellationToken = default)
    {
        if (!_adapter.IsConnected)
            throw new InvalidOperationException("Hardware adapter is not connected");

        lock (_stateLock)
        {
            if (_currentMode != StreamingMode.Stopped)
                throw new InvalidOperationException($"Streaming is already {_currentMode}");

            _currentMode = StreamingMode.Starting;
        }

        _streamingCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        _totalFrames = 0;
        _errorFrames = 0;
        _sessionStopwatch.Restart();

        // Configure the hardware for high-frequency mode
        await ConfigureHighFrequencyModeAsync(config, _streamingCts.Token);

        // Start the appropriate streaming strategy
        _streamingTask = config.Strategy switch
        {
            StreamingStrategy.ContinuousPolling => RunContinuousPollingAsync(config, _streamingCts.Token),
            StreamingStrategy.BurstMode => RunBurstModeAsync(config, _streamingCts.Token),
            StreamingStrategy.CANPassiveMonitor => RunPassiveMonitorAsync(config, _streamingCts.Token),
            _ => throw new ArgumentException($"Unknown streaming strategy: {config.Strategy}")
        };

        lock (_stateLock)
        {
            _currentMode = StreamingMode.Running;
        }
    }

    /// <summary>
    /// Stop the streaming
    /// </summary>
    public async Task StopStreamingAsync()
    {
        lock (_stateLock)
        {
            if (_currentMode == StreamingMode.Stopped)
                return;

            _currentMode = StreamingMode.Stopping;
        }

        _streamingCts?.Cancel();

        if (_streamingTask != null)
        {
            try
            {
                await _streamingTask.ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                // Expected
            }
        }

        _sessionStopwatch.Stop();

        // Publish final statistics
        PublishStatistics();

        lock (_stateLock)
        {
            _currentMode = StreamingMode.Stopped;
        }
    }

    /// <summary>
    /// Pause streaming temporarily
    /// </summary>
    public void Pause()
    {
        lock (_stateLock)
        {
            if (_currentMode == StreamingMode.Running)
            {
                _currentMode = StreamingMode.Paused;
                _sessionStopwatch.Stop();
            }
        }
    }

    /// <summary>
    /// Resume streaming after pause
    /// </summary>
    public void Resume()
    {
        lock (_stateLock)
        {
            if (_currentMode == StreamingMode.Paused)
            {
                _currentMode = StreamingMode.Running;
                _sessionStopwatch.Start();
            }
        }
    }

    private async Task ConfigureHighFrequencyModeAsync(HighFrequencyConfig config, CancellationToken ct)
    {
        if (_adapter.Type == HardwareType.ELM327)
        {
            // Configure ELM327 for faster response
            await _adapter.SendCommandAsync("ATST" + (config.ResponseTimeoutMs / 4).ToString("X2"), ct); // Reduce timeout
            await _adapter.SendCommandAsync("ATAL", ct); // Allow long messages
            await _adapter.SendCommandAsync("ATH1", ct); // Headers on for identification

            if (config.AdaptiveTiming)
            {
                await _adapter.SendCommandAsync("ATAT1", ct); // Adaptive timing on
            }
        }
        else if (_adapter.Type == HardwareType.J2534)
        {
            // J2534 devices handle timing at hardware level
            // Configure filter for specific PIDs if needed
        }
    }

    private async Task RunContinuousPollingAsync(HighFrequencyConfig config, CancellationToken ct)
    {
        var intervalMs = 1000.0 / config.TargetFrequencyHz;
        var pids = config.PIDs.ToArray();
        var pidIndex = 0;
        var lastPublishTime = DateTime.UtcNow;

        while (!ct.IsCancellationRequested)
        {
            if (_currentMode == StreamingMode.Paused)
            {
                await Task.Delay(50, ct);
                continue;
            }

            var frameStart = Stopwatch.GetTimestamp();

            try
            {
                // Round-robin through PIDs for maximum data rate
                var pid = pids[pidIndex];
                pidIndex = (pidIndex + 1) % pids.Length;

                var command = $"01{pid}";
                var response = await _adapter.SendCommandAsync(command, ct);

                if (!string.IsNullOrEmpty(response) && !response.Contains("NO DATA") && !response.Contains("ERROR"))
                {
                    var frame = ParseResponseToFrame(response, pid);
                    if (frame != null)
                    {
                        _latestReadings[pid] = frame.Reading;
                        _frameSubject.OnNext(frame);
                        _frameBuffer.Enqueue(frame);

                        // Keep buffer bounded
                        while (_frameBuffer.Count > config.BufferSize)
                            _frameBuffer.TryDequeue(out _);

                        Interlocked.Increment(ref _totalFrames);
                    }
                }
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref _errorFrames);
                // Log error but continue streaming
                System.Diagnostics.Debug.WriteLine($"Streaming error: {ex.Message}");
            }

            // Publish statistics periodically
            if ((DateTime.UtcNow - lastPublishTime).TotalMilliseconds >= config.StatsIntervalMs)
            {
                PublishStatistics();
                lastPublishTime = DateTime.UtcNow;
            }

            // Calculate remaining time to maintain target frequency
            var elapsed = Stopwatch.GetElapsedTime(frameStart).TotalMilliseconds;
            var delay = Math.Max(0, intervalMs / pids.Length - elapsed);

            if (delay > 0)
                await Task.Delay(TimeSpan.FromMilliseconds(delay), ct);
        }
    }

    private async Task RunBurstModeAsync(HighFrequencyConfig config, CancellationToken ct)
    {
        // Request all PIDs in a single burst using ELM327 multi-response mode
        var pids = string.Join("", config.PIDs);
        var lastPublishTime = DateTime.UtcNow;

        while (!ct.IsCancellationRequested)
        {
            if (_currentMode == StreamingMode.Paused)
            {
                await Task.Delay(50, ct);
                continue;
            }

            try
            {
                // Use ELM327 burst request if supported
                var command = $"01{pids}";
                var response = await _adapter.SendCommandAsync(command, ct);

                var frames = ParseBurstResponse(response, config.PIDs);
                foreach (var frame in frames)
                {
                    _latestReadings[frame.PID] = frame.Reading;
                    _frameSubject.OnNext(frame);
                    Interlocked.Increment(ref _totalFrames);
                }
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref _errorFrames);
                System.Diagnostics.Debug.WriteLine($"Burst mode error: {ex.Message}");
            }

            // Publish statistics periodically
            if ((DateTime.UtcNow - lastPublishTime).TotalMilliseconds >= config.StatsIntervalMs)
            {
                PublishStatistics();
                lastPublishTime = DateTime.UtcNow;
            }

            await Task.Delay(TimeSpan.FromMilliseconds(1000.0 / config.TargetFrequencyHz), ct);
        }
    }

    private async Task RunPassiveMonitorAsync(HighFrequencyConfig config, CancellationToken ct)
    {
        // J2534 passive CAN monitoring - highest frequency mode
        if (_adapter.Type != HardwareType.J2534)
            throw new NotSupportedException("Passive CAN monitoring requires J2534 adapter");

        var lastPublishTime = DateTime.UtcNow;

        // Subscribe to raw message stream from J2534
        using var subscription = _adapter.StreamMessages()
            .Where(msg => msg.Length >= 3) // Minimum valid CAN message
            .Subscribe(msg =>
            {
                try
                {
                    var frame = ParseCANMessage(msg, config.PIDs);
                    if (frame != null)
                    {
                        _latestReadings[frame.PID] = frame.Reading;
                        _frameSubject.OnNext(frame);
                        Interlocked.Increment(ref _totalFrames);
                    }
                }
                catch (Exception ex)
                {
                    Interlocked.Increment(ref _errorFrames);
                    System.Diagnostics.Debug.WriteLine($"CAN parse error: {ex.Message}");
                }
            });

        while (!ct.IsCancellationRequested)
        {
            if ((DateTime.UtcNow - lastPublishTime).TotalMilliseconds >= config.StatsIntervalMs)
            {
                PublishStatistics();
                lastPublishTime = DateTime.UtcNow;
            }

            await Task.Delay(100, ct);
        }
    }

    private OBD2Frame? ParseResponseToFrame(string response, string pid)
    {
        try
        {
            // Remove whitespace and parse hex response
            var cleaned = response.Replace(" ", "").Replace("\r", "").Replace("\n", "").ToUpper();

            // Expected format: 41XX... where XX is the PID
            if (!cleaned.StartsWith("41"))
                return null;

            var responsePid = cleaned.Substring(2, 2);
            if (responsePid != pid.ToUpper())
                return null;

            var dataBytes = cleaned.Substring(4);
            var reading = OBD2ResponseParser.ParseMode01Response(response, pid);

            if (reading == null)
                return null;

            return new OBD2Frame
            {
                Timestamp = DateTime.UtcNow,
                TimestampTicks = Stopwatch.GetTimestamp(),
                PID = pid,
                RawData = HexStringToBytes(dataBytes),
                Reading = reading,
                FrameNumber = _totalFrames
            };
        }
        catch
        {
            return null;
        }
    }

    private List<OBD2Frame> ParseBurstResponse(string response, IEnumerable<string> expectedPids)
    {
        var frames = new List<OBD2Frame>();
        var lines = response.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var cleaned = line.Replace(" ", "").ToUpper();
            if (cleaned.StartsWith("41") && cleaned.Length >= 4)
            {
                var pid = cleaned.Substring(2, 2);
                if (expectedPids.Any(p => p.ToUpper() == pid))
                {
                    var frame = ParseResponseToFrame(line, pid);
                    if (frame != null)
                        frames.Add(frame);
                }
            }
        }

        return frames;
    }

    private OBD2Frame? ParseCANMessage(byte[] message, IEnumerable<string> monitoredPids)
    {
        // CAN ISO 15765-4 format: [Length][Mode][PID][Data...]
        if (message.Length < 3)
            return null;

        var mode = message[1];
        if (mode != 0x41) // Not a PID response
            return null;

        var pid = message[2].ToString("X2");
        if (!monitoredPids.Any(p => p.ToUpper() == pid))
            return null;

        var dataBytes = message.Skip(3).ToArray();
        var reading = DecodeCANPidData(pid, dataBytes);

        if (reading == null)
            return null;

        return new OBD2Frame
        {
            Timestamp = DateTime.UtcNow,
            TimestampTicks = Stopwatch.GetTimestamp(),
            PID = pid,
            RawData = message,
            Reading = reading,
            FrameNumber = _totalFrames
        };
    }

    private OBD2Reading? DecodeCANPidData(string pid, byte[] data)
    {
        // Decode common PIDs from raw CAN data
        return pid.ToUpper() switch
        {
            "0C" when data.Length >= 2 => new OBD2Reading
            {
                PID = pid,
                Name = "Engine RPM",
                Value = ((data[0] * 256) + data[1]) / 4.0,
                Unit = "RPM",
                Timestamp = DateTime.UtcNow
            },
            "0D" when data.Length >= 1 => new OBD2Reading
            {
                PID = pid,
                Name = "Vehicle Speed",
                Value = data[0],
                Unit = "km/h",
                Timestamp = DateTime.UtcNow
            },
            "05" when data.Length >= 1 => new OBD2Reading
            {
                PID = pid,
                Name = "Engine Coolant Temp",
                Value = data[0] - 40,
                Unit = "°C",
                Timestamp = DateTime.UtcNow
            },
            "04" when data.Length >= 1 => new OBD2Reading
            {
                PID = pid,
                Name = "Engine Load",
                Value = data[0] * 100.0 / 255.0,
                Unit = "%",
                Timestamp = DateTime.UtcNow
            },
            "11" when data.Length >= 1 => new OBD2Reading
            {
                PID = pid,
                Name = "Throttle Position",
                Value = data[0] * 100.0 / 255.0,
                Unit = "%",
                Timestamp = DateTime.UtcNow
            },
            "10" when data.Length >= 2 => new OBD2Reading
            {
                PID = pid,
                Name = "MAF Air Flow",
                Value = ((data[0] * 256) + data[1]) / 100.0,
                Unit = "g/s",
                Timestamp = DateTime.UtcNow
            },
            _ => null
        };
    }

    private void PublishStatistics()
    {
        var stats = new StreamStatistics
        {
            TotalFrames = _totalFrames,
            ErrorFrames = _errorFrames,
            ElapsedTime = _sessionStopwatch.Elapsed,
            AverageFrequencyHz = _sessionStopwatch.Elapsed.TotalSeconds > 0
                ? _totalFrames / _sessionStopwatch.Elapsed.TotalSeconds
                : 0,
            BufferUtilization = _frameBuffer.Count,
            CurrentMode = _currentMode
        };

        _statsSubject.OnNext(stats);
    }

    private static byte[] HexStringToBytes(string hex)
    {
        if (hex.Length % 2 != 0)
            hex = "0" + hex;

        var bytes = new byte[hex.Length / 2];
        for (int i = 0; i < bytes.Length; i++)
        {
            bytes[i] = Convert.ToByte(hex.Substring(i * 2, 2), 16);
        }
        return bytes;
    }

    public void Dispose()
    {
        _streamingCts?.Cancel();
        _streamingCts?.Dispose();
        _frameSubject.Dispose();
        _statsSubject.Dispose();
    }
}

/// <summary>
/// Configuration for high-frequency streaming
/// </summary>
public class HighFrequencyConfig
{
    /// <summary>
    /// Target sampling frequency in Hz (10-50Hz typical)
    /// </summary>
    public double TargetFrequencyHz { get; set; } = 20;

    /// <summary>
    /// PIDs to monitor
    /// </summary>
    public List<string> PIDs { get; set; } = new()
    {
        OBD2PIDs.EngineRPM,
        OBD2PIDs.EngineLoad,
        OBD2PIDs.ThrottlePosition,
        OBD2PIDs.VehicleSpeed
    };

    /// <summary>
    /// Streaming strategy to use
    /// </summary>
    public StreamingStrategy Strategy { get; set; } = StreamingStrategy.ContinuousPolling;

    /// <summary>
    /// Response timeout in milliseconds
    /// </summary>
    public int ResponseTimeoutMs { get; set; } = 50;

    /// <summary>
    /// Enable adaptive timing (ELM327)
    /// </summary>
    public bool AdaptiveTiming { get; set; } = true;

    /// <summary>
    /// Internal buffer size for frames
    /// </summary>
    public int BufferSize { get; set; } = 10000;

    /// <summary>
    /// Statistics publish interval in milliseconds
    /// </summary>
    public int StatsIntervalMs { get; set; } = 1000;

    /// <summary>
    /// Create configuration for WOT (Wide Open Throttle) logging
    /// </summary>
    public static HighFrequencyConfig WOTConfig => new()
    {
        TargetFrequencyHz = 50,
        PIDs = new List<string>
        {
            OBD2PIDs.EngineRPM,
            OBD2PIDs.EngineLoad,
            OBD2PIDs.ThrottlePosition,
            OBD2PIDs.MAFSensor,
            OBD2PIDs.TimingAdvance
        },
        Strategy = StreamingStrategy.CANPassiveMonitor,
        ResponseTimeoutMs = 20
    };

    /// <summary>
    /// Create configuration for fuel economy monitoring
    /// </summary>
    public static HighFrequencyConfig EconomyConfig => new()
    {
        TargetFrequencyHz = 10,
        PIDs = new List<string>
        {
            OBD2PIDs.EngineRPM,
            OBD2PIDs.VehicleSpeed,
            OBD2PIDs.MAFSensor,
            OBD2PIDs.EngineLoad
        },
        Strategy = StreamingStrategy.ContinuousPolling,
        ResponseTimeoutMs = 100
    };
}

/// <summary>
/// Streaming strategy
/// </summary>
public enum StreamingStrategy
{
    /// <summary>
    /// Poll PIDs in round-robin fashion
    /// </summary>
    ContinuousPolling,

    /// <summary>
    /// Request multiple PIDs in single message (ELM327)
    /// </summary>
    BurstMode,

    /// <summary>
    /// Passive CAN bus monitoring (J2534 only)
    /// </summary>
    CANPassiveMonitor
}

/// <summary>
/// Streaming mode state
/// </summary>
public enum StreamingMode
{
    Stopped,
    Starting,
    Running,
    Paused,
    Stopping
}

/// <summary>
/// Single OBD2 data frame with high-precision timing
/// </summary>
public class OBD2Frame
{
    public DateTime Timestamp { get; init; }
    public long TimestampTicks { get; init; }
    public string PID { get; init; } = string.Empty;
    public byte[] RawData { get; init; } = Array.Empty<byte>();
    public OBD2Reading Reading { get; init; } = null!;
    public long FrameNumber { get; init; }
}

/// <summary>
/// Streaming statistics
/// </summary>
public class StreamStatistics
{
    public long TotalFrames { get; init; }
    public long ErrorFrames { get; init; }
    public TimeSpan ElapsedTime { get; init; }
    public double AverageFrequencyHz { get; init; }
    public int BufferUtilization { get; init; }
    public StreamingMode CurrentMode { get; init; }
    public double ErrorRate => TotalFrames > 0 ? (double)ErrorFrames / TotalFrames * 100 : 0;
}
