using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Reactive.Linq;
using HOPE.Core.Models;

namespace HOPE.Core.Services.OBD;

public class MockOBD2Service : IOBD2Service
{
    private bool _isConnected;
    private readonly Random _random = new();
    private IDisposable? _simulationTimer;

    public bool IsConnected => _isConnected;

    public string AdapterType => "Mock ELM327";

    public string? DetectedProtocol => "ISO 15765-4 CAN (11 bit ID, 500 kbaud)";

    public FocusMode CurrentFocusMode { get; private set; } = FocusMode.Standard;

    public event EventHandler<bool>? ConnectionStatusChanged;
    public event EventHandler<OBD2Reading>? DataReceived;
    public event EventHandler<OBD2ErrorEventArgs>? ErrorOccurred;

    public Task<bool> ConnectAsync(string portName, int baudRate = 9600, CancellationToken cancellationToken = default)
    {
        _isConnected = true;
        ConnectionStatusChanged?.Invoke(this, true);
        return Task.FromResult(true);
    }

    public Task DisconnectAsync()
    {
        _isConnected = false;
        _simulationTimer?.Dispose();
        ConnectionStatusChanged?.Invoke(this, false);
        return Task.CompletedTask;
    }

    public string[] GetAvailablePorts()
    {
        return new[] { "COM1", "COM2", "MOCK_PORT" };
    }

    public Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default)
    {
        return Task.FromResult("NO DATA");
    }

    public Task<List<string>> GetSupportedPIDsAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new List<string> { "0C", "0D", "04", "05" });
    }

    public Task<OBD2Reading> ReadPIDAsync(string pid, CancellationToken cancellationToken = default)
    {
        var reading = GenerateMockReading(pid);
        return Task.FromResult(reading);
    }

    public Task<List<OBD2Reading>> ReadPIDsAsync(IEnumerable<string> pids, CancellationToken cancellationToken = default)
    {
        var readings = pids.Select(GenerateMockReading).ToList();
        return Task.FromResult(readings);
    }

    public IObservable<OBD2Reading> StreamPIDs(IEnumerable<string> pids, int intervalMs = 200, CancellationToken cancellationToken = default)
    {
        // Adjust interval based on FocusMode
        int effectiveInterval = CurrentFocusMode switch
        {
            FocusMode.WOT => 20, // 50Hz
            FocusMode.Economy => 500, // 2Hz
            FocusMode.Diagnostic => 100, // 10Hz
            FocusMode.Panic => 10, // 100Hz (extreme monitoring)
            _ => intervalMs
        };

        return Observable.Interval(TimeSpan.FromMilliseconds(effectiveInterval))
            .SelectMany(_ => pids)
            .Select(GenerateMockReading)
            .Do(reading => DataReceived?.Invoke(this, reading));
    }

    public Task<List<DiagnosticTroubleCode>> ReadDTCsAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(new List<DiagnosticTroubleCode>());
    }

    public Task<bool> ClearDTCsAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult(true);
    }

    public Task<string?> GetVINAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult<string?>("1MOCKVIN123456789");
    }

    public Task<string?> GetECUInfoAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult<string?>("MOCK ECU VERSION 1.0");
    }

    public Task SetFocusModeAsync(FocusMode mode)
    {
        CurrentFocusMode = mode;
        return Task.CompletedTask;
    }

    private OBD2Reading GenerateMockReading(string pid)
    {
        double value = 0;
        string unit = "";
        string name = "";

        switch (pid)
        {
            case "0C": // RPM
                value = _random.Next(800, 7000);
                unit = "RPM";
                name = "Engine RPM";
                break;
            case "0D": // Speed
                value = _random.Next(0, 240);
                unit = "km/h";
                name = "Vehicle Speed";
                break;
            case "04": // Load
                value = _random.Next(10, 90);
                unit = "%";
                name = "Calculated Engine Load";
                break;
            case "05": // Coolant Temp
                value = _random.Next(80, 105);
                unit = "°C";
                name = "Engine Coolant Temperature";
                break;
            default:
                value = 0;
                unit = "N/A";
                name = "Unknown";
                break;
        }

        return new OBD2Reading
        {
            PID = pid,
            Value = value,
            Unit = unit,
            Name = name,
            Timestamp = DateTime.UtcNow,
            RawResponse = value.ToString()
        };
    }
}
