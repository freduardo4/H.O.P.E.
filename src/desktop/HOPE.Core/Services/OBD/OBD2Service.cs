using System.IO.Ports;
using System.Reactive.Linq;
using HOPE.Core.Models;

namespace HOPE.Core.Services.OBD;

public class OBD2Service : IOBD2Service, IDisposable
{
    private SerialPort? _serialPort;
    private bool _isConnected;
    private readonly SemaphoreSlim _semaphore = new(1, 1);

    public bool IsConnected => _isConnected;

    public string AdapterType => "ELM327";

    public string? DetectedProtocol { get; private set; }

    public event EventHandler<bool>? ConnectionStatusChanged;
    public event EventHandler<OBD2Reading>? DataReceived;
    public event EventHandler<OBD2ErrorEventArgs>? ErrorOccurred;

    public async Task<bool> ConnectAsync(string portName, int baudRate = 9600, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            if (_isConnected) return true;

            _serialPort = new SerialPort(portName, baudRate, Parity.None, 8, StopBits.One)
            {
                ReadTimeout = 2000,
                WriteTimeout = 2000,
                NewLine = "\r"
            };

            _serialPort.Open();
            _isConnected = true;
            ConnectionStatusChanged?.Invoke(this, true);

            // Initialize ELM327
            await SendCommandInternalAsync("ATZ", cancellationToken); // Reset
            await SendCommandInternalAsync("ATE0", cancellationToken); // Echo Off
            await SendCommandInternalAsync("ATL0", cancellationToken); // Linefeeds Off
            await SendCommandInternalAsync("ATH0", cancellationToken); // Headers Off
            await SendCommandInternalAsync("ATSP0", cancellationToken); // Auto Protocol

            // Get Protocol
            var protocolResponse = await SendCommandInternalAsync("ATDP", cancellationToken);
            DetectedProtocol = protocolResponse;

            return true;
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to connect", OBD2ErrorType.ConnectionLost, ex));
            _isConnected = false;
            return false;
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public async Task DisconnectAsync()
    {
        await _semaphore.WaitAsync();
        try
        {
            if (_serialPort?.IsOpen == true)
            {
                _serialPort.Close();
            }
            _isConnected = false;
            ConnectionStatusChanged?.Invoke(this, false);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public string[] GetAvailablePorts()
    {
        return SerialPort.GetPortNames();
    }

    public async Task<string> SendCommandAsync(string command, CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            return await SendCommandInternalAsync(command, cancellationToken);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    private async Task<string> SendCommandInternalAsync(string command, CancellationToken cancellationToken)
    {
        if (_serialPort == null || !_serialPort.IsOpen)
            throw new InvalidOperationException("Serial port is not open");

        // Clear buffers
        _serialPort.DiscardInBuffer();
        _serialPort.DiscardOutBuffer();

        _serialPort.WriteLine(command);

        // Simple read loop (in production this would be more robust)
        // ELM327 ends responses with '>'
        string response = "";
        while (!response.Contains('>'))
        {
             // This is a naive implementation; proper ELM327 reading requires reading char by char 
             // and handling buffers better, but this suffices for the skeleton.
             try 
             {
                string line = _serialPort.ReadExisting();
                if (!string.IsNullOrEmpty(line))
                    response += line;
                
                await Task.Delay(10, cancellationToken);
             }
             catch (TimeoutException)
             {
                 break;
             }
        }

        return response.Replace(">", "").Trim();
    }

    public Task<List<string>> GetSupportedPIDsAsync(CancellationToken cancellationToken = default)
    {
        // Placeholder implementation - would normally query PID 00
        return Task.FromResult(new List<string>());
    }

    public async Task<OBD2Reading> ReadPIDAsync(string pid, CancellationToken cancellationToken = default)
    {
        // Add mode 01 prefix if not present
        string command = pid.StartsWith("01") ? pid : "01" + pid;
        
        string response = await SendCommandAsync(command, cancellationToken);
        
        // Parse response (Simplified)
        // Response format: 41 0C 0F A0
        
        var reading = new OBD2Reading
        {
            PID = pid,
            RawResponse = response,
            Timestamp = DateTime.UtcNow,
            Value = 0 // Parsing logic would go here
        };

        return reading;
    }

    public Task<List<OBD2Reading>> ReadPIDsAsync(IEnumerable<string> pids, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    public IObservable<OBD2Reading> StreamPIDs(IEnumerable<string> pids, int intervalMs = 200, CancellationToken cancellationToken = default)
    {
        return Observable.Interval(TimeSpan.FromMilliseconds(intervalMs))
            .SelectMany(async _ => 
            {
                var readings = new List<OBD2Reading>();
                foreach(var pid in pids)
                {
                    try
                    {
                        var reading = await ReadPIDAsync(pid, cancellationToken);
                        readings.Add(reading);
                    }
                    catch
                    {
                        // Ignore errors during stream for now
                    }
                }
                return readings;
            })
            .SelectMany(r => r)
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
        return Task.FromResult<string?>(null);
    }

    public Task<string?> GetECUInfoAsync(CancellationToken cancellationToken = default)
    {
        return Task.FromResult<string?>(null);
    }
    
    public void Dispose()
    {
        _serialPort?.Dispose();
        _semaphore.Dispose();
    }
}
