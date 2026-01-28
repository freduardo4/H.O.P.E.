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
    
    public FocusMode CurrentFocusMode { get; private set; } = FocusMode.Standard;

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

    public async Task<List<string>> GetSupportedPIDsAsync(CancellationToken cancellationToken = default)
    {
        var supportedPIDs = new List<string>();

        try
        {
            // Query supported PIDs in ranges (00, 20, 40, 60, 80, A0, C0)
            string[] pidRanges = { "00", "20", "40", "60", "80", "A0", "C0" };

            foreach (var basePID in pidRanges)
            {
                string response = await SendCommandAsync($"01{basePID}", cancellationToken);
                var pidsInRange = OBD2ResponseParser.ParseSupportedPIDs(response, basePID);
                supportedPIDs.AddRange(pidsInRange);

                // If the last PID in the range (e.g., 20, 40, etc.) is not supported,
                // no need to query further ranges
                string nextRangePID = (Convert.ToInt32(basePID, 16) + 0x20).ToString("X2");
                if (!pidsInRange.Contains(nextRangePID))
                    break;
            }
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to get supported PIDs", OBD2ErrorType.CommunicationError, ex));
        }

        return supportedPIDs;
    }

    public async Task<OBD2Reading> ReadPIDAsync(string pid, CancellationToken cancellationToken = default)
    {
        // Normalize PID format (remove any prefix)
        pid = pid.Replace("01", "").Replace("0x", "").Replace("0X", "").Trim().ToUpper();

        // Add mode 01 prefix
        string command = "01" + pid;

        string response = await SendCommandAsync(command, cancellationToken);

        // Parse response using the parser
        var reading = OBD2ResponseParser.ParseMode01Response(response, pid);

        if (reading == null)
        {
            // Return a failed reading
            return new OBD2Reading
            {
                PID = pid,
                Name = "Unknown",
                RawResponse = response,
                Timestamp = DateTime.UtcNow,
                Value = 0,
                Unit = "N/A"
            };
        }

        // Raise data received event
        DataReceived?.Invoke(this, reading);

        return reading;
    }

    public async Task<List<OBD2Reading>> ReadPIDsAsync(IEnumerable<string> pids, CancellationToken cancellationToken = default)
    {
        var readings = new List<OBD2Reading>();

        foreach (var pid in pids)
        {
            if (cancellationToken.IsCancellationRequested)
                break;

            try
            {
                var reading = await ReadPIDAsync(pid, cancellationToken);
                readings.Add(reading);
            }
            catch (Exception ex)
            {
                ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs($"Failed to read PID {pid}", OBD2ErrorType.CommunicationError, ex));
            }
        }

        return readings;
    }

    public IObservable<OBD2Reading> StreamPIDs(IEnumerable<string> pids, int intervalMs = 200, CancellationToken cancellationToken = default)
    {
        // Adjust interval based on FocusMode if not explicitly overridden
        int effectiveInterval = intervalMs;
        if (intervalMs == 200) // Default value was used
        {
            effectiveInterval = CurrentFocusMode switch
            {
                FocusMode.WOT => 20,     // 50Hz
                FocusMode.Economy => 500, // 2Hz
                FocusMode.Diagnostic => 100, // 10Hz
                _ => 200
            };
        }

        return Observable.Interval(TimeSpan.FromMilliseconds(effectiveInterval))
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

    public async Task<List<DiagnosticTroubleCode>> ReadDTCsAsync(CancellationToken cancellationToken = default)
    {
        var allDTCs = new List<DiagnosticTroubleCode>();

        try
        {
            // Mode 03 - Request stored DTCs
            string storedResponse = await SendCommandAsync("03", cancellationToken);
            var storedDTCs = OBD2ResponseParser.ParseDTCs(storedResponse);
            foreach (var dtc in storedDTCs)
            {
                dtc.IsPending = false;
            }
            allDTCs.AddRange(storedDTCs);

            // Mode 07 - Request pending DTCs
            string pendingResponse = await SendCommandAsync("07", cancellationToken);
            var pendingDTCs = OBD2ResponseParser.ParseDTCs(pendingResponse);
            foreach (var dtc in pendingDTCs)
            {
                dtc.IsPending = true;
            }
            allDTCs.AddRange(pendingDTCs);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to read DTCs", OBD2ErrorType.CommunicationError, ex));
        }

        return allDTCs;
    }

    public async Task<bool> ClearDTCsAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Mode 04 - Clear DTCs and freeze frame data
            string response = await SendCommandAsync("04", cancellationToken);

            // Check for positive response (44)
            return response.Contains("44") || !response.Contains("ERROR");
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to clear DTCs", OBD2ErrorType.CommunicationError, ex));
            return false;
        }
    }

    public async Task<string?> GetVINAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Mode 09 PID 02 - Request VIN
            string response = await SendCommandAsync("0902", cancellationToken);
            return OBD2ResponseParser.ParseVIN(response);
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to get VIN", OBD2ErrorType.CommunicationError, ex));
            return null;
        }
    }

    public async Task<string?> GetECUInfoAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            // Mode 09 PID 0A - Request ECU Name
            string response = await SendCommandAsync("090A", cancellationToken);

            // Clean and return the ECU name
            response = response.Replace("49 0A", "").Trim();
            if (string.IsNullOrWhiteSpace(response) || response.Contains("NO DATA"))
                return null;

            // Convert hex to ASCII
            var bytes = response.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var asciiChars = new List<char>();

            foreach (var byteStr in bytes)
            {
                if (byte.TryParse(byteStr, System.Globalization.NumberStyles.HexNumber, null, out byte b))
                {
                    if (b >= 0x20 && b <= 0x7E) // Printable ASCII
                        asciiChars.Add((char)b);
                }
            }

            return asciiChars.Count > 0 ? new string(asciiChars.ToArray()).Trim() : null;
        }
        catch (Exception ex)
        {
            ErrorOccurred?.Invoke(this, new OBD2ErrorEventArgs("Failed to get ECU info", OBD2ErrorType.CommunicationError, ex));
            return null;
        }
    }

    public Task SetFocusModeAsync(FocusMode mode)
    {
        CurrentFocusMode = mode;
        // In a real implementation, we might reconfigure the adapter here
        // or trigger high-frequency mode for J2534
        return Task.CompletedTask;
    }
    
    public void Dispose()
    {
        _serialPort?.Dispose();
        _semaphore.Dispose();
    }
}
