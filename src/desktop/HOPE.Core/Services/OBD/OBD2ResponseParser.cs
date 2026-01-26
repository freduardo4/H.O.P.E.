using HOPE.Core.Data;
using HOPE.Core.Models;

namespace HOPE.Core.Services.OBD;

/// <summary>
/// Parser for OBD2 responses from ELM327 adapters.
/// Handles Mode 01 (live data) and Mode 03 (DTCs) responses.
/// </summary>
public static class OBD2ResponseParser
{
    /// <summary>
    /// PID information including name, unit, and decoding formula.
    /// </summary>
    private static readonly Dictionary<string, PIDInfo> PIDRegistry = new()
    {
        // Engine Parameters
        [OBD2PIDs.EngineLoad] = new PIDInfo("Engine Load", "%", 1, (a, _) => a * 100.0 / 255.0),
        [OBD2PIDs.CoolantTemp] = new PIDInfo("Coolant Temperature", "°C", 1, (a, _) => a - 40),
        [OBD2PIDs.ShortTermFuelTrim] = new PIDInfo("Short Term Fuel Trim", "%", 1, (a, _) => (a - 128) * 100.0 / 128.0),
        [OBD2PIDs.LongTermFuelTrim] = new PIDInfo("Long Term Fuel Trim", "%", 1, (a, _) => (a - 128) * 100.0 / 128.0),
        [OBD2PIDs.FuelPressure] = new PIDInfo("Fuel Pressure", "kPa", 1, (a, _) => a * 3),
        [OBD2PIDs.IntakeManifoldPressure] = new PIDInfo("Intake Manifold Pressure", "kPa", 1, (a, _) => a),
        [OBD2PIDs.EngineRPM] = new PIDInfo("Engine RPM", "RPM", 2, (a, b) => ((a * 256) + b) / 4.0),
        [OBD2PIDs.VehicleSpeed] = new PIDInfo("Vehicle Speed", "km/h", 1, (a, _) => a),
        [OBD2PIDs.TimingAdvance] = new PIDInfo("Timing Advance", "° before TDC", 1, (a, _) => (a / 2.0) - 64),
        [OBD2PIDs.IntakeAirTemp] = new PIDInfo("Intake Air Temperature", "°C", 1, (a, _) => a - 40),
        [OBD2PIDs.MAFSensor] = new PIDInfo("MAF Air Flow Rate", "g/s", 2, (a, b) => ((a * 256) + b) / 100.0),
        [OBD2PIDs.ThrottlePosition] = new PIDInfo("Throttle Position", "%", 1, (a, _) => a * 100.0 / 255.0),
        [OBD2PIDs.O2Sensor1Voltage] = new PIDInfo("O2 Sensor 1 Voltage", "V", 2, (a, _) => a / 200.0),
        [OBD2PIDs.O2Sensor2Voltage] = new PIDInfo("O2 Sensor 2 Voltage", "V", 2, (a, _) => a / 200.0),
        [OBD2PIDs.EngineRuntime] = new PIDInfo("Engine Runtime", "seconds", 2, (a, b) => (a * 256) + b),
        [OBD2PIDs.DistanceSinceDTCCleared] = new PIDInfo("Distance Since DTC Cleared", "km", 2, (a, b) => (a * 256) + b),
        [OBD2PIDs.BarometricPressure] = new PIDInfo("Barometric Pressure", "kPa", 1, (a, _) => a),
    };

    /// <summary>
    /// Parse a Mode 01 (live data) response.
    /// </summary>
    /// <param name="response">Raw response from ELM327 (e.g., "41 0C 0F A0")</param>
    /// <param name="requestedPID">The PID that was requested</param>
    /// <returns>Parsed OBD2Reading or null if parsing failed</returns>
    public static OBD2Reading? ParseMode01Response(string response, string requestedPID)
    {
        if (string.IsNullOrWhiteSpace(response))
            return null;

        // Clean up response
        response = CleanResponse(response);

        // Check for error responses
        if (IsErrorResponse(response))
            return null;

        // Split into bytes
        var bytes = response.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Mode 01 response should start with "41" followed by the PID
        if (bytes.Length < 3)
            return null;

        if (bytes[0] != "41")
            return null;

        string responsePID = bytes[1];
        if (!responsePID.Equals(requestedPID, StringComparison.OrdinalIgnoreCase))
            return null;

        // Get PID info
        if (!PIDRegistry.TryGetValue(requestedPID.ToUpper(), out var pidInfo))
        {
            // Unknown PID - return raw value
            return new OBD2Reading
            {
                PID = requestedPID,
                Name = $"Unknown PID {requestedPID}",
                Value = bytes.Length > 2 ? Convert.ToInt32(bytes[2], 16) : 0,
                Unit = "raw",
                RawResponse = response,
                Timestamp = DateTime.UtcNow
            };
        }

        // Parse data bytes
        try
        {
            int dataStartIndex = 2;
            byte a = bytes.Length > dataStartIndex ? Convert.ToByte(bytes[dataStartIndex], 16) : (byte)0;
            byte b = bytes.Length > dataStartIndex + 1 ? Convert.ToByte(bytes[dataStartIndex + 1], 16) : (byte)0;

            double value = pidInfo.Formula(a, b);

            return new OBD2Reading
            {
                PID = requestedPID,
                Name = pidInfo.Name,
                Value = value,
                Unit = pidInfo.Unit,
                RawResponse = response,
                Timestamp = DateTime.UtcNow
            };
        }
        catch
        {
            return null;
        }
    }

    /// <summary>
    /// Parse supported PIDs response (PID 00, 20, 40, etc.)
    /// </summary>
    /// <param name="response">Response to PID 00, 20, 40, etc.</param>
    /// <param name="basePID">The base PID that was queried (00, 20, 40, etc.)</param>
    /// <returns>List of supported PIDs</returns>
    public static List<string> ParseSupportedPIDs(string response, string basePID)
    {
        var supportedPIDs = new List<string>();

        response = CleanResponse(response);
        var bytes = response.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        if (bytes.Length < 6 || bytes[0] != "41")
            return supportedPIDs;

        // Parse 4 data bytes (32 bits) indicating supported PIDs
        int baseValue = Convert.ToInt32(basePID, 16);

        for (int byteIndex = 0; byteIndex < 4; byteIndex++)
        {
            if (bytes.Length <= 2 + byteIndex)
                break;

            byte dataByte = Convert.ToByte(bytes[2 + byteIndex], 16);

            for (int bit = 7; bit >= 0; bit--)
            {
                if ((dataByte & (1 << bit)) != 0)
                {
                    int pidNumber = baseValue + (byteIndex * 8) + (7 - bit) + 1;
                    supportedPIDs.Add(pidNumber.ToString("X2"));
                }
            }
        }

        return supportedPIDs;
    }

    /// <summary>
    /// Parse Mode 03 (Read DTCs) response.
    /// </summary>
    /// <param name="response">Raw response from Mode 03 request</param>
    /// <returns>List of diagnostic trouble codes</returns>
    public static List<DiagnosticTroubleCode> ParseDTCs(string response)
    {
        var dtcs = new List<DiagnosticTroubleCode>();

        response = CleanResponse(response);

        if (string.IsNullOrEmpty(response) || IsErrorResponse(response))
            return dtcs;

        var bytes = response.Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Mode 03 response starts with "43"
        if (bytes.Length < 1 || bytes[0] != "43")
            return dtcs;

        // Each DTC is 2 bytes
        for (int i = 1; i + 1 < bytes.Length; i += 2)
        {
            try
            {
                byte high = Convert.ToByte(bytes[i], 16);
                byte low = Convert.ToByte(bytes[i + 1], 16);

                // Skip empty DTCs (00 00)
                if (high == 0 && low == 0)
                    continue;

                string dtcCode = DecodeDTC(high, low);
                dtcs.Add(new DiagnosticTroubleCode
                {
                    Code = dtcCode,
                    Description = GetDTCDescription(dtcCode),
                    Severity = GetDTCSeverity(dtcCode),
                    IsPending = false,
                    DetectedAt = DateTime.UtcNow
                });
            }
            catch
            {
                // Skip malformed DTCs
            }
        }

        return dtcs;
    }

    /// <summary>
    /// Parse Mode 09 PID 02 (VIN) response.
    /// </summary>
    /// <param name="response">Raw response from Mode 09 PID 02 request</param>
    /// <returns>VIN string or null if parsing failed</returns>
    public static string? ParseVIN(string response)
    {
        response = CleanResponse(response);

        if (string.IsNullOrEmpty(response) || IsErrorResponse(response))
            return null;

        // VIN response can be multi-line for ISO 15765 (CAN)
        // Format varies by protocol, but data bytes should be ASCII characters

        var allBytes = new List<byte>();
        var lines = response.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

        foreach (var line in lines)
        {
            var bytes = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);

            // Skip header bytes (49 02 xx) and collect data
            int startIndex = bytes.Length > 3 && bytes[0] == "49" && bytes[1] == "02" ? 3 : 0;

            for (int i = startIndex; i < bytes.Length; i++)
            {
                try
                {
                    byte b = Convert.ToByte(bytes[i], 16);
                    if (b >= 0x20 && b <= 0x7E) // Printable ASCII
                        allBytes.Add(b);
                }
                catch { }
            }
        }

        if (allBytes.Count >= 17)
        {
            return System.Text.Encoding.ASCII.GetString(allBytes.Take(17).ToArray());
        }

        return null;
    }

    /// <summary>
    /// Decode DTC bytes to standard format (e.g., P0300).
    /// </summary>
    private static string DecodeDTC(byte high, byte low)
    {
        // First 2 bits of high byte determine the type
        char[] typeChars = { 'P', 'C', 'B', 'U' };
        char type = typeChars[(high >> 6) & 0x03];

        // Second digit
        int secondDigit = (high >> 4) & 0x03;

        // Third digit
        int thirdDigit = high & 0x0F;

        // Fourth and fifth digits from low byte
        int fourthDigit = (low >> 4) & 0x0F;
        int fifthDigit = low & 0x0F;

        return $"{type}{secondDigit}{thirdDigit:X}{fourthDigit:X}{fifthDigit:X}";
    }

    /// <summary>
    /// Clean up raw ELM327 response.
    /// </summary>
    private static string CleanResponse(string response)
    {
        if (string.IsNullOrEmpty(response))
            return string.Empty;

        // Remove common ELM327 artifacts
        response = response
            .Replace(">", "")
            .Replace("\r", " ")
            .Replace("\n", " ")
            .Replace("SEARCHING...", "")
            .Replace("BUS INIT...", "")
            .Trim();

        // Normalize spaces
        while (response.Contains("  "))
            response = response.Replace("  ", " ");

        return response.ToUpper();
    }

    /// <summary>
    /// Check if response is an error.
    /// </summary>
    private static bool IsErrorResponse(string response)
    {
        string[] errorIndicators = { "NO DATA", "UNABLE TO CONNECT", "ERROR", "?", "CAN ERROR", "BUS ERROR" };
        return errorIndicators.Any(e => response.Contains(e, StringComparison.OrdinalIgnoreCase));
    }

    /// <summary>
    /// Get description for DTCs from the comprehensive database.
    /// </summary>
    private static string GetDTCDescription(string code)
    {
        return DTCDatabase.GetDescription(code);
    }

    /// <summary>
    /// Get severity for DTC from the comprehensive database.
    /// </summary>
    private static DTCSeverity GetDTCSeverity(string code)
    {
        var info = DTCDatabase.GetInfo(code);
        if (info != null)
        {
            return info.Severity switch
            {
                Data.DTCSeverity.Critical => DTCSeverity.Critical,
                Data.DTCSeverity.High => DTCSeverity.Critical,
                Data.DTCSeverity.Medium => DTCSeverity.Warning,
                Data.DTCSeverity.Low => DTCSeverity.Info,
                _ => DTCSeverity.Warning
            };
        }

        // Fallback for codes not in database
        if (code.StartsWith("P3") || code.StartsWith("U0"))
            return DTCSeverity.Critical;
        if (code.StartsWith("P0") || code.StartsWith("P1") || code.StartsWith("P2"))
            return DTCSeverity.Warning;

        return DTCSeverity.Info;
    }

    /// <summary>
    /// Information about a PID including decoding formula.
    /// </summary>
    private record PIDInfo(string Name, string Unit, int ByteCount, Func<byte, byte, double> Formula);
}
