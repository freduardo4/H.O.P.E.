using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Models;
using HOPE.Core.Protocols;
using HOPE.Core.Services.OBD;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.BiDirectional;

/// <summary>
/// Service for bi-directional vehicle control via UDS Service 0x2F (InputOutputControlByIdentifier).
/// Implements comprehensive safety interlocks to prevent vehicle damage.
/// </summary>
public class BiDirectionalService : IBiDirectionalService
{
    private readonly IHardwareAdapter _adapter;
    private readonly VoltageMonitor? _voltageMonitor;
    private readonly ILogger<BiDirectionalService>? _logger;
    private readonly SemaphoreSlim _semaphore = new(1, 1);
    private bool _securityAccessGranted;
    private DateTime? _lastSecurityAccessTime;
    private const int SECURITY_ACCESS_TIMEOUT_MS = 300000; // 5 minutes

    /// <summary>
    /// Event raised when an actuator control operation completes
    /// </summary>
    public event EventHandler<ActuatorControlEventArgs>? ActuatorControlCompleted;

    /// <summary>
    /// Event raised when a safety violation is detected
    /// </summary>
    public event EventHandler<SafetyViolationEventArgs>? SafetyViolationDetected;

    public BiDirectionalService(
        IHardwareAdapter adapter,
        VoltageMonitor? voltageMonitor = null,
        ILogger<BiDirectionalService>? logger = null)
    {
        _adapter = adapter ?? throw new ArgumentNullException(nameof(adapter));
        _voltageMonitor = voltageMonitor;
        _logger = logger;

        if (!adapter.SupportsBiDirectionalControl)
        {
            throw new NotSupportedException(
                $"Adapter {adapter.AdapterName} does not support bi-directional control. Use a J2534 adapter.");
        }
    }

    /// <summary>
    /// Control an actuator with safety validation
    /// </summary>
    /// <param name="command">Actuator command to execute</param>
    /// <param name="currentState">Current vehicle state for safety validation</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Response from actuator control</returns>
    public async Task<ActuatorResponse> ControlActuatorAsync(
        ActuatorCommand command,
        VehicleState currentState,
        CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            _logger?.LogInformation(
                "Actuator control requested: {ActuatorType} - {Action}",
                command.ActuatorType,
                command.Action);

            // Step 1: Validate command parameters
            var commandValidation = command.Validate();
            if (!commandValidation.IsValid)
            {
                var error = string.Join("; ", commandValidation.Errors);
                _logger?.LogWarning("Command validation failed: {Errors}", error);
                return ActuatorResponse.Failed($"Invalid command: {error}");
            }

            // Step 2: Validate safety preconditions
            var safetyResult = ValidateSafetyConditions(command, currentState);
            if (!safetyResult.IsSafe)
            {
                var violations = string.Join("; ", safetyResult.Violations.Select(v => v.ToString()));
                _logger?.LogWarning("Safety validation failed: {Violations}", violations);

                SafetyViolationDetected?.Invoke(this, new SafetyViolationEventArgs(command, safetyResult));

                return ActuatorResponse.Failed($"Safety conditions not met: {violations}");
            }

            // Step 3: Validate adapter connection
            if (!_adapter.IsConnected)
            {
                _logger?.LogError("Adapter not connected");
                return ActuatorResponse.Failed("Hardware adapter not connected");
            }

            // Step 4: Validate battery voltage (if voltage monitor available)
            if (_voltageMonitor != null)
            {
                var voltageResult = await _voltageMonitor.ValidateForOperationAsync(
                    OperationType.BiDirectionalControl,
                    cancellationToken);

                if (!voltageResult.IsValid)
                {
                    _logger?.LogWarning(
                        "Voltage validation failed: {Voltage}V < {Required}V",
                        voltageResult.CurrentVoltage,
                        voltageResult.RequiredVoltage);

                    return ActuatorResponse.Failed(voltageResult.Message);
                }
            }

            // Step 5: Ensure security access if required
            if (command.RequiredPreconditions.RequireSecurityAccess && !IsSecurityAccessValid())
            {
                var securityResult = await RequestSecurityAccessAsync(cancellationToken);
                if (!securityResult)
                {
                    _logger?.LogWarning("Security access denied");
                    return ActuatorResponse.Failed(
                        "Security access denied",
                        UDSNegativeResponseCode.SecurityAccessDenied);
                }
            }

            // Step 6: Execute the actuator control
            return await ExecuteActuatorControlAsync(command, cancellationToken);
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Validate safety conditions for an actuator command
    /// </summary>
    public SafetyValidationResult ValidateSafetyConditions(ActuatorCommand command, VehicleState state)
    {
        var result = new SafetyValidationResult { IsSafe = true };
        var preconditions = command.RequiredPreconditions;

        // Check engine state
        if (preconditions.RequireEngineOff && state.IsEngineRunning)
        {
            result.AddViolation("Engine", $"Running ({state.EngineRPM} RPM)", "Off");
            result.IsSafe = false;
        }

        // Check vehicle speed
        if (preconditions.RequireVehicleStationary && !state.IsStationary)
        {
            result.AddViolation("Vehicle Speed", $"{state.VehicleSpeedKmh:F1} km/h", "0 km/h");
            result.IsSafe = false;
        }

        // Check gear position
        if (preconditions.RequireParkOrNeutral && !state.IsInParkOrNeutral)
        {
            result.AddViolation("Gear Position", state.GearPosition.ToString(), "Park or Neutral");
            result.IsSafe = false;
        }

        // Check parking brake
        if (preconditions.RequireParkingBrake && !state.ParkingBrakeEngaged)
        {
            result.AddViolation("Parking Brake", "Not Engaged", "Engaged");
            result.IsSafe = false;
        }

        // Check battery voltage
        if (state.BatteryVoltage < preconditions.MinBatteryVoltage)
        {
            result.AddViolation(
                "Battery Voltage",
                $"{state.BatteryVoltage:F2}V",
                $">= {preconditions.MinBatteryVoltage:F1}V");
            result.IsSafe = false;
        }

        // Check adapter type
        if (preconditions.RequireJ2534 && _adapter.Type != HardwareType.J2534)
        {
            result.AddViolation("Adapter Type", _adapter.Type.ToString(), "J2534");
            result.IsSafe = false;
        }

        // Check ignition state
        if (preconditions.RequireIgnitionOn && state.IgnitionState != IgnitionState.On)
        {
            result.AddViolation("Ignition", state.IgnitionState.ToString(), "On");
            result.IsSafe = false;
        }

        return result;
    }

    /// <summary>
    /// Return control of all actuators to the ECU
    /// </summary>
    public async Task<ActuatorResponse> ReturnControlToECUAsync(
        ushort dataIdentifier,
        CancellationToken cancellationToken = default)
    {
        await _semaphore.WaitAsync(cancellationToken);
        try
        {
            _logger?.LogInformation("Returning control to ECU for DID 0x{DID:X4}", dataIdentifier);

            // UDS Service 0x2F with sub-function 0x00 (ReturnControlToECU)
            byte[] request = new byte[]
            {
                0x2F, // Service ID: InputOutputControlByIdentifier
                (byte)(dataIdentifier >> 8),
                (byte)(dataIdentifier & 0xFF),
                0x00  // Sub-function: ReturnControlToECU
            };

            var response = await _adapter.SendMessageAsync(request, 2000, cancellationToken);

            if (response.Length > 0 && response[0] == 0x6F)
            {
                _logger?.LogInformation("Control returned to ECU successfully");
                return ActuatorResponse.Successful(0, "Control returned to ECU");
            }

            if (response.Length >= 3 && response[0] == 0x7F)
            {
                var nrc = (UDSNegativeResponseCode)response[2];
                _logger?.LogWarning("Return control failed: NRC 0x{NRC:X2}", response[2]);
                return ActuatorResponse.Failed($"Failed to return control: {nrc}", nrc);
            }

            return ActuatorResponse.Failed("Unknown response from ECU");
        }
        finally
        {
            _semaphore.Release();
        }
    }

    /// <summary>
    /// Get available actuators for a vehicle (based on supported DIDs)
    /// </summary>
    public async Task<List<AvailableActuator>> GetAvailableActuatorsAsync(
        CancellationToken cancellationToken = default)
    {
        var actuators = new List<AvailableActuator>();

        // Common actuator DIDs - would be expanded based on vehicle-specific definitions
        var commonActuators = new Dictionary<ushort, (ActuatorType Type, string Name)>
        {
            { 0xF100, (ActuatorType.FuelPump, "Fuel Pump Relay") },
            { 0xF101, (ActuatorType.CoolingFan, "Cooling Fan Relay") },
            { 0xF102, (ActuatorType.ACCompressor, "A/C Compressor Clutch") },
            { 0xF110, (ActuatorType.Injector, "Fuel Injector #1") },
            { 0xF111, (ActuatorType.Injector, "Fuel Injector #2") },
            { 0xF112, (ActuatorType.Injector, "Fuel Injector #3") },
            { 0xF113, (ActuatorType.Injector, "Fuel Injector #4") },
            { 0xF120, (ActuatorType.IgnitionCoil, "Ignition Coil #1") },
            { 0xF121, (ActuatorType.IgnitionCoil, "Ignition Coil #2") },
            { 0xF122, (ActuatorType.IgnitionCoil, "Ignition Coil #3") },
            { 0xF123, (ActuatorType.IgnitionCoil, "Ignition Coil #4") },
            { 0xF130, (ActuatorType.PurgeSolenoid, "EVAP Purge Solenoid") },
            { 0xF131, (ActuatorType.EGRValve, "EGR Valve") },
            { 0xF132, (ActuatorType.IdleAirControl, "Idle Air Control Valve") },
            { 0xF133, (ActuatorType.VVTSolenoid, "Variable Valve Timing Solenoid") },
            { 0xF140, (ActuatorType.ThrottleBody, "Electronic Throttle Body") },
            { 0xF150, (ActuatorType.WastegateSolenoid, "Turbo Wastegate Solenoid") },
        };

        foreach (var (did, info) in commonActuators)
        {
            // Try to read DID to see if it's supported
            try
            {
                byte[] request = { 0x22, (byte)(did >> 8), (byte)(did & 0xFF) };
                var response = await _adapter.SendMessageAsync(request, 500, cancellationToken);

                if (response.Length > 0 && response[0] == 0x62)
                {
                    actuators.Add(new AvailableActuator
                    {
                        DataIdentifier = did,
                        ActuatorType = info.Type,
                        Name = info.Name,
                        IsSupported = true,
                        RequiredPreconditions = GetDefaultPreconditions(info.Type)
                    });
                }
            }
            catch
            {
                // DID not supported, skip
            }
        }

        return actuators;
    }

    private async Task<ActuatorResponse> ExecuteActuatorControlAsync(
        ActuatorCommand command,
        CancellationToken cancellationToken)
    {
        _logger?.LogInformation(
            "Executing actuator control: DID=0x{DID:X4}, Action={Action}, Duration={Duration}ms",
            command.DataIdentifier,
            command.Action,
            command.DurationMs);

        // Build UDS request: Service 0x2F + DID + SubFunction + ControlValue (if applicable)
        var requestBytes = new List<byte>
        {
            0x2F, // Service ID: InputOutputControlByIdentifier
            (byte)(command.DataIdentifier >> 8),
            (byte)(command.DataIdentifier & 0xFF),
            (byte)command.Action
        };

        if (command.ControlValue.HasValue)
        {
            requestBytes.Add(command.ControlValue.Value);
        }

        var startTime = DateTime.UtcNow;

        try
        {
            // Send activation request
            var response = await _adapter.SendMessageAsync(
                requestBytes.ToArray(),
                2000,
                cancellationToken);

            if (response.Length == 0)
            {
                return ActuatorResponse.Failed("No response from ECU");
            }

            // Check for positive response (0x6F)
            if (response[0] == 0x6F)
            {
                _logger?.LogInformation("Actuator activated successfully");

                // Wait for specified duration (with early cancellation support)
                var effectiveDuration = Math.Min(command.DurationMs, command.MaxDurationMs);

                try
                {
                    await Task.Delay(effectiveDuration, cancellationToken);
                }
                catch (OperationCanceledException)
                {
                    _logger?.LogWarning("Actuator control cancelled - returning control to ECU");
                }

                // Return control to ECU
                await ReturnControlToECUAsync(command.DataIdentifier, CancellationToken.None);

                var actualDuration = (int)(DateTime.UtcNow - startTime).TotalMilliseconds;

                var result = ActuatorResponse.Successful(
                    actualDuration,
                    $"{command.ActuatorType} activated for {actualDuration}ms");

                ActuatorControlCompleted?.Invoke(this, new ActuatorControlEventArgs(command, result));

                return result;
            }

            // Check for negative response (0x7F)
            if (response.Length >= 3 && response[0] == 0x7F)
            {
                var nrc = (UDSNegativeResponseCode)response[2];
                var errorMessage = GetNegativeResponseMessage(nrc);

                _logger?.LogWarning(
                    "Actuator control failed: NRC=0x{NRC:X2} ({Message})",
                    response[2],
                    errorMessage);

                var failedResult = ActuatorResponse.Failed(errorMessage, nrc);
                ActuatorControlCompleted?.Invoke(this, new ActuatorControlEventArgs(command, failedResult));

                return failedResult;
            }

            return ActuatorResponse.Failed($"Unexpected response: 0x{response[0]:X2}");
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Actuator control failed with exception");

            // Attempt to return control to ECU even on error
            try
            {
                await ReturnControlToECUAsync(command.DataIdentifier, CancellationToken.None);
            }
            catch
            {
                // Ignore cleanup errors
            }

            return ActuatorResponse.Failed($"Exception: {ex.Message}");
        }
    }

    private async Task<bool> RequestSecurityAccessAsync(CancellationToken cancellationToken)
    {
        _logger?.LogInformation("Requesting security access");

        // UDS Service 0x27: SecurityAccess
        // Sub-function 0x01: Request Seed
        byte[] seedRequest = { 0x27, 0x01 };
        var seedResponse = await _adapter.SendMessageAsync(seedRequest, 2000, cancellationToken);

        if (seedResponse.Length < 3 || seedResponse[0] != 0x67)
        {
            _logger?.LogWarning("Failed to get security seed");
            return false;
        }

        // Extract seed (skip service ID and sub-function)
        var seed = seedResponse.Skip(2).ToArray();

        if (seed.All(b => b == 0))
        {
            // Zero seed means security already unlocked
            _securityAccessGranted = true;
            _lastSecurityAccessTime = DateTime.UtcNow;
            return true;
        }

        // Calculate key from seed (this would be vehicle-specific)
        var key = CalculateSecurityKey(seed);

        // Sub-function 0x02: Send Key
        var keyRequest = new byte[2 + key.Length];
        keyRequest[0] = 0x27;
        keyRequest[1] = 0x02;
        Array.Copy(key, 0, keyRequest, 2, key.Length);

        var keyResponse = await _adapter.SendMessageAsync(keyRequest, 2000, cancellationToken);

        if (keyResponse.Length > 0 && keyResponse[0] == 0x67)
        {
            _securityAccessGranted = true;
            _lastSecurityAccessTime = DateTime.UtcNow;
            _logger?.LogInformation("Security access granted");
            return true;
        }

        _logger?.LogWarning("Security key rejected");
        return false;
    }

    private bool IsSecurityAccessValid()
    {
        if (!_securityAccessGranted || !_lastSecurityAccessTime.HasValue)
            return false;

        var elapsed = (DateTime.UtcNow - _lastSecurityAccessTime.Value).TotalMilliseconds;
        return elapsed < SECURITY_ACCESS_TIMEOUT_MS;
    }

    private static byte[] CalculateSecurityKey(byte[] seed)
    {
        // Placeholder implementation - actual algorithm is ECU-specific
        // Common algorithms include XOR with constant, bit rotation, or proprietary calculations
        var key = new byte[seed.Length];
        const uint SECRET = 0xCAFEBABE;

        for (int i = 0; i < seed.Length; i++)
        {
            key[i] = (byte)(seed[i] ^ ((SECRET >> (i * 8)) & 0xFF));
        }

        return key;
    }

    private static SafetyPreconditions GetDefaultPreconditions(ActuatorType type)
    {
        return type switch
        {
            ActuatorType.FuelPump or
            ActuatorType.Injector or
            ActuatorType.IgnitionCoil or
            ActuatorType.ThrottleBody or
            ActuatorType.WastegateSolenoid => SafetyPreconditions.Strict,

            ActuatorType.HeadlightRelay or
            ActuatorType.HornRelay or
            ActuatorType.InteriorLight => SafetyPreconditions.Relaxed,

            _ => SafetyPreconditions.Default
        };
    }

    private static string GetNegativeResponseMessage(UDSNegativeResponseCode nrc)
    {
        return nrc switch
        {
            UDSNegativeResponseCode.ConditionsNotCorrect => "Conditions not correct for this operation",
            UDSNegativeResponseCode.SecurityAccessDenied => "Security access required",
            UDSNegativeResponseCode.RequestOutOfRange => "Actuator not supported",
            UDSNegativeResponseCode.EngineIsRunning => "Engine must be off",
            UDSNegativeResponseCode.EngineIsNotRunning => "Engine must be running",
            UDSNegativeResponseCode.VehicleSpeedTooHigh => "Vehicle must be stationary",
            UDSNegativeResponseCode.VoltageTooLow => "Battery voltage too low",
            UDSNegativeResponseCode.VoltageTooHigh => "Battery voltage too high",
            UDSNegativeResponseCode.ShifterLeverNotInPark => "Transmission must be in Park",
            UDSNegativeResponseCode.RPMTooHigh => "Engine RPM too high",
            _ => $"ECU rejected request (NRC: 0x{(byte)nrc:X2})"
        };
    }
}

/// <summary>
/// Interface for bi-directional control service
/// </summary>
public interface IBiDirectionalService
{
    Task<ActuatorResponse> ControlActuatorAsync(
        ActuatorCommand command,
        VehicleState currentState,
        CancellationToken cancellationToken = default);

    SafetyValidationResult ValidateSafetyConditions(ActuatorCommand command, VehicleState state);

    Task<ActuatorResponse> ReturnControlToECUAsync(
        ushort dataIdentifier,
        CancellationToken cancellationToken = default);

    Task<List<AvailableActuator>> GetAvailableActuatorsAsync(CancellationToken cancellationToken = default);

    event EventHandler<ActuatorControlEventArgs>? ActuatorControlCompleted;
    event EventHandler<SafetyViolationEventArgs>? SafetyViolationDetected;
}

/// <summary>
/// Information about an available actuator
/// </summary>
public class AvailableActuator
{
    public ushort DataIdentifier { get; set; }
    public ActuatorType ActuatorType { get; set; }
    public string Name { get; set; } = string.Empty;
    public bool IsSupported { get; set; }
    public SafetyPreconditions RequiredPreconditions { get; set; } = SafetyPreconditions.Default;
}

/// <summary>
/// Event args for actuator control completion
/// </summary>
public class ActuatorControlEventArgs : EventArgs
{
    public ActuatorCommand Command { get; }
    public ActuatorResponse Response { get; }

    public ActuatorControlEventArgs(ActuatorCommand command, ActuatorResponse response)
    {
        Command = command;
        Response = response;
    }
}

/// <summary>
/// Event args for safety violations
/// </summary>
public class SafetyViolationEventArgs : EventArgs
{
    public ActuatorCommand Command { get; }
    public SafetyValidationResult ValidationResult { get; }

    public SafetyViolationEventArgs(ActuatorCommand command, SafetyValidationResult result)
    {
        Command = command;
        ValidationResult = result;
    }
}
