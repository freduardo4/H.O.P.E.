namespace HOPE.Core.Models;

/// <summary>
/// Command to control a vehicle actuator via UDS Service 0x2F (I/O Control).
/// </summary>
public class ActuatorCommand
{
    /// <summary>
    /// Type of actuator to control
    /// </summary>
    public ActuatorType ActuatorType { get; set; }

    /// <summary>
    /// Control action to perform
    /// </summary>
    public ActuatorAction Action { get; set; }

    /// <summary>
    /// Duration of activation in milliseconds (max 5000ms for safety)
    /// </summary>
    public int DurationMs { get; set; }

    /// <summary>
    /// Control value (0-255 for adjustable actuators)
    /// </summary>
    public byte? ControlValue { get; set; }

    /// <summary>
    /// UDS Data Identifier (DID) for this actuator
    /// </summary>
    public ushort DataIdentifier { get; set; }

    /// <summary>
    /// Optional sub-function for specific control modes
    /// </summary>
    public byte? SubFunction { get; set; }

    /// <summary>
    /// Human-readable description of the command
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Safety preconditions that must be met before execution
    /// </summary>
    public SafetyPreconditions RequiredPreconditions { get; set; } = SafetyPreconditions.Default;

    /// <summary>
    /// Maximum allowed duration for this actuator type (safety limit)
    /// </summary>
    public int MaxDurationMs => ActuatorType switch
    {
        ActuatorType.FuelPump => 5000,
        ActuatorType.CoolingFan => 30000,
        ActuatorType.Injector => 500,
        ActuatorType.IgnitionCoil => 500,
        ActuatorType.ThrottleBody => 3000,
        ActuatorType.EGRValve => 5000,
        ActuatorType.PurgeSolenoid => 5000,
        ActuatorType.IdleAirControl => 5000,
        ActuatorType.VVTSolenoid => 5000,
        ActuatorType.WastegateSolenoid => 3000,
        ActuatorType.ACCompressor => 10000,
        ActuatorType.HornRelay => 2000,
        ActuatorType.HeadlightRelay => 30000,
        ActuatorType.WindowMotor => 10000,
        ActuatorType.DoorLock => 2000,
        ActuatorType.MirrorMotor => 5000,
        _ => 5000
    };

    /// <summary>
    /// Validate the command parameters
    /// </summary>
    public ActuatorCommandValidation Validate()
    {
        var validation = new ActuatorCommandValidation();

        if (DurationMs <= 0)
        {
            validation.AddError("Duration must be greater than 0");
        }
        else if (DurationMs > MaxDurationMs)
        {
            validation.AddError($"Duration {DurationMs}ms exceeds maximum allowed {MaxDurationMs}ms for {ActuatorType}");
        }

        if (DataIdentifier == 0)
        {
            validation.AddError("Data Identifier (DID) must be specified");
        }

        if (RequiresControlValue() && !ControlValue.HasValue)
        {
            validation.AddError($"Control value is required for {ActuatorType} with action {Action}");
        }

        return validation;
    }

    private bool RequiresControlValue()
    {
        return Action == ActuatorAction.ShortTermAdjustment ||
               Action == ActuatorAction.LongTermAdjustment;
    }
}

/// <summary>
/// Types of vehicle actuators that can be controlled
/// </summary>
public enum ActuatorType
{
    // Engine/Fuel System
    FuelPump,
    Injector,
    IgnitionCoil,
    ThrottleBody,
    IdleAirControl,
    EGRValve,
    PurgeSolenoid,
    VVTSolenoid,
    WastegateSolenoid,

    // Cooling System
    CoolingFan,
    ThermostatHeater,

    // HVAC
    ACCompressor,
    BlowerMotor,
    HeaterValve,

    // Lighting
    HeadlightRelay,
    TaillightRelay,
    TurnSignal,
    InteriorLight,

    // Body Control
    HornRelay,
    WindowMotor,
    DoorLock,
    MirrorMotor,
    SunroofMotor,
    WiperMotor,

    // Transmission
    ShiftSolenoid,
    TorqueConverterClutch,

    // Other
    Custom
}

/// <summary>
/// UDS I/O Control actions (Service 0x2F sub-functions)
/// </summary>
public enum ActuatorAction
{
    /// <summary>Return control to ECU (0x00)</summary>
    ReturnControlToECU = 0x00,

    /// <summary>Reset to default value (0x01)</summary>
    ResetToDefault = 0x01,

    /// <summary>Freeze current state (0x02)</summary>
    FreezeCurrentState = 0x02,

    /// <summary>Short-term adjustment (0x03) - value returns to normal after timeout</summary>
    ShortTermAdjustment = 0x03,

    /// <summary>Long-term adjustment (0x04) - value persists until reset</summary>
    LongTermAdjustment = 0x04
}

/// <summary>
/// Safety preconditions that must be met before actuator control
/// </summary>
public class SafetyPreconditions
{
    /// <summary>Engine must be off</summary>
    public bool RequireEngineOff { get; set; }

    /// <summary>Vehicle must be stationary</summary>
    public bool RequireVehicleStationary { get; set; }

    /// <summary>Transmission must be in Park or Neutral</summary>
    public bool RequireParkOrNeutral { get; set; }

    /// <summary>Parking brake must be engaged</summary>
    public bool RequireParkingBrake { get; set; }

    /// <summary>Minimum battery voltage required</summary>
    public double MinBatteryVoltage { get; set; } = 12.5;

    /// <summary>J2534 adapter must be connected (not ELM327)</summary>
    public bool RequireJ2534 { get; set; }

    /// <summary>Security access must be granted</summary>
    public bool RequireSecurityAccess { get; set; }

    /// <summary>Ignition must be on</summary>
    public bool RequireIgnitionOn { get; set; }

    /// <summary>Default safety preconditions for most actuators</summary>
    public static SafetyPreconditions Default => new()
    {
        RequireEngineOff = true,
        RequireVehicleStationary = true,
        RequireParkOrNeutral = true,
        RequireParkingBrake = true,
        MinBatteryVoltage = 12.5,
        RequireJ2534 = true,
        RequireSecurityAccess = false,
        RequireIgnitionOn = true
    };

    /// <summary>Relaxed preconditions for non-critical actuators (lights, horn)</summary>
    public static SafetyPreconditions Relaxed => new()
    {
        RequireEngineOff = false,
        RequireVehicleStationary = true,
        RequireParkOrNeutral = false,
        RequireParkingBrake = false,
        MinBatteryVoltage = 12.0,
        RequireJ2534 = false,
        RequireSecurityAccess = false,
        RequireIgnitionOn = true
    };

    /// <summary>Strict preconditions for engine/powertrain actuators</summary>
    public static SafetyPreconditions Strict => new()
    {
        RequireEngineOff = true,
        RequireVehicleStationary = true,
        RequireParkOrNeutral = true,
        RequireParkingBrake = true,
        MinBatteryVoltage = 13.0,
        RequireJ2534 = true,
        RequireSecurityAccess = true,
        RequireIgnitionOn = true
    };
}

/// <summary>
/// Current state of the vehicle for safety validation
/// </summary>
public class VehicleState
{
    /// <summary>Engine RPM (0 = off)</summary>
    public int EngineRPM { get; set; }

    /// <summary>Vehicle speed in km/h</summary>
    public double VehicleSpeedKmh { get; set; }

    /// <summary>Current gear position</summary>
    public GearPosition GearPosition { get; set; }

    /// <summary>Parking brake engaged</summary>
    public bool ParkingBrakeEngaged { get; set; }

    /// <summary>Ignition state</summary>
    public IgnitionState IgnitionState { get; set; }

    /// <summary>Battery voltage</summary>
    public double BatteryVoltage { get; set; }

    /// <summary>Is engine running</summary>
    public bool IsEngineRunning => EngineRPM > 400;

    /// <summary>Is vehicle stationary</summary>
    public bool IsStationary => VehicleSpeedKmh < 1.0;

    /// <summary>Is vehicle in Park or Neutral</summary>
    public bool IsInParkOrNeutral => GearPosition == GearPosition.Park || GearPosition == GearPosition.Neutral;
}

/// <summary>
/// Gear position enumeration
/// </summary>
public enum GearPosition
{
    Unknown,
    Park,
    Reverse,
    Neutral,
    Drive,
    Low,
    Manual
}

/// <summary>
/// Ignition state enumeration
/// </summary>
public enum IgnitionState
{
    Off,
    Accessory,
    On,
    Start
}

/// <summary>
/// Response from actuator control operation
/// </summary>
public class ActuatorResponse
{
    /// <summary>Whether the operation was successful</summary>
    public bool Success { get; set; }

    /// <summary>UDS response code</summary>
    public byte ResponseCode { get; set; }

    /// <summary>Response message</summary>
    public string Message { get; set; } = string.Empty;

    /// <summary>Actual duration of activation in milliseconds</summary>
    public int ActualDurationMs { get; set; }

    /// <summary>Any returned data from the ECU</summary>
    public byte[]? ResponseData { get; set; }

    /// <summary>Negative response code if operation failed</summary>
    public UDSNegativeResponseCode? NegativeResponseCode { get; set; }

    /// <summary>Create a successful response</summary>
    public static ActuatorResponse Successful(int durationMs, string message = "Actuator control successful")
    {
        return new ActuatorResponse
        {
            Success = true,
            ResponseCode = 0x6F, // Positive response for InputOutputControlByIdentifier
            Message = message,
            ActualDurationMs = durationMs
        };
    }

    /// <summary>Create a failed response</summary>
    public static ActuatorResponse Failed(string message, UDSNegativeResponseCode? nrc = null)
    {
        return new ActuatorResponse
        {
            Success = false,
            ResponseCode = 0x7F, // Negative response
            Message = message,
            NegativeResponseCode = nrc
        };
    }
}

/// <summary>
/// UDS Negative Response Codes (ISO 14229-1)
/// </summary>
public enum UDSNegativeResponseCode : byte
{
    GeneralReject = 0x10,
    ServiceNotSupported = 0x11,
    SubFunctionNotSupported = 0x12,
    IncorrectMessageLengthOrFormat = 0x13,
    ResponseTooLong = 0x14,
    BusyRepeatRequest = 0x21,
    ConditionsNotCorrect = 0x22,
    RequestSequenceError = 0x24,
    NoResponseFromSubnetComponent = 0x25,
    FailurePreventsExecution = 0x26,
    RequestOutOfRange = 0x31,
    SecurityAccessDenied = 0x33,
    InvalidKey = 0x35,
    ExceededNumberOfAttempts = 0x36,
    RequiredTimeDelayNotExpired = 0x37,
    UploadDownloadNotAccepted = 0x70,
    TransferDataSuspended = 0x71,
    GeneralProgrammingFailure = 0x72,
    WrongBlockSequenceCounter = 0x73,
    RequestCorrectlyReceivedButResponsePending = 0x78,
    SubFunctionNotSupportedInActiveSession = 0x7E,
    ServiceNotSupportedInActiveSession = 0x7F,
    RPMTooHigh = 0x81,
    RPMTooLow = 0x82,
    EngineIsRunning = 0x83,
    EngineIsNotRunning = 0x84,
    EngineRunTimeTooLow = 0x85,
    TemperatureTooHigh = 0x86,
    TemperatureTooLow = 0x87,
    VehicleSpeedTooHigh = 0x88,
    VehicleSpeedTooLow = 0x89,
    ThrottlePedalTooHigh = 0x8A,
    ThrottlePedalTooLow = 0x8B,
    TransmissionRangeNotInNeutral = 0x8C,
    TransmissionRangeNotInGear = 0x8D,
    BrakeSwitchNotClosed = 0x8F,
    ShifterLeverNotInPark = 0x90,
    TorqueConverterClutchLocked = 0x91,
    VoltageTooHigh = 0x92,
    VoltageTooLow = 0x93
}

/// <summary>
/// Validation result for actuator commands
/// </summary>
public class ActuatorCommandValidation
{
    public bool IsValid => Errors.Count == 0;
    public List<string> Errors { get; } = new();
    public List<string> Warnings { get; } = new();

    public void AddError(string error) => Errors.Add(error);
    public void AddWarning(string warning) => Warnings.Add(warning);
}

/// <summary>
/// Result of safety precondition validation
/// </summary>
public class SafetyValidationResult
{
    public bool IsSafe { get; set; }
    public List<SafetyViolation> Violations { get; } = new();

    public void AddViolation(string condition, string current, string required)
    {
        Violations.Add(new SafetyViolation(condition, current, required));
    }
}

/// <summary>
/// Details of a safety precondition violation
/// </summary>
public class SafetyViolation
{
    public string Condition { get; }
    public string CurrentState { get; }
    public string RequiredState { get; }

    public SafetyViolation(string condition, string current, string required)
    {
        Condition = condition;
        CurrentState = current;
        RequiredState = required;
    }

    public override string ToString() => $"{Condition}: {CurrentState} (required: {RequiredState})";
}
