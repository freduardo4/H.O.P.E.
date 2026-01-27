using System;
using System.Collections.Generic;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Interface for Remaining Useful Life (RUL) prediction service.
/// Uses LSTM-based time-series forecasting to predict component degradation.
/// </summary>
public interface IRULPredictorService
{
    /// <summary>
    /// Predict remaining useful life for all tracked components.
    /// </summary>
    /// <param name="vehicleId">Vehicle identifier</param>
    /// <param name="currentOdometerKm">Current odometer reading in km</param>
    /// <param name="telemetryData">Component telemetry data for prediction</param>
    /// <param name="avgDailyKm">Average daily driving distance</param>
    /// <param name="progress">Progress reporter</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Complete maintenance prediction for the vehicle</returns>
    Task<MaintenancePrediction> PredictMaintenanceAsync(
        string vehicleId,
        double currentOdometerKm,
        IEnumerable<ComponentTelemetry> telemetryData,
        double avgDailyKm = 50.0,
        IProgress<RULPredictionProgress>? progress = null,
        CancellationToken ct = default);

    /// <summary>
    /// Predict remaining useful life for a single component.
    /// </summary>
    Task<ComponentHealth> PredictComponentRULAsync(
        VehicleComponentType component,
        double[] recentData,
        double currentOdometerKm,
        double avgDailyKm = 50.0,
        CancellationToken ct = default);

    /// <summary>
    /// Check if the predictor is available (Python environment ready)
    /// </summary>
    bool IsAvailable { get; }

    /// <summary>
    /// Get the path to the Python RUL forecaster script
    /// </summary>
    string ForecasterScriptPath { get; }
}

/// <summary>
/// Types of vehicle components tracked for RUL prediction.
/// </summary>
public enum VehicleComponentType
{
    /// <summary>Catalytic converter efficiency</summary>
    CatalyticConverter,
    /// <summary>O2 sensor response time</summary>
    O2Sensor,
    /// <summary>Spark plug degradation</summary>
    SparkPlugs,
    /// <summary>Battery health</summary>
    Battery,
    /// <summary>Brake pad wear</summary>
    BrakePads,
    /// <summary>Air filter restriction</summary>
    AirFilter,
    /// <summary>Fuel filter condition</summary>
    FuelFilter,
    /// <summary>Timing belt wear</summary>
    TimingBelt,
    /// <summary>Coolant condition</summary>
    Coolant,
    /// <summary>Transmission fluid condition</summary>
    TransmissionFluid
}

/// <summary>
/// Warning level for component health.
/// </summary>
public enum WarningLevel
{
    /// <summary>Component is in good condition</summary>
    Normal,
    /// <summary>Component needs attention soon</summary>
    Warning,
    /// <summary>Component needs immediate attention</summary>
    Critical
}

/// <summary>
/// Telemetry data for a specific component.
/// </summary>
public class ComponentTelemetry
{
    /// <summary>
    /// Component type this data is for
    /// </summary>
    public VehicleComponentType Component { get; set; }

    /// <summary>
    /// Time-series sensor data (health values from 0.0 to 1.0)
    /// </summary>
    public double[] SensorData { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Timestamps for each data point (optional)
    /// </summary>
    public DateTime[]? Timestamps { get; set; }
}

/// <summary>
/// Health status of a vehicle component.
/// </summary>
public class ComponentHealth
{
    /// <summary>
    /// Component type
    /// </summary>
    public VehicleComponentType Component { get; set; }

    /// <summary>
    /// Health score (0.0 = failed, 1.0 = new)
    /// </summary>
    public double HealthScore { get; set; }

    /// <summary>
    /// Estimated remaining useful life in kilometers
    /// </summary>
    public double EstimatedRulKm { get; set; }

    /// <summary>
    /// Estimated remaining useful life in days
    /// </summary>
    public int EstimatedRulDays { get; set; }

    /// <summary>
    /// Confidence in the prediction (0.0 to 1.0)
    /// </summary>
    public double Confidence { get; set; }

    /// <summary>
    /// Rate of degradation per 1000 km
    /// </summary>
    public double DegradationRate { get; set; }

    /// <summary>
    /// Odometer at last service
    /// </summary>
    public double LastServiceKm { get; set; }

    /// <summary>
    /// Recommended next service odometer
    /// </summary>
    public double RecommendedServiceKm { get; set; }

    /// <summary>
    /// Warning level for this component
    /// </summary>
    public WarningLevel WarningLevel { get; set; }

    /// <summary>
    /// Factors contributing to degradation
    /// </summary>
    public List<string> ContributingFactors { get; set; } = new();
}

/// <summary>
/// Complete maintenance prediction for a vehicle.
/// </summary>
public class MaintenancePrediction
{
    /// <summary>
    /// Vehicle identifier
    /// </summary>
    public string VehicleId { get; set; } = string.Empty;

    /// <summary>
    /// Current odometer reading in km
    /// </summary>
    public double OdometerKm { get; set; }

    /// <summary>
    /// Timestamp of this prediction
    /// </summary>
    public DateTime PredictionDate { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Health predictions for all components
    /// </summary>
    public List<ComponentHealth> Components { get; set; } = new();

    /// <summary>
    /// Average health score across all components
    /// </summary>
    public double OverallHealth { get; set; }

    /// <summary>
    /// Recommended next service date
    /// </summary>
    public DateTime NextRecommendedService { get; set; }

    /// <summary>
    /// Items requiring urgent attention
    /// </summary>
    public List<string> UrgentItems { get; set; } = new();

    /// <summary>
    /// Estimated total maintenance cost
    /// </summary>
    public double EstimatedMaintenanceCost { get; set; }

    /// <summary>
    /// Whether the prediction was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if prediction failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Duration of the prediction process
    /// </summary>
    public TimeSpan Duration { get; set; }
}

/// <summary>
/// Progress update during RUL prediction.
/// </summary>
public class RULPredictionProgress
{
    /// <summary>
    /// Component currently being analyzed
    /// </summary>
    public VehicleComponentType? CurrentComponent { get; set; }

    /// <summary>
    /// Number of components completed
    /// </summary>
    public int ComponentsCompleted { get; set; }

    /// <summary>
    /// Total number of components to analyze
    /// </summary>
    public int TotalComponents { get; set; }

    /// <summary>
    /// Overall percent complete (0-100)
    /// </summary>
    public int PercentComplete { get; set; }

    /// <summary>
    /// Status message
    /// </summary>
    public string StatusMessage { get; set; } = string.Empty;
}
