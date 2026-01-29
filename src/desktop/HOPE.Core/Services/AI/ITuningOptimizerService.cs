using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Interface for AI-powered ECU tuning optimization using genetic algorithms.
/// </summary>
public interface ITuningOptimizerService
{
    /// <summary>
    /// Optimize a calibration map using genetic algorithm evolution.
    /// </summary>
    /// <param name="baselineMap">The baseline VE/fuel map to optimize</param>
    /// <param name="telemetryData">Telemetry data for fitness evaluation</param>
    /// <param name="options">Optimization configuration options</param>
    /// <param name="progress">Progress reporter</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Optimization result containing the optimized map</returns>
    Task<OptimizationResult> OptimizeAsync(
        CalibrationMap baselineMap,
        IEnumerable<TelemetryDataPoint> telemetryData,
        OptimizationOptions options,
        IProgress<OptimizationProgress>? progress = null,
        CancellationToken ct = default);

    /// <summary>
    /// Check if the optimizer is available (Python environment ready)
    /// </summary>
    bool IsAvailable { get; }

    /// <summary>
    /// Get the path to the Python optimizer script
    /// </summary>
    string OptimizerScriptPath { get; }
}

/// <summary>
/// Represents a 2D calibration map (VE table, fuel table, etc.)
/// </summary>
public class CalibrationMap
{
    /// <summary>
    /// Name of the map (e.g., "VE Table", "Fuel Map")
    /// </summary>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// RPM axis breakpoints
    /// </summary>
    public double[] RpmAxis { get; set; } = Array.Empty<double>();

    /// <summary>
    /// Load/MAP axis breakpoints
    /// </summary>
    public double[] LoadAxis { get; set; } = Array.Empty<double>();

    /// <summary>
    /// 2D array of values [rpm_index, load_index]
    /// </summary>
    public double[,] Values { get; set; } = new double[0, 0];

    /// <summary>
    /// Minimum allowed value in the map
    /// </summary>
    public double MinValue { get; set; } = 0.0;

    /// <summary>
    /// Maximum allowed value in the map
    /// </summary>
    public double MaxValue { get; set; } = 200.0;

    /// <summary>
    /// Interpolate a value at the given RPM and load
    /// </summary>
    public double Interpolate(double rpm, double load)
    {
        // Find surrounding indices
        int rpmLow = 0, rpmHigh = RpmAxis.Length - 1;
        int loadLow = 0, loadHigh = LoadAxis.Length - 1;

        for (int i = 0; i < RpmAxis.Length - 1; i++)
        {
            if (rpm >= RpmAxis[i] && rpm <= RpmAxis[i + 1])
            {
                rpmLow = i;
                rpmHigh = i + 1;
                break;
            }
        }

        for (int i = 0; i < LoadAxis.Length - 1; i++)
        {
            if (load >= LoadAxis[i] && load <= LoadAxis[i + 1])
            {
                loadLow = i;
                loadHigh = i + 1;
                break;
            }
        }

        // Bilinear interpolation
        double rpmFrac = (RpmAxis[rpmHigh] - RpmAxis[rpmLow]) > 0
            ? (rpm - RpmAxis[rpmLow]) / (RpmAxis[rpmHigh] - RpmAxis[rpmLow])
            : 0;
        double loadFrac = (LoadAxis[loadHigh] - LoadAxis[loadLow]) > 0
            ? (load - LoadAxis[loadLow]) / (LoadAxis[loadHigh] - LoadAxis[loadLow])
            : 0;

        double v00 = Values[rpmLow, loadLow];
        double v01 = Values[rpmLow, loadHigh];
        double v10 = Values[rpmHigh, loadLow];
        double v11 = Values[rpmHigh, loadHigh];

        double v0 = v00 * (1 - loadFrac) + v01 * loadFrac;
        double v1 = v10 * (1 - loadFrac) + v11 * loadFrac;

        return v0 * (1 - rpmFrac) + v1 * rpmFrac;
    }

    /// <summary>
    /// Create a deep copy of this map
    /// </summary>
    public CalibrationMap Clone()
    {
        var clone = new CalibrationMap
        {
            Name = Name,
            RpmAxis = (double[])RpmAxis.Clone(),
            LoadAxis = (double[])LoadAxis.Clone(),
            MinValue = MinValue,
            MaxValue = MaxValue
        };

        clone.Values = new double[Values.GetLength(0), Values.GetLength(1)];
        Array.Copy(Values, clone.Values, Values.Length);

        return clone;
    }
}

/// <summary>
/// Single telemetry data point for optimization fitness evaluation
/// </summary>
public class TelemetryDataPoint
{
    public double Rpm { get; set; }
    public double Load { get; set; }
    public double ActualAfr { get; set; }
    public double TargetAfr { get; set; }
    public double Maf { get; set; }
    public double CoolantTemp { get; set; } = 90.0;
    public double IntakeTemp { get; set; } = 25.0;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Options for configuring the genetic algorithm optimization
/// </summary>
public class OptimizationOptions
{
    /// <summary>
    /// Optimization objective/goal
    /// </summary>
    public OptimizationObjective Objective { get; set; } = OptimizationObjective.AfrAccuracy;

    /// <summary>
    /// Number of generations to evolve
    /// </summary>
    public int Generations { get; set; } = 50;

    /// <summary>
    /// Population size for the genetic algorithm
    /// </summary>
    public int PopulationSize { get; set; } = 50;

    /// <summary>
    /// Mutation rate (0.0 to 1.0)
    /// </summary>
    public double MutationRate { get; set; } = 0.1;

    /// <summary>
    /// Crossover rate (0.0 to 1.0)
    /// </summary>
    public double CrossoverRate { get; set; } = 0.7;

    /// <summary>
    /// Target fitness to stop early (0.0 to 1.0)
    /// </summary>
    public double TargetFitness { get; set; } = 0.95;

    /// <summary>
    /// Whether to use RL-guided parameter optimization
    /// </summary>
    public bool UseRlGuided { get; set; } = false;

    /// <summary>
    /// Whether to run safety audit before and after optimization
    /// </summary>
    public bool RunSafetyAudit { get; set; } = true;

    /// <summary>
    /// Whether to automatically identify map type
    /// </summary>
    public bool AutoClassifyMap { get; set; } = true;
}

/// <summary>
/// Optimization objective types
/// </summary>
public enum OptimizationObjective
{
    /// <summary>
    /// Minimize AFR deviation from target
    /// </summary>
    AfrAccuracy,

    /// <summary>
    /// Optimize for fuel economy (slightly lean)
    /// </summary>
    FuelEconomy,

    /// <summary>
    /// Optimize for power output (slightly rich)
    /// </summary>
    PowerOutput,

    /// <summary>
    /// Optimize for emissions (stoichiometric)
    /// </summary>
    Emissions,

    /// <summary>
    /// Balanced multi-objective optimization
    /// </summary>
    Balanced
}

/// <summary>
/// Result of the optimization process
/// </summary>
public class OptimizationResult
{
    /// <summary>
    /// The optimized calibration map
    /// </summary>
    public CalibrationMap OptimizedMap { get; set; } = new();

    /// <summary>
    /// Final fitness score (0.0 to 1.0)
    /// </summary>
    public double FinalFitness { get; set; }

    /// <summary>
    /// Final AFR error (RMSE)
    /// </summary>
    public double FinalAfrError { get; set; }

    /// <summary>
    /// Number of generations completed
    /// </summary>
    public int GenerationsCompleted { get; set; }

    /// <summary>
    /// Number of cells changed from baseline
    /// </summary>
    public int CellsChanged { get; set; }

    /// <summary>
    /// Maximum value increase
    /// </summary>
    public double MaxIncrease { get; set; }

    /// <summary>
    /// Maximum value decrease
    /// </summary>
    public double MaxDecrease { get; set; }

    /// <summary>
    /// Mean change across all cells
    /// </summary>
    public double MeanChange { get; set; }

    /// <summary>
    /// Evolution history (fitness per generation)
    /// </summary>
    public List<GenerationStats> History { get; set; } = new();

    /// <summary>
    /// Whether the optimization was successful
    /// </summary>
    public bool Success { get; set; }

    /// <summary>
    /// Error message if optimization failed
    /// </summary>
    public string? ErrorMessage { get; set; }

    /// <summary>
    /// Duration of the optimization
    /// </summary>
    public TimeSpan Duration { get; set; }
}

/// <summary>
/// Statistics for a single generation
/// </summary>
public class GenerationStats
{
    public int Generation { get; set; }
    public double BestFitness { get; set; }
    public double MeanFitness { get; set; }
    public double AfrError { get; set; }
}

/// <summary>
/// Progress update during optimization
/// </summary>
public class OptimizationProgress
{
    public int CurrentGeneration { get; set; }
    public int TotalGenerations { get; set; }
    public double CurrentFitness { get; set; }
    public double CurrentAfrError { get; set; }
    public int PercentComplete { get; set; }
    public string StatusMessage { get; set; } = string.Empty;
}
