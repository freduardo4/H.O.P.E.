using System.IO;
using System.Text.Json;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Production ONNX-based implementation of anomaly detection.
/// Uses an LSTM Autoencoder model trained on normal vehicle behavior.
/// Anomalies are detected based on reconstruction error threshold.
/// </summary>
public class OnnxAnomalyService : IAnomalyService, IDisposable
{
    private InferenceSession? _session;
    private ModelConfig? _config;
    private bool _disposed;

    /// <summary>
    /// Feature names in the order expected by the model.
    /// </summary>
    private static readonly string[] FeatureNames =
    {
        "engine_rpm",
        "vehicle_speed",
        "engine_load",
        "coolant_temp",
        "intake_air_temp",
        "maf_flow",
        "throttle_position",
        "fuel_pressure",
        "short_term_fuel_trim",
        "long_term_fuel_trim"
    };

    /// <summary>
    /// Mapping from OBD2 PID to feature index.
    /// </summary>
    private static readonly Dictionary<string, int> PidToFeatureIndex = new()
    {
        { OBD2PIDs.EngineRPM, 0 },
        { OBD2PIDs.VehicleSpeed, 1 },
        { OBD2PIDs.EngineLoad, 2 },
        { OBD2PIDs.CoolantTemp, 3 },
        { OBD2PIDs.IntakeAirTemp, 4 },
        { OBD2PIDs.MAFSensor, 5 },
        { OBD2PIDs.ThrottlePosition, 6 },
        { OBD2PIDs.FuelPressure, 7 },
        { OBD2PIDs.ShortTermFuelTrim, 8 },
        { OBD2PIDs.LongTermFuelTrim, 9 }
    };

    /// <summary>
    /// Number of features expected by the model.
    /// </summary>
    private const int NumFeatures = 10;

    /// <summary>
    /// Sequence length (timesteps) expected by the model.
    /// </summary>
    private const int SequenceLength = 60;

    public bool IsModelLoaded => _session != null && _config != null;

    public async Task LoadModelAsync(string modelPath)
    {
        await Task.Run(() =>
        {
            // Load ONNX model
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _session = new InferenceSession(modelPath, sessionOptions);

            // Load configuration (scaler parameters and threshold)
            var configPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "config.json");
            if (File.Exists(configPath))
            {
                var configJson = File.ReadAllText(configPath);
                _config = JsonSerializer.Deserialize<ModelConfig>(configJson);
            }
            else
            {
                // Use default configuration if config file not found
                _config = ModelConfig.CreateDefault();
            }
        });
    }

    public async Task<AnomalyResult> AnalyzeAsync(IEnumerable<OBD2Reading> readings)
    {
        var readingsList = readings.ToList();

        if (!IsModelLoaded)
        {
            return new AnomalyResult
            {
                Score = 0,
                IsAnomaly = false,
                Confidence = 0,
                Description = "Model not loaded"
            };
        }

        if (readingsList.Count == 0)
        {
            return new AnomalyResult
            {
                Score = 0,
                IsAnomaly = false,
                Confidence = 0,
                Description = "Insufficient data for analysis"
            };
        }

        try
        {
            return await Task.Run(() => RunInference(readingsList));
        }
        catch (Exception ex)
        {
            return new AnomalyResult
            {
                Score = 0,
                IsAnomaly = false,
                Confidence = 0,
                Description = $"Analysis error: {ex.Message}"
            };
        }
    }

    private AnomalyResult RunInference(List<OBD2Reading> readings)
    {
        // Prepare input tensor
        var (inputTensor, validSamples) = PrepareInputTensor(readings);

        if (validSamples < SequenceLength / 2)
        {
            return new AnomalyResult
            {
                Score = 0,
                IsAnomaly = false,
                Confidence = 0.3,
                Description = "Insufficient valid readings for analysis"
            };
        }

        // Run inference
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", inputTensor)
        };

        using var results = _session!.Run(inputs);
        var outputTensor = results.First().AsTensor<float>();

        // Calculate reconstruction error (MSE)
        double mse = CalculateMSE(inputTensor, outputTensor);

        // Normalize score to 0-1 range based on threshold
        double normalizedScore = Math.Min(mse / (_config!.Threshold * 2), 1.0);

        // Determine if anomaly
        bool isAnomaly = mse > _config.Threshold;

        // Calculate confidence based on how far from threshold
        double confidence = CalculateConfidence(mse);

        // Identify contributing parameters
        var contributingParams = IdentifyContributingParameters(inputTensor, outputTensor, readings);

        // Generate description
        string description = GenerateDescription(isAnomaly, mse, contributingParams);

        return new AnomalyResult
        {
            Score = normalizedScore,
            IsAnomaly = isAnomaly,
            Confidence = confidence,
            Description = description,
            ContributingParameters = contributingParams,
            Timestamp = DateTime.UtcNow
        };
    }

    private (DenseTensor<float> tensor, int validSamples) PrepareInputTensor(List<OBD2Reading> readings)
    {
        // Create tensor with shape [1, sequence_length, num_features]
        var tensor = new DenseTensor<float>(new[] { 1, SequenceLength, NumFeatures });

        // Group readings by timestamp (approximate to nearest second)
        var groupedReadings = readings
            .GroupBy(r => r.Timestamp.Ticks / TimeSpan.TicksPerSecond)
            .OrderByDescending(g => g.Key)
            .Take(SequenceLength)
            .Reverse()
            .ToList();

        int validSamples = 0;

        // Fill tensor with most recent readings
        for (int t = 0; t < SequenceLength; t++)
        {
            var timestepReadings = t < groupedReadings.Count
                ? groupedReadings[t].ToList()
                : new List<OBD2Reading>();

            bool hasValidReading = false;

            foreach (var reading in timestepReadings)
            {
                if (PidToFeatureIndex.TryGetValue(reading.PID, out int featureIdx))
                {
                    // Apply scaling (standardization)
                    double scaledValue = ScaleFeature(featureIdx, reading.Value);
                    tensor[0, t, featureIdx] = (float)scaledValue;
                    hasValidReading = true;
                }
            }

            if (hasValidReading)
            {
                validSamples++;
            }

            // Fill missing features with scaled zeros (mean after scaling = 0)
            for (int f = 0; f < NumFeatures; f++)
            {
                if (tensor[0, t, f] == 0)
                {
                    // Use mean value (which becomes 0 after scaling)
                    tensor[0, t, f] = 0;
                }
            }
        }

        return (tensor, validSamples);
    }

    private double ScaleFeature(int featureIndex, double value)
    {
        if (_config?.ScalerMean == null || _config.ScalerStd == null ||
            featureIndex >= _config.ScalerMean.Length)
        {
            return value; // No scaling available
        }

        double mean = _config.ScalerMean[featureIndex];
        double std = _config.ScalerStd[featureIndex];

        if (std == 0) return 0;

        return (value - mean) / std;
    }

    private static double CalculateMSE(DenseTensor<float> input, Tensor<float> output)
    {
        double sumSquaredError = 0;
        int count = 0;

        for (int t = 0; t < SequenceLength; t++)
        {
            for (int f = 0; f < NumFeatures; f++)
            {
                double inputVal = input[0, t, f];
                double outputVal = output[0, t, f];
                sumSquaredError += Math.Pow(inputVal - outputVal, 2);
                count++;
            }
        }

        return sumSquaredError / count;
    }

    private double CalculateConfidence(double mse)
    {
        if (_config == null) return 0.5;

        // Calculate confidence based on distance from threshold
        double ratio = mse / _config.Threshold;

        if (ratio < 0.5)
        {
            // Very normal - high confidence it's not anomalous
            return 0.95;
        }
        else if (ratio < 0.8)
        {
            // Somewhat normal
            return 0.85;
        }
        else if (ratio < 1.0)
        {
            // Near threshold - lower confidence
            return 0.70;
        }
        else if (ratio < 1.5)
        {
            // Anomalous but borderline
            return 0.75;
        }
        else
        {
            // Clearly anomalous - high confidence
            return 0.90 + Math.Min(ratio - 1.5, 0.5) * 0.1;
        }
    }

    private List<string> IdentifyContributingParameters(
        DenseTensor<float> input,
        Tensor<float> output,
        List<OBD2Reading> readings)
    {
        var contributions = new List<(string name, double error)>();

        // Calculate per-feature reconstruction error
        for (int f = 0; f < NumFeatures; f++)
        {
            double featureError = 0;
            for (int t = 0; t < SequenceLength; t++)
            {
                featureError += Math.Pow(input[0, t, f] - output[0, t, f], 2);
            }
            featureError /= SequenceLength;

            contributions.Add((FeatureNames[f], featureError));
        }

        // Return top 3 contributors with error above threshold
        double avgError = contributions.Average(c => c.error);

        return contributions
            .Where(c => c.error > avgError * 1.5)
            .OrderByDescending(c => c.error)
            .Take(3)
            .Select(c => FormatParameterContribution(c.name, c.error, readings))
            .ToList();
    }

    private string FormatParameterContribution(string featureName, double error, List<OBD2Reading> readings)
    {
        var recentReadings = readings
            .Where(r => PidToFeatureIndex.ContainsKey(r.PID) &&
                        PidToFeatureIndex[r.PID] == Array.IndexOf(FeatureNames, featureName))
            .OrderByDescending(r => r.Timestamp)
            .Take(5)
            .ToList();

        if (recentReadings.Count > 0)
        {
            var avg = recentReadings.Average(r => r.Value);
            var unit = recentReadings.First().Unit;
            return $"{FormatFeatureName(featureName)}: {avg:F1} {unit} (deviation detected)";
        }

        return $"{FormatFeatureName(featureName)}: Unusual pattern detected";
    }

    private static string FormatFeatureName(string name)
    {
        return name switch
        {
            "engine_rpm" => "Engine RPM",
            "vehicle_speed" => "Vehicle Speed",
            "engine_load" => "Engine Load",
            "coolant_temp" => "Coolant Temperature",
            "intake_air_temp" => "Intake Air Temperature",
            "maf_flow" => "MAF Airflow",
            "throttle_position" => "Throttle Position",
            "fuel_pressure" => "Fuel Pressure",
            "short_term_fuel_trim" => "Short Term Fuel Trim",
            "long_term_fuel_trim" => "Long Term Fuel Trim",
            _ => name.Replace('_', ' ')
        };
    }

    private string GenerateDescription(bool isAnomaly, double mse, List<string> contributingParams)
    {
        if (!isAnomaly)
        {
            return "Vehicle parameters within normal operating range";
        }

        if (contributingParams.Count == 0)
        {
            return "Unusual behavior detected - review vehicle parameters";
        }

        if (contributingParams.Count == 1)
        {
            return $"Potential issue detected: {contributingParams[0]}";
        }

        return $"Multiple anomalies detected involving {contributingParams.Count} parameters";
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _session?.Dispose();
            }
            _disposed = true;
        }
    }

    ~OnnxAnomalyService()
    {
        Dispose(false);
    }
}

/// <summary>
/// Model configuration including scaler parameters and threshold.
/// </summary>
internal class ModelConfig
{
    public int SequenceLength { get; set; }
    public int NumFeatures { get; set; }
    public int LatentDim { get; set; }
    public double Threshold { get; set; }
    public double[]? ScalerMean { get; set; }
    public double[]? ScalerStd { get; set; }
    public string[]? Features { get; set; }

    public static ModelConfig CreateDefault()
    {
        // Default configuration based on training script defaults
        return new ModelConfig
        {
            SequenceLength = 60,
            NumFeatures = 10,
            LatentDim = 16,
            Threshold = 0.1,
            // Default means and stds for typical OBD2 values
            ScalerMean = new[]
            {
                2000.0,  // engine_rpm
                50.0,    // vehicle_speed
                35.0,    // engine_load
                90.0,    // coolant_temp
                35.0,    // intake_air_temp
                20.0,    // maf_flow
                25.0,    // throttle_position
                350.0,   // fuel_pressure
                0.0,     // short_term_fuel_trim
                0.0      // long_term_fuel_trim
            },
            ScalerStd = new[]
            {
                1000.0,  // engine_rpm
                40.0,    // vehicle_speed
                20.0,    // engine_load
                15.0,    // coolant_temp
                15.0,    // intake_air_temp
                15.0,    // maf_flow
                20.0,    // throttle_position
                50.0,    // fuel_pressure
                5.0,     // short_term_fuel_trim
                5.0      // long_term_fuel_trim
            },
            Features = new[]
            {
                "engine_rpm", "vehicle_speed", "engine_load", "coolant_temp",
                "intake_air_temp", "maf_flow", "throttle_position", "fuel_pressure",
                "short_term_fuel_trim", "long_term_fuel_trim"
            }
        };
    }
}
