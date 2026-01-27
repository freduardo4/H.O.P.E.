using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Implementation of IRULPredictorService that invokes the Python RUL forecaster.
/// </summary>
public class RULPredictorService : IRULPredictorService
{
    private readonly ILogger<RULPredictorService>? _logger;
    private readonly string _pythonPath;
    private readonly string _scriptsPath;
    private string? _forecasterScriptPath;

    // Estimated maintenance costs by component (USD)
    private static readonly Dictionary<VehicleComponentType, double> ComponentCosts = new()
    {
        { VehicleComponentType.CatalyticConverter, 1500 },
        { VehicleComponentType.O2Sensor, 200 },
        { VehicleComponentType.SparkPlugs, 150 },
        { VehicleComponentType.Battery, 200 },
        { VehicleComponentType.BrakePads, 300 },
        { VehicleComponentType.AirFilter, 50 },
        { VehicleComponentType.FuelFilter, 100 },
        { VehicleComponentType.TimingBelt, 800 },
        { VehicleComponentType.Coolant, 100 },
        { VehicleComponentType.TransmissionFluid, 200 },
    };

    public RULPredictorService(ILogger<RULPredictorService>? logger = null)
    {
        _logger = logger;

        // Find Python installation
        _pythonPath = FindPythonPath();

        // Find scripts directory (relative to assembly location)
        var assemblyDir = Path.GetDirectoryName(typeof(RULPredictorService).Assembly.Location) ?? ".";
        _scriptsPath = Path.GetFullPath(Path.Combine(assemblyDir, "..", "..", "..", "..", "ai-training", "scripts"));

        // Fallback: try relative to working directory
        if (!Directory.Exists(_scriptsPath))
        {
            _scriptsPath = Path.GetFullPath(Path.Combine(".", "src", "ai-training", "scripts"));
        }
    }

    public RULPredictorService(string pythonPath, string scriptsPath, ILogger<RULPredictorService>? logger = null)
    {
        _logger = logger;
        _pythonPath = pythonPath;
        _scriptsPath = scriptsPath;
    }

    public bool IsAvailable => !string.IsNullOrEmpty(_pythonPath) && File.Exists(ForecasterScriptPath);

    public string ForecasterScriptPath
    {
        get
        {
            _forecasterScriptPath ??= Path.Combine(_scriptsPath, "rul_forecaster.py");
            return _forecasterScriptPath;
        }
    }

    public async Task<MaintenancePrediction> PredictMaintenanceAsync(
        string vehicleId,
        double currentOdometerKm,
        IEnumerable<ComponentTelemetry> telemetryData,
        double avgDailyKm = 50.0,
        IProgress<RULPredictionProgress>? progress = null,
        CancellationToken ct = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new MaintenancePrediction
        {
            VehicleId = vehicleId,
            OdometerKm = currentOdometerKm,
            PredictionDate = DateTime.UtcNow
        };

        try
        {
            if (!IsAvailable)
            {
                _logger?.LogWarning("Python RUL forecaster not available, using fallback estimation");
                return await PerformFallbackPredictionAsync(vehicleId, currentOdometerKm, telemetryData, avgDailyKm, progress, ct);
            }

            // Create temporary files for data exchange
            var tempDir = Path.Combine(Path.GetTempPath(), "hope_rul", Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempDir);

            try
            {
                var telemetryList = telemetryData.ToList();
                var telemetryPath = Path.Combine(tempDir, "telemetry.csv");
                var outputPath = Path.Combine(tempDir, "prediction.json");

                // Write telemetry data to CSV
                await WriteTelemetryToCsvAsync(telemetryList, telemetryPath, ct);

                // Build arguments
                var args = new StringBuilder();
                args.Append($"\"{ForecasterScriptPath}\"");
                args.Append($" --vehicle_id \"{vehicleId}\"");
                args.Append($" --odometer {currentOdometerKm.ToString(CultureInfo.InvariantCulture)}");
                args.Append($" --data_path \"{telemetryPath}\"");
                args.Append(" --output_json");
                args.Append($" --avg_daily_km {avgDailyKm.ToString(CultureInfo.InvariantCulture)}");

                _logger?.LogInformation("Starting RUL forecaster: python {Args}", args);

                // Start Python process
                var psi = new ProcessStartInfo
                {
                    FileName = _pythonPath,
                    Arguments = args.ToString(),
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true,
                    WorkingDirectory = _scriptsPath
                };

                using var process = new Process { StartInfo = psi };

                var outputBuilder = new StringBuilder();
                var errorBuilder = new StringBuilder();

                process.OutputDataReceived += (s, e) =>
                {
                    if (e.Data != null)
                    {
                        outputBuilder.AppendLine(e.Data);
                        ParseProgressOutput(e.Data, progress);
                    }
                };

                process.ErrorDataReceived += (s, e) =>
                {
                    if (e.Data != null)
                    {
                        errorBuilder.AppendLine(e.Data);
                        _logger?.LogWarning("RUL forecaster stderr: {Message}", e.Data);
                    }
                };

                process.Start();
                process.BeginOutputReadLine();
                process.BeginErrorReadLine();

                // Wait for completion with cancellation support
                using var registration = ct.Register(() =>
                {
                    try { process.Kill(true); } catch { }
                });

                await process.WaitForExitAsync(ct);

                if (process.ExitCode != 0)
                {
                    _logger?.LogWarning("RUL forecaster exited with code {Code}, using fallback", process.ExitCode);
                    return await PerformFallbackPredictionAsync(vehicleId, currentOdometerKm, telemetryData, avgDailyKm, progress, ct);
                }

                // Read JSON results
                if (File.Exists(outputPath))
                {
                    var jsonContent = await File.ReadAllTextAsync(outputPath, ct);
                    result = ParsePythonResults(jsonContent, vehicleId, currentOdometerKm);
                }
                else
                {
                    // Parse from stdout if JSON file not created
                    var jsonStart = outputBuilder.ToString().IndexOf('{');
                    if (jsonStart >= 0)
                    {
                        var jsonContent = outputBuilder.ToString()[jsonStart..];
                        result = ParsePythonResults(jsonContent, vehicleId, currentOdometerKm);
                    }
                    else
                    {
                        return await PerformFallbackPredictionAsync(vehicleId, currentOdometerKm, telemetryData, avgDailyKm, progress, ct);
                    }
                }

                result.Success = true;
            }
            finally
            {
                // Cleanup temp directory
                try { Directory.Delete(tempDir, true); } catch { }
            }
        }
        catch (OperationCanceledException)
        {
            result.Success = false;
            result.ErrorMessage = "Prediction was cancelled";
            _logger?.LogInformation("RUL prediction cancelled");
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = ex.Message;
            _logger?.LogError(ex, "RUL prediction failed, using fallback");
            return await PerformFallbackPredictionAsync(vehicleId, currentOdometerKm, telemetryData, avgDailyKm, progress, ct);
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;

        return result;
    }

    public async Task<ComponentHealth> PredictComponentRULAsync(
        VehicleComponentType component,
        double[] recentData,
        double currentOdometerKm,
        double avgDailyKm = 50.0,
        CancellationToken ct = default)
    {
        var telemetry = new List<ComponentTelemetry>
        {
            new() { Component = component, SensorData = recentData }
        };

        var prediction = await PredictMaintenanceAsync(
            "single-component",
            currentOdometerKm,
            telemetry,
            avgDailyKm,
            null,
            ct);

        return prediction.Components.FirstOrDefault(c => c.Component == component)
            ?? CreateDefaultComponentHealth(component, currentOdometerKm, avgDailyKm, recentData);
    }

    private async Task<MaintenancePrediction> PerformFallbackPredictionAsync(
        string vehicleId,
        double currentOdometerKm,
        IEnumerable<ComponentTelemetry> telemetryData,
        double avgDailyKm,
        IProgress<RULPredictionProgress>? progress,
        CancellationToken ct)
    {
        var result = new MaintenancePrediction
        {
            VehicleId = vehicleId,
            OdometerKm = currentOdometerKm,
            PredictionDate = DateTime.UtcNow,
            Success = true
        };

        var telemetryDict = telemetryData.ToDictionary(t => t.Component, t => t.SensorData);
        var components = Enum.GetValues<VehicleComponentType>();
        var totalComponents = components.Length;
        var completed = 0;

        foreach (var component in components)
        {
            ct.ThrowIfCancellationRequested();

            progress?.Report(new RULPredictionProgress
            {
                CurrentComponent = component,
                ComponentsCompleted = completed,
                TotalComponents = totalComponents,
                PercentComplete = (int)(completed * 100.0 / totalComponents),
                StatusMessage = $"Analyzing {ComponentToDisplayName(component)}..."
            });

            var sensorData = telemetryDict.GetValueOrDefault(component);
            var health = CreateDefaultComponentHealth(component, currentOdometerKm, avgDailyKm, sensorData);
            result.Components.Add(health);

            if (health.WarningLevel == WarningLevel.Critical)
            {
                result.UrgentItems.Add($"{ComponentToDisplayName(component)}: Immediate attention required");
                result.EstimatedMaintenanceCost += ComponentCosts.GetValueOrDefault(component, 200);
            }
            else if (health.WarningLevel == WarningLevel.Warning)
            {
                result.EstimatedMaintenanceCost += ComponentCosts.GetValueOrDefault(component, 200) * 0.5;
            }

            completed++;
            await Task.Yield(); // Allow cancellation checks
        }

        // Calculate overall health
        result.OverallHealth = result.Components.Average(c => c.HealthScore);

        // Find next service date based on minimum RUL
        var minRulDays = result.Components.Min(c => c.EstimatedRulDays);
        result.NextRecommendedService = DateTime.Now.AddDays(Math.Max(1, (int)(minRulDays * 0.8)));

        progress?.Report(new RULPredictionProgress
        {
            ComponentsCompleted = totalComponents,
            TotalComponents = totalComponents,
            PercentComplete = 100,
            StatusMessage = "Prediction complete"
        });

        return result;
    }

    private ComponentHealth CreateDefaultComponentHealth(
        VehicleComponentType component,
        double currentOdometerKm,
        double avgDailyKm,
        double[]? sensorData)
    {
        var typicalLifeKm = GetTypicalLifeKm(component);
        var threshold = GetDegradationThreshold(component);

        // Simple linear degradation model
        var estimatedAgeKm = currentOdometerKm % typicalLifeKm;
        var healthScore = Math.Max(0.0, 1.0 - (estimatedAgeKm / typicalLifeKm));
        var degradationRate = 1.0 / typicalLifeKm * 1000; // per 1000 km

        // Use actual sensor data if available
        if (sensorData != null && sensorData.Length > 0)
        {
            healthScore = sensorData[^1];
            if (healthScore > 1 && healthScore <= 100)
            {
                healthScore /= 100; // Normalize percentage
            }

            if (sensorData.Length > 1)
            {
                var totalDegradation = sensorData[0] - sensorData[^1];
                var stepsPerThousandKm = avgDailyKm > 0 ? 1000 / avgDailyKm : 20;
                degradationRate = Math.Abs(totalDegradation) < 1e-6
                    ? 0.0
                    : (totalDegradation / sensorData.Length) * stepsPerThousandKm;
            }
        }

        var rulKm = Math.Max(0, typicalLifeKm - estimatedAgeKm);
        var rulDays = avgDailyKm > 0 ? (int)(rulKm / avgDailyKm) : 365;

        var warningLevel = healthScore < threshold
            ? WarningLevel.Critical
            : healthScore < threshold + 0.2
                ? WarningLevel.Warning
                : WarningLevel.Normal;

        var factors = new List<string>();
        if (sensorData == null)
        {
            factors.Add("Default estimation - no sensor data available");
        }
        else if (sensorData.Length > 1)
        {
            if (sensorData[^1] - sensorData[^Math.Min(10, sensorData.Length)] < -0.1)
            {
                factors.Add("Accelerated degradation detected");
            }
        }

        return new ComponentHealth
        {
            Component = component,
            HealthScore = healthScore,
            EstimatedRulKm = rulKm,
            EstimatedRulDays = rulDays,
            Confidence = sensorData != null ? 0.7 : 0.5,
            DegradationRate = degradationRate,
            LastServiceKm = currentOdometerKm - estimatedAgeKm,
            RecommendedServiceKm = currentOdometerKm + rulKm * 0.8,
            WarningLevel = warningLevel,
            ContributingFactors = factors
        };
    }

    private MaintenancePrediction ParsePythonResults(string json, string vehicleId, double odometerKm)
    {
        var result = new MaintenancePrediction
        {
            VehicleId = vehicleId,
            OdometerKm = odometerKm,
            PredictionDate = DateTime.UtcNow
        };

        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.TryGetProperty("overall_health", out var overallHealth))
            {
                result.OverallHealth = overallHealth.GetDouble();
            }

            if (root.TryGetProperty("estimated_maintenance_cost", out var cost))
            {
                result.EstimatedMaintenanceCost = cost.GetDouble();
            }

            if (root.TryGetProperty("next_recommended_service", out var nextService))
            {
                if (DateTime.TryParse(nextService.GetString(), out var serviceDate))
                {
                    result.NextRecommendedService = serviceDate;
                }
            }

            if (root.TryGetProperty("urgent_items", out var urgentItems))
            {
                foreach (var item in urgentItems.EnumerateArray())
                {
                    result.UrgentItems.Add(item.GetString() ?? "Unknown");
                }
            }

            if (root.TryGetProperty("components", out var components))
            {
                foreach (var comp in components.EnumerateArray())
                {
                    var health = new ComponentHealth();

                    if (comp.TryGetProperty("component", out var compType))
                    {
                        health.Component = ParseComponentType(compType.GetString());
                    }

                    if (comp.TryGetProperty("health_score", out var score))
                    {
                        health.HealthScore = score.GetDouble();
                    }

                    if (comp.TryGetProperty("estimated_rul_km", out var rulKm))
                    {
                        health.EstimatedRulKm = rulKm.GetDouble();
                    }

                    if (comp.TryGetProperty("estimated_rul_days", out var rulDays))
                    {
                        health.EstimatedRulDays = rulDays.GetInt32();
                    }

                    if (comp.TryGetProperty("confidence", out var confidence))
                    {
                        health.Confidence = confidence.GetDouble();
                    }

                    if (comp.TryGetProperty("degradation_rate", out var rate))
                    {
                        health.DegradationRate = rate.GetDouble();
                    }

                    if (comp.TryGetProperty("warning_level", out var warning))
                    {
                        health.WarningLevel = ParseWarningLevel(warning.GetString());
                    }

                    if (comp.TryGetProperty("contributing_factors", out var factors))
                    {
                        foreach (var factor in factors.EnumerateArray())
                        {
                            health.ContributingFactors.Add(factor.GetString() ?? "Unknown");
                        }
                    }

                    result.Components.Add(health);
                }
            }

            result.Success = true;
        }
        catch (Exception ex)
        {
            _logger?.LogError(ex, "Failed to parse Python RUL results");
            result.Success = false;
            result.ErrorMessage = $"Failed to parse results: {ex.Message}";
        }

        return result;
    }

    private void ParseProgressOutput(string line, IProgress<RULPredictionProgress>? progress)
    {
        if (progress == null) return;

        // Parse lines like: "Analyzing catalytic_converter (1/10)..."
        if (line.Contains("Analyzing") && line.Contains("/"))
        {
            try
            {
                var parts = line.Split('(', ')');
                if (parts.Length >= 2)
                {
                    var progressPart = parts[1].Split('/');
                    if (int.TryParse(progressPart[0], out var current) &&
                        int.TryParse(progressPart[1], out var total))
                    {
                        var componentName = line.Split("Analyzing")[1].Split('(')[0].Trim();

                        progress.Report(new RULPredictionProgress
                        {
                            CurrentComponent = ParseComponentType(componentName),
                            ComponentsCompleted = current - 1,
                            TotalComponents = total,
                            PercentComplete = (int)((current - 1) * 100.0 / total),
                            StatusMessage = line.Trim()
                        });
                    }
                }
            }
            catch
            {
                // Ignore parsing errors
            }
        }
    }

    private static async Task WriteTelemetryToCsvAsync(
        List<ComponentTelemetry> telemetryData,
        string path,
        CancellationToken ct)
    {
        var sb = new StringBuilder();
        sb.AppendLine("component,timestamp,value");

        foreach (var telemetry in telemetryData)
        {
            var componentName = ComponentToSnakeCase(telemetry.Component);

            for (int i = 0; i < telemetry.SensorData.Length; i++)
            {
                var timestamp = telemetry.Timestamps != null && i < telemetry.Timestamps.Length
                    ? telemetry.Timestamps[i].ToString("O")
                    : DateTime.UtcNow.AddDays(-telemetry.SensorData.Length + i).ToString("O");

                sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                    "{0},{1},{2}",
                    componentName,
                    timestamp,
                    telemetry.SensorData[i]));
            }
        }

        await File.WriteAllTextAsync(path, sb.ToString(), ct);
    }

    private static VehicleComponentType ParseComponentType(string? value)
    {
        return value?.ToLowerInvariant().Replace("_", "") switch
        {
            "catalyticconverter" => VehicleComponentType.CatalyticConverter,
            "o2sensor" => VehicleComponentType.O2Sensor,
            "sparkplugs" => VehicleComponentType.SparkPlugs,
            "battery" => VehicleComponentType.Battery,
            "brakepads" => VehicleComponentType.BrakePads,
            "airfilter" => VehicleComponentType.AirFilter,
            "fuelfilter" => VehicleComponentType.FuelFilter,
            "timingbelt" => VehicleComponentType.TimingBelt,
            "coolant" => VehicleComponentType.Coolant,
            "transmissionfluid" => VehicleComponentType.TransmissionFluid,
            _ => VehicleComponentType.Battery
        };
    }

    private static WarningLevel ParseWarningLevel(string? value)
    {
        return value?.ToLowerInvariant() switch
        {
            "critical" => WarningLevel.Critical,
            "warning" => WarningLevel.Warning,
            _ => WarningLevel.Normal
        };
    }

    private static string ComponentToSnakeCase(VehicleComponentType component)
    {
        return component switch
        {
            VehicleComponentType.CatalyticConverter => "catalytic_converter",
            VehicleComponentType.O2Sensor => "o2_sensor",
            VehicleComponentType.SparkPlugs => "spark_plugs",
            VehicleComponentType.Battery => "battery",
            VehicleComponentType.BrakePads => "brake_pads",
            VehicleComponentType.AirFilter => "air_filter",
            VehicleComponentType.FuelFilter => "fuel_filter",
            VehicleComponentType.TimingBelt => "timing_belt",
            VehicleComponentType.Coolant => "coolant",
            VehicleComponentType.TransmissionFluid => "transmission_fluid",
            _ => "unknown"
        };
    }

    private static string ComponentToDisplayName(VehicleComponentType component)
    {
        return component switch
        {
            VehicleComponentType.CatalyticConverter => "Catalytic Converter",
            VehicleComponentType.O2Sensor => "O2 Sensor",
            VehicleComponentType.SparkPlugs => "Spark Plugs",
            VehicleComponentType.Battery => "Battery",
            VehicleComponentType.BrakePads => "Brake Pads",
            VehicleComponentType.AirFilter => "Air Filter",
            VehicleComponentType.FuelFilter => "Fuel Filter",
            VehicleComponentType.TimingBelt => "Timing Belt",
            VehicleComponentType.Coolant => "Coolant",
            VehicleComponentType.TransmissionFluid => "Transmission Fluid",
            _ => "Unknown"
        };
    }

    private static double GetTypicalLifeKm(VehicleComponentType component)
    {
        return component switch
        {
            VehicleComponentType.CatalyticConverter => 150000,
            VehicleComponentType.O2Sensor => 100000,
            VehicleComponentType.SparkPlugs => 50000,
            VehicleComponentType.Battery => 80000,
            VehicleComponentType.BrakePads => 50000,
            VehicleComponentType.AirFilter => 20000,
            VehicleComponentType.FuelFilter => 40000,
            VehicleComponentType.TimingBelt => 100000,
            VehicleComponentType.Coolant => 50000,
            VehicleComponentType.TransmissionFluid => 60000,
            _ => 100000
        };
    }

    private static double GetDegradationThreshold(VehicleComponentType component)
    {
        return component switch
        {
            VehicleComponentType.CatalyticConverter => 0.7,
            VehicleComponentType.O2Sensor => 0.6,
            VehicleComponentType.SparkPlugs => 0.5,
            VehicleComponentType.Battery => 0.6,
            VehicleComponentType.BrakePads => 0.3,
            VehicleComponentType.AirFilter => 0.5,
            VehicleComponentType.FuelFilter => 0.5,
            VehicleComponentType.TimingBelt => 0.4,
            VehicleComponentType.Coolant => 0.5,
            VehicleComponentType.TransmissionFluid => 0.5,
            _ => 0.5
        };
    }

    private static string FindPythonPath()
    {
        // Check common Python paths
        var candidates = new[]
        {
            "python",
            "python3",
            @"C:\Python312\python.exe",
            @"C:\Python311\python.exe",
            @"C:\Python310\python.exe",
            @"C:\Python39\python.exe",
            @"C:\Users\Default\AppData\Local\Programs\Python\Python312\python.exe",
            @"C:\Users\Default\AppData\Local\Programs\Python\Python311\python.exe",
            "/usr/bin/python3",
            "/usr/local/bin/python3",
        };

        foreach (var candidate in candidates)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = candidate,
                    Arguments = "--version",
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using var process = Process.Start(psi);
                if (process != null)
                {
                    process.WaitForExit(5000);
                    if (process.ExitCode == 0)
                    {
                        return candidate;
                    }
                }
            }
            catch
            {
                // Continue to next candidate
            }
        }

        // Try PATH
        var pathEnv = Environment.GetEnvironmentVariable("PATH") ?? "";
        var pathDirs = pathEnv.Split(Path.PathSeparator);

        foreach (var dir in pathDirs)
        {
            var pythonPath = Path.Combine(dir, "python.exe");
            if (File.Exists(pythonPath))
                return pythonPath;

            pythonPath = Path.Combine(dir, "python");
            if (File.Exists(pythonPath))
                return pythonPath;
        }

        return "python"; // Fallback to PATH lookup
    }
}
