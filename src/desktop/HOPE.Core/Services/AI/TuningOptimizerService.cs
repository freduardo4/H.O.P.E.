using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Implementation of ITuningOptimizerService that invokes the Python genetic algorithm optimizer.
/// </summary>
public class TuningOptimizerService : ITuningOptimizerService
{
    private readonly ILogger<TuningOptimizerService>? _logger;
    private readonly string _pythonPath;
    private readonly string _scriptsPath;
    private string? _optimizerScriptPath;

    public TuningOptimizerService(ILogger<TuningOptimizerService>? logger = null)
    {
        _logger = logger;

        // Find Python installation
        _pythonPath = FindPythonPath();

        // Find scripts directory (relative to assembly location)
        var assemblyDir = Path.GetDirectoryName(typeof(TuningOptimizerService).Assembly.Location) ?? ".";
        _scriptsPath = Path.GetFullPath(Path.Combine(assemblyDir, "..", "..", "..", "..", "ai-training", "scripts"));

        // Fallback: try relative to working directory
        if (!Directory.Exists(_scriptsPath))
        {
            _scriptsPath = Path.GetFullPath(Path.Combine(".", "src", "ai-training", "scripts"));
        }
    }

    public TuningOptimizerService(string pythonPath, string scriptsPath, ILogger<TuningOptimizerService>? logger = null)
    {
        _logger = logger;
        _pythonPath = pythonPath;
        _scriptsPath = scriptsPath;
    }

    public bool IsAvailable => !string.IsNullOrEmpty(_pythonPath) && File.Exists(OptimizerScriptPath);

    public string OptimizerScriptPath
    {
        get
        {
            if (_optimizerScriptPath == null)
            {
                _optimizerScriptPath = Path.Combine(_scriptsPath, "genetic_optimizer.py");
            }
            return _optimizerScriptPath;
        }
    }

    public async Task<OptimizationResult> OptimizeAsync(
        CalibrationMap baselineMap,
        IEnumerable<TelemetryDataPoint> telemetryData,
        OptimizationOptions options,
        IProgress<OptimizationProgress>? progress = null,
        CancellationToken ct = default)
    {
        var stopwatch = Stopwatch.StartNew();
        var result = new OptimizationResult();

        try
        {
            if (!IsAvailable)
            {
                throw new InvalidOperationException(
                    $"Tuning optimizer is not available. Python path: {_pythonPath}, Script: {OptimizerScriptPath}");
            }

            // Create temporary files for data exchange
            var tempDir = Path.Combine(Path.GetTempPath(), "hope_optimizer", Guid.NewGuid().ToString());
            Directory.CreateDirectory(tempDir);

            try
            {
                var baselineMapPath = Path.Combine(tempDir, "baseline_map.csv");
                var telemetryPath = Path.Combine(tempDir, "telemetry.csv");
                var outputMapPath = Path.Combine(tempDir, "optimized_map.csv");
                var resultsPath = Path.Combine(tempDir, "optimized_map.json");

                // Write baseline map to CSV
                await WriteMapToCsvAsync(baselineMap, baselineMapPath, ct);

                // Write telemetry data to CSV
                await WriteTelemetryToCsvAsync(telemetryData, telemetryPath, ct);

                // Build arguments
                var args = new StringBuilder();
                args.Append($"\"{OptimizerScriptPath}\"");
                args.Append($" --baseline_map \"{baselineMapPath}\"");
                args.Append($" --telemetry \"{telemetryPath}\"");
                args.Append($" --output \"{outputMapPath}\"");
                args.Append($" --generations {options.Generations}");
                args.Append($" --population {options.PopulationSize}");
                args.Append($" --mutation_rate {options.MutationRate.ToString(CultureInfo.InvariantCulture)}");
                args.Append($" --crossover_rate {options.CrossoverRate.ToString(CultureInfo.InvariantCulture)}");
                args.Append($" --objective {ObjectiveToString(options.Objective)}");

                _logger?.LogInformation("Starting genetic optimizer: python {Args}", args);

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
                        ParseProgressOutput(e.Data, options.Generations, progress);
                    }
                };

                process.ErrorDataReceived += (s, e) =>
                {
                    if (e.Data != null)
                    {
                        errorBuilder.AppendLine(e.Data);
                        _logger?.LogWarning("Optimizer stderr: {Message}", e.Data);
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
                    throw new InvalidOperationException(
                        $"Optimizer exited with code {process.ExitCode}: {errorBuilder}");
                }

                // Read results
                if (File.Exists(resultsPath))
                {
                    var resultsJson = await File.ReadAllTextAsync(resultsPath, ct);
                    var pythonResults = JsonSerializer.Deserialize<PythonOptimizerResults>(resultsJson);

                    if (pythonResults != null)
                    {
                        result.FinalFitness = pythonResults.improvement_report?.fitness_improvement ?? 0;
                        result.FinalAfrError = pythonResults.improvement_report?.final_afr_error ?? 0;
                        result.GenerationsCompleted = pythonResults.improvement_report?.generations_run ?? options.Generations;
                        result.CellsChanged = pythonResults.improvement_report?.cells_changed ?? 0;
                        result.MaxIncrease = pythonResults.improvement_report?.max_increase ?? 0;
                        result.MaxDecrease = pythonResults.improvement_report?.max_decrease ?? 0;
                        result.MeanChange = pythonResults.improvement_report?.mean_change ?? 0;

                        // Parse history
                        if (pythonResults.evolution_history != null)
                        {
                            result.History = pythonResults.evolution_history
                                .Select(h => new GenerationStats
                                {
                                    Generation = h.generation,
                                    BestFitness = h.best_fitness,
                                    MeanFitness = h.mean_fitness,
                                    AfrError = h.best_afr_error
                                })
                                .ToList();
                        }
                    }
                }

                // Read optimized map
                if (File.Exists(outputMapPath))
                {
                    result.OptimizedMap = await ReadMapFromCsvAsync(outputMapPath, baselineMap.Name, ct);
                    result.Success = true;
                }
                else
                {
                    result.Success = false;
                    result.ErrorMessage = "Optimizer did not produce output file";
                }
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
            result.ErrorMessage = "Optimization was cancelled";
            _logger?.LogInformation("Optimization cancelled");
        }
        catch (Exception ex)
        {
            result.Success = false;
            result.ErrorMessage = ex.Message;
            _logger?.LogError(ex, "Optimization failed");
        }

        stopwatch.Stop();
        result.Duration = stopwatch.Elapsed;

        return result;
    }

    private void ParseProgressOutput(string line, int totalGenerations, IProgress<OptimizationProgress>? progress)
    {
        if (progress == null) return;

        // Parse lines like: "Generation 10/50 - Best fitness: 0.8500, AFR error: 0.350"
        if (line.Contains("Generation") && line.Contains("fitness"))
        {
            try
            {
                var parts = line.Split('-', StringSplitOptions.TrimEntries);
                if (parts.Length >= 2)
                {
                    var genPart = parts[0].Replace("Generation", "").Trim();
                    var genParts = genPart.Split('/');

                    if (int.TryParse(genParts[0], out var currentGen))
                    {
                        double fitness = 0, afrError = 0;

                        if (line.Contains("fitness:"))
                        {
                            var fitnessIdx = line.IndexOf("fitness:", StringComparison.Ordinal) + 8;
                            var fitnessEnd = line.IndexOf(',', fitnessIdx);
                            if (fitnessEnd > fitnessIdx)
                            {
                                double.TryParse(line[fitnessIdx..fitnessEnd].Trim(),
                                    NumberStyles.Float, CultureInfo.InvariantCulture, out fitness);
                            }
                        }

                        if (line.Contains("AFR error:"))
                        {
                            var errorIdx = line.IndexOf("AFR error:", StringComparison.Ordinal) + 10;
                            double.TryParse(line[errorIdx..].Trim(),
                                NumberStyles.Float, CultureInfo.InvariantCulture, out afrError);
                        }

                        progress.Report(new OptimizationProgress
                        {
                            CurrentGeneration = currentGen,
                            TotalGenerations = totalGenerations,
                            CurrentFitness = fitness,
                            CurrentAfrError = afrError,
                            PercentComplete = (int)(currentGen * 100.0 / totalGenerations),
                            StatusMessage = $"Generation {currentGen}/{totalGenerations}"
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

    private static async Task WriteMapToCsvAsync(CalibrationMap map, string path, CancellationToken ct)
    {
        var sb = new StringBuilder();

        // Header row with load axis values
        sb.Append("RPM\\Load");
        foreach (var load in map.LoadAxis)
        {
            sb.Append(',').Append(load.ToString(CultureInfo.InvariantCulture));
        }
        sb.AppendLine();

        // Data rows
        for (int i = 0; i < map.RpmAxis.Length; i++)
        {
            sb.Append(map.RpmAxis[i].ToString(CultureInfo.InvariantCulture));
            for (int j = 0; j < map.LoadAxis.Length; j++)
            {
                sb.Append(',').Append(map.Values[i, j].ToString(CultureInfo.InvariantCulture));
            }
            sb.AppendLine();
        }

        await File.WriteAllTextAsync(path, sb.ToString(), ct);
    }

    private static async Task WriteTelemetryToCsvAsync(IEnumerable<TelemetryDataPoint> data, string path, CancellationToken ct)
    {
        var sb = new StringBuilder();
        sb.AppendLine("rpm,load,actual_afr,target_afr,maf,coolant_temp,intake_temp");

        foreach (var point in data)
        {
            sb.AppendLine(string.Format(CultureInfo.InvariantCulture,
                "{0},{1},{2},{3},{4},{5},{6}",
                point.Rpm, point.Load, point.ActualAfr, point.TargetAfr,
                point.Maf, point.CoolantTemp, point.IntakeTemp));
        }

        await File.WriteAllTextAsync(path, sb.ToString(), ct);
    }

    private static async Task<CalibrationMap> ReadMapFromCsvAsync(string path, string name, CancellationToken ct)
    {
        var lines = await File.ReadAllLinesAsync(path, ct);
        if (lines.Length < 2)
            throw new InvalidOperationException("Invalid map CSV format");

        // Parse header (load axis)
        var headerParts = lines[0].Split(',');
        var loadAxis = headerParts.Skip(1)
            .Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture))
            .ToArray();

        // Parse data rows
        var rpmAxis = new List<double>();
        var values = new List<double[]>();

        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrWhiteSpace(lines[i])) continue;

            var parts = lines[i].Split(',');
            rpmAxis.Add(double.Parse(parts[0].Trim(), CultureInfo.InvariantCulture));

            var row = parts.Skip(1)
                .Select(s => double.Parse(s.Trim(), CultureInfo.InvariantCulture))
                .ToArray();
            values.Add(row);
        }

        var map = new CalibrationMap
        {
            Name = name,
            RpmAxis = rpmAxis.ToArray(),
            LoadAxis = loadAxis,
            Values = new double[rpmAxis.Count, loadAxis.Length]
        };

        for (int i = 0; i < values.Count; i++)
        {
            for (int j = 0; j < values[i].Length; j++)
            {
                map.Values[i, j] = values[i][j];
            }
        }

        return map;
    }

    private static string ObjectiveToString(OptimizationObjective objective)
    {
        return objective switch
        {
            OptimizationObjective.AfrAccuracy => "afr_accuracy",
            OptimizationObjective.FuelEconomy => "fuel_economy",
            OptimizationObjective.PowerOutput => "power_output",
            OptimizationObjective.Emissions => "emissions",
            OptimizationObjective.Balanced => "balanced",
            _ => "afr_accuracy"
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

    // JSON models for parsing Python output
    private class PythonOptimizerResults
    {
        public ImprovementReport? improvement_report { get; set; }
        public List<HistoryEntry>? evolution_history { get; set; }
        public Dictionary<string, double>? final_objectives { get; set; }
    }

    private class ImprovementReport
    {
        public int total_cells { get; set; }
        public int cells_changed { get; set; }
        public double max_increase { get; set; }
        public double max_decrease { get; set; }
        public double mean_change { get; set; }
        public double std_change { get; set; }
        public double fitness_improvement { get; set; }
        public double final_afr_error { get; set; }
        public int generations_run { get; set; }
    }

    private class HistoryEntry
    {
        public int generation { get; set; }
        public double best_fitness { get; set; }
        public double mean_fitness { get; set; }
        public double best_afr_error { get; set; }
    }
}
