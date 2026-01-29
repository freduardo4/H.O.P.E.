using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using HOPE.Core.Models; // For any shared models if needed, though we defined specific ones in Interface
using Microsoft.Extensions.Logging;

namespace HOPE.Core.Services.AI;

public class TuningCopilotService : ITuningCopilotService, IDisposable
{
    private readonly ILogger<TuningCopilotService> _logger;
    private Process? _ragProcess;
    private readonly SemaphoreSlim _lock = new(1, 1);
    private bool _isReady;

    public bool IsAvailable => _isReady && _ragProcess != null && !_ragProcess.HasExited;

    public TuningCopilotService(ILogger<TuningCopilotService> logger)
    {
        _logger = logger;
        InitializeBackend();
    }

    private void InitializeBackend()
    {
        try
        {
            var projectRoot = FindProjectRoot();
            var scriptPath = Path.Combine(projectRoot, "src", "ai-training", "hope_ai", "rag_server.py");
            var pythonPath = "python"; // Assumes in PATH or venv active environment

            // If running from VS, CWD might be bin/debug. Detailed finding might be needed.
            // For now, assuming standard layout.
            
            if (!File.Exists(scriptPath))
            {
                 _logger.LogWarning($"RAG Server script not found at {scriptPath}");
                 return;
            }

            var startInfo = new ProcessStartInfo
            {
                FileName = pythonPath,
                Arguments = scriptPath,
                UseShellExecute = false,
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
                WorkingDirectory = projectRoot // Important for relative paths in python script
            };
            
            // Set PYTHONPATH
            startInfo.EnvironmentVariables["PYTHONPATH"] = Path.Combine(projectRoot, "src", "ai-training");

            _ragProcess = new Process { StartInfo = startInfo };
            _ragProcess.OutputDataReceived += (sender, e) => 
            {
                if (!string.IsNullOrEmpty(e.Data) && e.Data == "READY")
                {
                    _isReady = true;
                    _logger.LogInformation("RAG Backend Ready.");
                }
            };
            _ragProcess.ErrorDataReceived += (sender, e) =>
            {
                if (!string.IsNullOrEmpty(e.Data)) _logger.LogWarning($"RAG stderr: {e.Data}");
            };

            _ragProcess.Start();
            _ragProcess.BeginOutputReadLine();
            _ragProcess.BeginErrorReadLine();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to start RAG backend.");
        }
    }

    private string FindProjectRoot()
    {
        // Simple heuristic: go up until we find src folder or .git
        var current = AppDomain.CurrentDomain.BaseDirectory;
        while (current != null)
        {
            if (Directory.Exists(Path.Combine(current, "src")) && Directory.Exists(Path.Combine(current, "docs")))
                return current;
            current = Directory.GetParent(current)?.FullName;
        }
        return @"C:\Users\Test\Documents\H.O.P.E"; // Fallback for this environment
    }

    public async Task<CopilotResponse> AskAsync(string query, CancellationToken ct = default)
    {
        if (!IsAvailable)
        {
            return new CopilotResponse { Answer = "I'm sorry, my knowledge base is currently offline." };
        }

        await _lock.WaitAsync(ct);
        try
        {
            // We need to temporarily stop async read to read synchronously or implement a request/response queue.
            // Since Process doesn't support mixing async/sync read easily, we should use a request queue mechanism
            // or event-based completion.
            
            // Simplified approach: For this turn, since we set up BeginOutputReadLine, 
            // we need to capture the next line that is JSON.
            
            var tcs = new TaskCompletionSource<string>();
            
            DataReceivedEventHandler handler = null;
            handler = (s, e) => 
            {
                if (string.IsNullOrEmpty(e.Data)) return;
                // Ignore log lines if any slip through, assume JSON starts with {
                if (e.Data.TrimStart().StartsWith("{")) 
                {
                    tcs.TrySetResult(e.Data);
                }
            };

            _ragProcess!.OutputDataReceived += handler;
            
            try 
            {
                 await _ragProcess.StandardInput.WriteLineAsync(query);
                 
                 // Wait for response with timeout
                 var responseJson = await tcs.Task.WaitAsync(TimeSpan.FromSeconds(5), ct);
                 return ParseResponse(responseJson);
            }
            finally
            {
                _ragProcess.OutputDataReceived -= handler;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error querying RAG backend");
            return new CopilotResponse { Answer = "I encountered an error retrieving information." };
        }
        finally
        {
            _lock.Release();
        }
    }

    private CopilotResponse ParseResponse(string json)
    {
        try 
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            
            var response = new CopilotResponse();
            
            if (root.TryGetProperty("error", out var error))
            {
                response.Answer = $"Backend Error: {error.GetString()}";
                return response;
            }

            if (root.TryGetProperty("results", out var results))
            {
                 var sb = new StringBuilder();
                 sb.AppendLine("Here is what I found in the documentation:\n");

                 foreach (var result in results.EnumerateArray())
                 {
                     var file = result.GetProperty("file").GetString();
                     var snippet = result.GetProperty("content").GetString();
                     var score = result.GetProperty("score").GetDouble();
                     
                     var fileName = Path.GetFileName(file);
                     
                     response.Sources.Add(new CopilotContextSource
                     {
                         FilePath = file,
                         Relevance = score,
                         Snippet = snippet
                     });

                     sb.AppendLine($"**File: {fileName}** (Relevance: {score:F2})");
                     sb.AppendLine($"> {snippet.Replace("\n", " ").Trim()}...\n");
                 }
                 
                 if (response.Sources.Count == 0)
                     response.Answer = "I couldn't find any relevant information in the documentation.";
                 else
                     response.Answer = sb.ToString();
            }
            
            return response;
        }
        catch
        {
            return new CopilotResponse { Answer = "Failed to parse backend response." };
        }
    }

    public void Dispose()
    {
        try 
        {
            if (_ragProcess != null && !_ragProcess.HasExited)
            {
                _ragProcess.StandardInput.WriteLine("EXIT");
                _ragProcess.WaitForExit(1000);
                _ragProcess.Kill(); // Ensure dead
                _ragProcess.Dispose();
            }
        } 
        catch { }
    }
}
