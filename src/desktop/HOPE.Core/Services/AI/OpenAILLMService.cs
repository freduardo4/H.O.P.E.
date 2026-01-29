using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Professional LLM service implementation using OpenAI API (or compatible local LLM).
/// </summary>
public class OpenAILLMService : ILlmService
{
    private readonly HttpClient _httpClient;
    private readonly string _apiKey;
    private readonly string _model = "gpt-4o-mini"; // Curated for performance/cost

    public OpenAILLMService(HttpClient httpClient, string apiKey)
    {
        _httpClient = httpClient;
        _apiKey = apiKey;
    }

    public async Task<string> TranslateDtcAsync(string code, string originalDescription)
    {
        var prompt = $"Explain automotive trouble code {code} ({originalDescription}) to a car owner. " +
                     "Use simple terms, explain why it matters, and what they should expect a technician to check first.";
        
        return await GenerateAsync(prompt);
    }

    public async Task<string> SummarizeSessionAsync(DiagnosticSession session)
    {
        var prompt = $"Summarize this vehicle diagnostic session. DTCs detected: {string.Join(", ", session.DTCs.Select(d => d.Code))}. " +
                     $"There are {session.AIInsights.Count} AI-detected anomalies. " +
                     "Provide a 3-sentence professional summary and a 'Next Steps' recommendation.";
        
        return await GenerateAsync(prompt);
    }

    public async Task<string> GenerateAsync(string prompt, CancellationToken ct = default)
    {
        if (string.IsNullOrEmpty(_apiKey))
            return "LLM Service not configured. Please add an API key in settings.";

        try
        {
            var request = new
            {
                model = _model,
                messages = new[]
                {
                    new { role = "system", content = "You are a professional automotive diagnostic expert." },
                    new { role = "user", content = prompt }
                }
            };

            _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _apiKey);
            
            var response = await _httpClient.PostAsJsonAsync("https://api.openai.com/v1/chat/completions", request, ct);
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            return json.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString() ?? "No response from AI.";
        }
        catch (Exception ex)
        {
            return $"AI Error: {ex.Message}";
        }
    }
}
