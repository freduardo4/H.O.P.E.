using System.Net.Http;
using System.Text;
using System.Text.Json;
using HOPE.Core.Services.ECU;

namespace HOPE.Core.Services.Safety;

public class CloudSafetyService
{
    private readonly HttpClient _httpClient;
    private const string BASE_URL = "http://localhost:3000/safety"; // TODO: Move to config

    public CloudSafetyService(HttpClient httpClient)
    {
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
    }

    /// <summary>
    /// Validates if a flash operation is allowed by cloud policy
    /// </summary>
    public virtual async Task<bool> ValidateFlashOperationAsync(string ecuId, double voltage, CancellationToken ct = default)
    {
        try
        {
            var payload = new
            {
                ecuId,
                voltage,
                timestamp = DateTime.UtcNow
            };

            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            var response = await _httpClient.PostAsync($"{BASE_URL}/validate", content, ct);
            
            if (!response.IsSuccessStatusCode)
            {
                return false; // Fail safe if cloud rejects or errors
            }

            var responseJson = await response.Content.ReadAsStringAsync(ct);
            using var doc = JsonDocument.Parse(responseJson);
            return doc.RootElement.GetProperty("allowed").GetBoolean();
        }
        catch
        {
            // If cloud is unreachable, what is the policy?
            // For now, let's assume we REQUIRE cloud validation for "Connected" features, 
            // but maybe we allow offline if it's a critical fix?
            // Sticking to "Fail Safe" -> return false if error.
            return false; 
        }
    }

    /// <summary>
    /// Logs a safety-critical event to the cloud
    /// </summary>
    public virtual async Task LogSafetyEventAsync(SafetyEvent evt, CancellationToken ct = default)
    {
        try
        {
            var json = JsonSerializer.Serialize(evt);
            var content = new StringContent(json, Encoding.UTF8, "application/json");

            // Fire and forget-ish, or await? Await to ensure we at least try.
            await _httpClient.PostAsync($"{BASE_URL}/telemetry", content, ct);
        }
        catch
        {
            // Logging failure shouldn't crash the app, but good to know
        }
    }
}

public class SafetyEvent
{
    public string EventType { get; set; } = string.Empty;
    public string EcuId { get; set; } = string.Empty;
    public double? Voltage { get; set; }
    public string Message { get; set; } = string.Empty;
    public string? Metadata { get; set; }
    public DateTime Timestamp { get; set; }
    public bool Success { get; set; }
}
