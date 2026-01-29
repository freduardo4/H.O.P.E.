using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

public class MockLlmService : ILlmService
{
    public Task<string> TranslateDtcAsync(string code, string originalDescription)
    {
        string explanation = code switch
        {
            "P0300" => "Random or Multiple Cylinder Misfire Detected. This means the engine is not burning fuel correctly in one or more cylinders. Common causes include worn spark plugs, faulty ignition coils, or a vacuum leak.",
            "P0171" => "System Too Lean (Bank 1). The engine is getting too much air or too little fuel. This can cause poor performance or engine damage if ignored. Check for vacuum leaks or a dirty MAF sensor.",
            "P0420" => "Catalyst System Efficiency Below Threshold. This usually indicates the catalytic converter is failing, or there is an issue with the rear O2 sensor or an exhaust leak.",
            _ => $"{originalDescription}. This vehicle code indicates a system malfunction that requires attention to prevent further damage."
        };

        return Task.FromResult(explanation);
    }

    public Task<string> SummarizeSessionAsync(DiagnosticSession session)
    {
        var summary = $"## Session Summary\n\n" +
                      $"Diagnostic session completed for vehicle {session.VehicleId}.\n" +
                      $"- **DTCs Found**: {session.DTCs.Count}\n" +
                      $"- **Anomalies Detected**: {session.AIInsights.Count}\n\n" +
                      $"The vehicle shows signs of performance degradation. Recommend checking the ignition system based on observed misfire patterns.";

        return Task.FromResult(summary);
    }

    public Task<string> GenerateAsync(string prompt, CancellationToken ct = default)
    {
        string response = "Based on the provided telemetry, the engine is exhibiting sporadic ignition timing retard under high load, potentially due to poor fuel quality or a failing knock sensor.";
        return Task.FromResult(response);
    }
}
