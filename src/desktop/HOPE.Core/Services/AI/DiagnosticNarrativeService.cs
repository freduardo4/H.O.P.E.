using System.Text;
using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// "Lead Mechanic" Mode service that provides plain-English explanations 
/// of complex ECU data, DTCs, and tuning changes.
/// </summary>
public class DiagnosticNarrativeService
{
    private readonly ILlmService _llmService;

    public DiagnosticNarrativeService(ILlmService llmService)
    {
        _llmService = llmService;
    }

    /// <summary>
    /// Generates a "Lead Mechanic" explanation for a set of DTCs and vehicle state.
    /// </summary>
    public async Task<string> GetPlainEnglishDiagnosisAsync(
        IEnumerable<string> dtcCodes, 
        VehicleContext context,
        CancellationToken ct = default)
    {
        var prompt = new StringBuilder();
        prompt.AppendLine("You are a master diagnostic technician with 30 years of experience. Your goal is to explain the vehicle's issues in plain English to a non-technical car owner.");
        prompt.AppendLine($"Vehicle: {context.Year} {context.Make} {context.Model}");
        prompt.AppendLine($"Current State: RPM {context.CurrentRPM}, Load {context.EngineLoad}%, Speed {context.VehicleSpeed}km/h");
        prompt.AppendLine("The following diagnostic trouble codes were detected:");
        
        foreach (var code in dtcCodes)
        {
            prompt.AppendLine($"- {code}");
        }

        prompt.AppendLine("\nPlease provide:");
        prompt.AppendLine("1. A simple, reassuring summary of what is happening.");
        prompt.AppendLine("2. The likely 'root cause' in everyday terms.");
        prompt.AppendLine("3. Whether it's safe to drive or requires immediate attention.");
        prompt.AppendLine("4. A 'Lead Mechanic's Tip' for the repair.");

        return await _llmService.GenerateAsync(prompt.ToString(), ct);
    }

    /// <summary>
    /// Explains a tuning map change in plain English.
    /// </summary>
    public async Task<string> ExplainTuningChangeAsync(
        string mapName, 
        double averageChange, 
        string reason,
        CancellationToken ct = default)
    {
        var prompt = $"Explain this ECU tuning change to an enthusiast: We modified the '{mapName}' table by an average of {averageChange:P1}. The primary goal was: {reason}. Describe how the car will feel to drive differently and any safety considerations.";
        
        return await _llmService.GenerateAsync(prompt, ct);
    }
}
