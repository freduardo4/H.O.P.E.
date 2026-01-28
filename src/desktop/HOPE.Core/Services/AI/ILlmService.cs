using HOPE.Core.Models;

namespace HOPE.Core.Services.AI;

/// <summary>
/// Service for interacting with Large Language Models for diagnostic explanations.
/// </summary>
public interface ILlmService
{
    /// <summary>
    /// Translates a technical DTC into a human-friendly explanation.
    /// </summary>
    /// <param name="code">DTC Code (e.g. P0300)</param>
    /// <param name="originalDescription">Technical description</param>
    /// <returns>Plain-english explanation and suggested checks</returns>
    Task<string> TranslateDtcAsync(string code, string originalDescription);

    /// <summary>
    /// Summarizes a diagnostic session into a concise report.
    /// </summary>
    /// <param name="session">The session to summarize</param>
    /// <returns>Markdown-formatted summary</returns>
    Task<string> SummarizeSessionAsync(DiagnosticSession session);
}
