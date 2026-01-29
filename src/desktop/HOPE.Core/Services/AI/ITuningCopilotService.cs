namespace HOPE.Core.Services.AI;

public interface ITuningCopilotService
{
    /// <summary>
    /// Serves as the primary interaction point for the RAG-based assistant.
    /// </summary>
    /// <param name="query">User's question.</param>
    /// <returns>A response containing relevant context or an answer.</returns>
    Task<CopilotResponse> AskAsync(string query, CancellationToken ct = default);

    /// <summary>
    /// Checks if the backend RAG service is available.
    /// </summary>
    bool IsAvailable { get; }
}

public class CopilotResponse
{
    public string Answer { get; set; } = string.Empty;
    public List<CopilotContextSource> Sources { get; set; } = new();
}

public class CopilotContextSource
{
    public string FilePath { get; set; } = string.Empty;
    public double Relevance { get; set; }
    public string Snippet { get; set; } = string.Empty;
}
