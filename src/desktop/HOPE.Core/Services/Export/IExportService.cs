namespace HOPE.Core.Services.Export;

/// <summary>
/// Interface for exporting session data in various formats
/// </summary>
public interface IExportService
{
    /// <summary>
    /// Export session data to CSV format
    /// </summary>
    /// <param name="sessionId">Session to export</param>
    /// <param name="outputPath">Output file path</param>
    Task ExportToCsvAsync(Guid sessionId, string outputPath);
    
    /// <summary>
    /// Export session data to PDF report format
    /// </summary>
    /// <param name="sessionId">Session to export</param>
    /// <param name="outputPath">Output file path</param>
    Task ExportToPdfAsync(Guid sessionId, string outputPath);
    
    /// <summary>
    /// Get the default export directory
    /// </summary>
    string GetDefaultExportDirectory();
}
