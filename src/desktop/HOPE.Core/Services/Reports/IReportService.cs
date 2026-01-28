using HOPE.Core.Models;

namespace HOPE.Core.Services.Reports;

/// <summary>
/// Service for generating diagnostic and vehicle health reports.
/// </summary>
public interface IReportService
{
    /// <summary>
    /// Generates a professional PDF report for a diagnostic session.
    /// </summary>
    /// <param name="session">The session data to include</param>
    /// <param name="outputPath">Where to save the PDF</param>
    /// <returns>Path to the generated file</returns>
    Task<string> GenerateVehicleHealthReportAsync(DiagnosticSession session, string outputPath);
}
