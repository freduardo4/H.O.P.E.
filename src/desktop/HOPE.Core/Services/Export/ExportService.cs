using System.IO;
using System.Text;
using HOPE.Core.Models;
using HOPE.Core.Services.Database;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;

namespace HOPE.Core.Services.Export;

/// <summary>
/// Service for exporting session data to CSV and PDF formats
/// </summary>
public class ExportService : IExportService
{
    private readonly IDatabaseService _dbService;
    
    static ExportService()
    {
        // Configure QuestPDF license (community edition)
        QuestPDF.Settings.License = LicenseType.Community;
    }

    public ExportService(IDatabaseService dbService)
    {
        _dbService = dbService;
    }

    public string GetDefaultExportDirectory()
    {
        var documentsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
        var exportPath = Path.Combine(documentsPath, "HOPE", "Exports");
        Directory.CreateDirectory(exportPath);
        return exportPath;
    }

    public async Task ExportToCsvAsync(Guid sessionId, string outputPath)
    {
        var readings = await _dbService.GetSessionDataAsync(sessionId);
        var sessions = await _dbService.GetSessionsAsync();
        var session = sessions.FirstOrDefault(s => s.Id == sessionId);
        
        var sb = new StringBuilder();
        
        // Header
        sb.AppendLine("HOPE Session Export");
        sb.AppendLine($"Session ID,{sessionId}");
        sb.AppendLine($"Start Time,{session?.StartTime:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"End Time,{session?.EndTime:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine($"Total Readings,{readings.Count}");
        sb.AppendLine();
        
        // Data header
        sb.AppendLine("Timestamp,PID,Name,Value,Unit,RawResponse");
        
        // Data rows
        foreach (var reading in readings.OrderBy(r => r.Timestamp))
        {
            var timestamp = reading.Timestamp.ToString("yyyy-MM-dd HH:mm:ss.fff");
            var rawResponse = reading.RawResponse?.Replace(",", ";") ?? "";
            sb.AppendLine($"{timestamp},{reading.PID},{reading.Name},{reading.Value:F2},{reading.Unit},{rawResponse}");
        }
        
        await File.WriteAllTextAsync(outputPath, sb.ToString(), Encoding.UTF8);
    }

    public async Task ExportToPdfAsync(Guid sessionId, string outputPath)
    {
        var readings = await _dbService.GetSessionDataAsync(sessionId);
        var sessions = await _dbService.GetSessionsAsync();
        var session = sessions.FirstOrDefault(s => s.Id == sessionId);
        
        // Group readings by PID for statistics
        var groupedReadings = readings
            .GroupBy(r => r.PID)
            .Select(g => new
            {
                PID = g.Key,
                Name = g.First().Name,
                Unit = g.First().Unit,
                Count = g.Count(),
                Min = g.Min(r => r.Value),
                Max = g.Max(r => r.Value),
                Avg = g.Average(r => r.Value)
            })
            .ToList();

        var document = Document.Create(container =>
        {
            container.Page(page =>
            {
                page.Size(PageSizes.A4);
                page.Margin(40);
                page.DefaultTextStyle(x => x.FontSize(10));

                page.Header()
                    .Column(column =>
                    {
                        column.Item().Text("HOPE Diagnostic Report")
                            .FontSize(24).Bold().FontColor(Colors.Blue.Darken2);
                        column.Item().Text($"Session: {sessionId}")
                            .FontSize(10).FontColor(Colors.Grey.Darken1);
                    });

                page.Content()
                    .PaddingVertical(20)
                    .Column(column =>
                    {
                        // Session Info
                        column.Item().Text("Session Information").FontSize(14).Bold();
                        column.Item().PaddingBottom(10).Table(table =>
                        {
                            table.ColumnsDefinition(columns =>
                            {
                                columns.ConstantColumn(120);
                                columns.RelativeColumn();
                            });

                            table.Cell().Text("Start Time:").Bold();
                            table.Cell().Text(session?.StartTime.ToString("yyyy-MM-dd HH:mm:ss") ?? "N/A");
                            
                            table.Cell().Text("End Time:").Bold();
                            table.Cell().Text(session?.EndTime?.ToString("yyyy-MM-dd HH:mm:ss") ?? "Ongoing");
                            
                            table.Cell().Text("Total Readings:").Bold();
                            table.Cell().Text(readings.Count.ToString());
                        });

                        column.Item().PaddingTop(15);
                        
                        // Statistics Table
                        column.Item().Text("Parameter Statistics").FontSize(14).Bold();
                        column.Item().PaddingBottom(10).Table(table =>
                        {
                            table.ColumnsDefinition(columns =>
                            {
                                columns.RelativeColumn(2);  // Name
                                columns.RelativeColumn(1);  // Unit
                                columns.RelativeColumn(1);  // Count
                                columns.RelativeColumn(1);  // Min
                                columns.RelativeColumn(1);  // Max
                                columns.RelativeColumn(1);  // Avg
                            });

                            // Header
                            table.Header(header =>
                            {
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Parameter").FontColor(Colors.White).Bold();
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Unit").FontColor(Colors.White).Bold();
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Count").FontColor(Colors.White).Bold();
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Min").FontColor(Colors.White).Bold();
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Max").FontColor(Colors.White).Bold();
                                header.Cell().Background(Colors.Blue.Darken2).Padding(5)
                                    .Text("Avg").FontColor(Colors.White).Bold();
                            });

                            // Rows
                            foreach (var stat in groupedReadings)
                            {
                                var bgColor = groupedReadings.IndexOf(stat) % 2 == 0 
                                    ? Colors.White 
                                    : Colors.Grey.Lighten4;
                                    
                                table.Cell().Background(bgColor).Padding(5).Text(stat.Name);
                                table.Cell().Background(bgColor).Padding(5).Text(stat.Unit);
                                table.Cell().Background(bgColor).Padding(5).Text(stat.Count.ToString());
                                table.Cell().Background(bgColor).Padding(5).Text($"{stat.Min:F1}");
                                table.Cell().Background(bgColor).Padding(5).Text($"{stat.Max:F1}");
                                table.Cell().Background(bgColor).Padding(5).Text($"{stat.Avg:F1}");
                            }
                        });
                    });

                page.Footer()
                    .AlignCenter()
                    .Text(x =>
                    {
                        x.Span("Generated by HOPE - High-Output Performance Engineering | ");
                        x.Span(DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"));
                    });
            });
        });

        document.GeneratePdf(outputPath);
    }
}
