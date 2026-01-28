using HOPE.Core.Models;
using HOPE.Core.Services.AI;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;
using Colors = QuestPDF.Helpers.Colors;

namespace HOPE.Core.Services.Reports;

public class GenerativeReportService : IReportService
{
    private readonly ILlmService _llmService;

    public GenerativeReportService(ILlmService llmService)
    {
        _llmService = llmService;
        QuestPDF.Settings.License = LicenseType.Community;
    }

    public async Task<string> GenerateVehicleHealthReportAsync(DiagnosticSession session, string outputPath)
    {
        // Enforce file extension
        if (!outputPath.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase))
            outputPath += ".pdf";

        // Get AI explanations for DTCs
        var dtcExplanations = new Dictionary<string, string>();
        foreach (var dtc in session.DTCs)
        {
            dtcExplanations[dtc.Code] = await _llmService.TranslateDtcAsync(dtc.Code, dtc.Description);
        }

        var aiSummary = await _llmService.SummarizeSessionAsync(session);

        Document.Create(container =>
        {
            container.Page(page =>
            {
                page.Size(PageSizes.A4);
                page.Margin(1, Unit.Centimetre);
                page.PageColor(Colors.White);
                page.DefaultTextStyle(x => x.FontSize(10).FontFamily("Arial"));

                page.Header().Row(row =>
                {
                    row.RelativeItem().Column(col =>
                    {
                        col.Item().Text("H.O.P.E. VEHICLE HEALTH REPORT").FontSize(24).ExtraBold().FontColor(Colors.Green.Medium);
                        col.Item().Text($"Session ID: {session.Id:N}").FontSize(10).FontColor(Colors.Grey.Medium);
                    });

                    row.ConstantItem(100).AlignRight().Column(col =>
                    {
                        col.Item().AlignRight().Text(DateTime.Now.ToString("yyyy-MM-dd"));
                        col.Item().AlignRight().Text(DateTime.Now.ToString("HH:mm"));
                    });
                });

                page.Content().PaddingVertical(1, Unit.Centimetre).Column(x =>
                {
                    x.Spacing(20);

                    // Vehicle Info Section
                    x.Item().Table(table =>
                    {
                        table.ColumnsDefinition(columns =>
                        {
                            columns.RelativeColumn();
                            columns.RelativeColumn();
                        });

                        table.Cell().Element(CellStyle).Text("VEHICLE INFO").Bold();
                        table.Cell().Element(CellStyle).Text($"Vehicle ID: {session.VehicleId}");
                    });

                    // AI Summary Section
                    x.Item().Background(Colors.Grey.Lighten4).Padding(10).Column(col =>
                    {
                        col.Spacing(5);
                        col.Item().Text("AI DIAGNOSTIC INSIGHT").Bold().FontSize(12);
                        col.Item().Text(aiSummary);
                    });

                    // DTC Section
                    if (session.DTCs.Any())
                    {
                        x.Item().Column(col =>
                        {
                            col.Spacing(10);
                            col.Item().Text("DIAGNOSTIC TROUBLE CODES").Bold().FontSize(14).FontColor(Colors.Red.Medium);
                            
                            foreach (var dtc in session.DTCs)
                            {
                                col.Item().BorderBottom(1).BorderColor(Colors.Grey.Lighten2).PaddingVertical(5).Row(row =>
                                {
                                    row.ConstantItem(60).Text(dtc.Code).Bold().FontColor(Colors.Red.Medium);
                                    row.RelativeItem().Column(c =>
                                    {
                                        c.Item().Text(dtc.Description).Italic();
                                        if (dtcExplanations.TryGetValue(dtc.Code, out var explanation))
                                        {
                                            c.Item().PaddingTop(2).Text(explanation).FontSize(9).FontColor(Colors.Grey.Darken2);
                                        }
                                    });
                                });
                            }
                        });
                    }
                    else
                    {
                        x.Item().Text("No Diagnostic Trouble Codes detected.").Italic().FontColor(Colors.Green.Medium);
                    }

                    // AI Insights (Anomalies)
                    if (session.AIInsights.Any())
                    {
                        x.Item().Column(col =>
                        {
                            col.Spacing(10);
                            col.Item().Text("AI ANOMALY DETECTION").Bold().FontSize(14);
                            
                            foreach (var insight in session.AIInsights)
                            {
                                col.Item().Border(1).BorderColor(Colors.Grey.Lighten3).Padding(8).Column(c =>
                                {
                                    c.Item().Row(r =>
                                    {
                                        r.RelativeItem().Text(insight.Description).Bold();
                                        r.ConstantItem(60).AlignRight().Text($"{insight.Confidence:P0} Conf.").FontSize(8);
                                    });
                                    if (!string.IsNullOrEmpty(insight.RecommendedAction))
                                    {
                                        c.Item().PaddingTop(5).Text($"Recommendation: {insight.RecommendedAction}").FontSize(9).FontColor(Colors.Blue.Medium);
                                    }
                                });
                            }
                        });
                    }
                });

                page.Footer().AlignCenter().Text(x =>
                {
                    x.Span("Page ");
                    x.CurrentPageNumber();
                });
            });
        })
        .GeneratePdf(outputPath);

        return outputPath;

        static IContainer CellStyle(IContainer container)
        {
            return container.BorderBottom(1).BorderColor(Colors.Grey.Lighten2).PaddingVertical(5);
        }
    }
}
