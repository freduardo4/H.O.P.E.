using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.AI;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Services.AI;

public class DiagnosticNarrativeServiceTests
{
    private readonly Mock<ILlmService> _llmMock;
    private readonly DiagnosticNarrativeService _service;

    public DiagnosticNarrativeServiceTests()
    {
        _llmMock = new Mock<ILlmService>();
        _service = new DiagnosticNarrativeService(_llmMock.Object);
    }

    [Fact]
    public async Task GetPlainEnglishDiagnosisAsync_CallsLLMWithProperPrompt()
    {
        // Arrange
        var dtcs = new[] { "P0300", "P0171" };
        var context = new VehicleContext 
        { 
            Make = "Toyota", 
            Model = "Supra", 
            Year = 2020, 
            CurrentRPM = 2500, 
            EngineLoad = 45.0, 
            VehicleSpeed = 80.0 
        };

        _llmMock.Setup(l => l.GenerateAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Detailed diagnosis from AI");

        // Act
        var result = await _service.GetPlainEnglishDiagnosisAsync(dtcs, context);

        // Assert
        Assert.Equal("Detailed diagnosis from AI", result);
        _llmMock.Verify(l => l.GenerateAsync(It.Is<string>(p => 
            p.Contains("Toyota Supra") && 
            p.Contains("P0300") && 
            p.Contains("master diagnostic technician")), It.IsAny<CancellationToken>()), Times.Once);
    }

    [Fact]
    public async Task ExplainTuningChangeAsync_CallsLLMWithTuningPrompt()
    {
        // Arrange
        _llmMock.Setup(l => l.GenerateAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync("Enthusiast-level explanation");

        // Act
        var result = await _service.ExplainTuningChangeAsync("Ignition Timing", 0.05, "Improve throttle response");

        // Assert
        Assert.Equal("Enthusiast-level explanation", result);
        _llmMock.Verify(l => l.GenerateAsync(It.Is<string>(p => 
            p.Contains("Ignition Timing") && 
            p.Contains("5.0%") && 
            p.Contains("enthusiast")), It.IsAny<CancellationToken>()), Times.Once);
    }
}
