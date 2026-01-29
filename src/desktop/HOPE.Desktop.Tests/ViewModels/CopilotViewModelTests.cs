using Xunit;
using Moq;
using HOPE.Desktop.ViewModels;
using HOPE.Core.Services.AI;
using Prism.Regions;
using System.Threading.Tasks;
using System.Linq;
using System;
using System.Threading;

namespace HOPE.Desktop.Tests.ViewModels;

public class CopilotViewModelTests
{
    private readonly Mock<ITuningCopilotService> _mockCopilotService;
    private readonly Mock<IRegionManager> _mockRegionManager;

    public CopilotViewModelTests()
    {
        _mockCopilotService = new Mock<ITuningCopilotService>();
        _mockRegionManager = new Mock<IRegionManager>();
    }

    [Fact]
    public void Constructor_ShouldAddInitialGreeting()
    {
        // Act
        var viewModel = new CopilotViewModel(_mockCopilotService.Object, _mockRegionManager.Object);

        // Assert
        Assert.Single(viewModel.ChatHistory);
        Assert.False(viewModel.ChatHistory[0].IsUser);
        Assert.Contains("Tuning Copilot", viewModel.ChatHistory[0].Message);
    }

    [Fact]
    public async Task SendQueryCommand_ShouldAddUserMessageAndResponse()
    {
        // Arrange
        var query = "How to fix P0300?";
        var answer = "Check spark plugs and coils.";
        _mockCopilotService.Setup(s => s.AskAsync(query, It.IsAny<CancellationToken>()))
            .ReturnsAsync(new CopilotResponse { Answer = answer });

        var viewModel = new CopilotViewModel(_mockCopilotService.Object, _mockRegionManager.Object);
        viewModel.CurrentQuery = query;

        // Act
        viewModel.SendQueryCommand.Execute();
        
        // Wait for async execution in DelegateCommand
        await Task.Delay(100); 

        // Assert
        Assert.Equal(3, viewModel.ChatHistory.Count); // Greeting + User + Bot
        Assert.Equal(query, viewModel.ChatHistory[1].Message);
        Assert.True(viewModel.ChatHistory[1].IsUser);
        Assert.Equal(answer, viewModel.ChatHistory[2].Message);
        Assert.False(viewModel.ChatHistory[2].IsUser);
        Assert.Empty(viewModel.CurrentQuery);
    }

    [Fact]
    public async Task SendQueryCommand_ShouldHandleErrorGracefully()
    {
        // Arrange
        var query = "Trigger error";
        _mockCopilotService.Setup(s => s.AskAsync(query, It.IsAny<CancellationToken>()))
            .ThrowsAsync(new Exception("Network error"));

        var viewModel = new CopilotViewModel(_mockCopilotService.Object, _mockRegionManager.Object);
        viewModel.CurrentQuery = query;

        // Act
        viewModel.SendQueryCommand.Execute();
        await Task.Delay(100);

        // Assert
        Assert.Equal(3, viewModel.ChatHistory.Count);
        Assert.Contains("Error: Network error", viewModel.ChatHistory[2].Message);
        Assert.False(viewModel.IsBusy);
    }

    [Fact]
    public void SendQueryCommand_ShouldNotExecute_WhenQueryIsWhitespace()
    {
        // Arrange
        var viewModel = new CopilotViewModel(_mockCopilotService.Object, _mockRegionManager.Object);
        viewModel.CurrentQuery = "   ";

        // Act
        viewModel.SendQueryCommand.Execute();

        // Assert
        Assert.Single(viewModel.ChatHistory); // Only greeting
        _mockCopilotService.Verify(s => s.AskAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()), Times.Never);
    }
}
