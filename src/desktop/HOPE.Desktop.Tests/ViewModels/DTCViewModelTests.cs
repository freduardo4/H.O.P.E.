using HOPE.Core.Models;
using HOPE.Core.Services.Community;
using HOPE.Core.Services.OBD;
using HOPE.Desktop.ViewModels;
using Moq;
using Xunit;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;
using Prism.Regions;
using Microsoft.Extensions.Logging;

namespace HOPE.Desktop.Tests.ViewModels;

public class DTCViewModelTests
{
    private readonly Mock<IOBD2Service> _mockObdService;
    private readonly Mock<IWikiFixService> _mockWikiFixService;
    private readonly Mock<IRegionManager> _mockRegionManager;
    private readonly Mock<ILogger<DTCViewModel>> _mockLogger;
    private readonly DTCViewModel _viewModel;

    public DTCViewModelTests()
    {
        _mockObdService = new Mock<IOBD2Service>();
        _mockWikiFixService = new Mock<IWikiFixService>();
        _mockRegionManager = new Mock<IRegionManager>();
        _mockLogger = new Mock<ILogger<DTCViewModel>>();
        
        _viewModel = new DTCViewModel(_mockObdService.Object, _mockRegionManager.Object, _mockWikiFixService.Object);
    }

    [Fact]
    public void Constructor_InitializesDiagnosticCodesView()
    {
        // Assert
        Assert.NotNull(_viewModel.DiagnosticCodesView);
        Assert.Equal(_viewModel.DiagnosticCodes, _viewModel.DiagnosticCodesView.SourceCollection);
    }

    [Fact]
    public void FilterDTCs_WithEmptySearch_ReturnsTrue()
    {
        // Arrange
        _viewModel.SearchText = string.Empty;
        var item = new DTCItem { Code = "P0101", Description = "MAF Sensor" };

        // Act - Invoke filter manually (or check via view if items were added) or trust view behavior
        // Since Filter is private delegate, we verify effect on View.
        _viewModel.DiagnosticCodes.Add(item);
        
        // Assert
        Assert.Contains(item, _viewModel.DiagnosticCodesView.Cast<DTCItem>());
    }

    [Fact]
    public void FilterDTCs_WithMatchingSearch_ShowsItem()
    {
        // Arrange
        _viewModel.SearchText = "P0101";
        var item = new DTCItem { Code = "P0101", Description = "MAF Sensor" };
        var item2 = new DTCItem { Code = "P0300", Description = "Misfire" };
        
        _viewModel.DiagnosticCodes.Add(item);
        _viewModel.DiagnosticCodes.Add(item2);

        // Act
        // Refresh is called by OnSearchTextChanged, but we set it before adding items.
        // View updates automatically on collection change usually, but filtering is applied.
        
        // Assert
        var viewItems = _viewModel.DiagnosticCodesView.Cast<DTCItem>().ToList();
        Assert.Contains(item, viewItems);
        Assert.DoesNotContain(item2, viewItems);
    }

    [Fact]
    public void FilterDTCs_WithDescriptionSearch_ShowsItem()
    {
        // Arrange
        _viewModel.SearchText = "Misfire";
        var item = new DTCItem { Code = "P0101", Description = "MAF Sensor" };
        var item2 = new DTCItem { Code = "P0300", Description = "Random Misfire" };
        
        _viewModel.DiagnosticCodes.Add(item);
        _viewModel.DiagnosticCodes.Add(item2);

        // Assert
        var viewItems = _viewModel.DiagnosticCodesView.Cast<DTCItem>().ToList();
        Assert.DoesNotContain(item, viewItems);
        Assert.Contains(item2, viewItems);
    }

    [Fact]
    public void SearchText_Change_RefreshesView()
    {
        // Arrange
        var item = new DTCItem { Code = "P0101", Description = "MAF" };
        var item2 = new DTCItem { Code = "P0300", Description = "Misfire" };
        _viewModel.DiagnosticCodes.Add(item);
        _viewModel.DiagnosticCodes.Add(item2);
        
        // Act
        _viewModel.SearchText = "P0300";

        // Assert
        var viewItems = _viewModel.DiagnosticCodesView.Cast<DTCItem>().ToList();
        Assert.Single(viewItems);
        Assert.Equal("P0300", viewItems[0].Code);
        
        // Act 2 - Clear search
        _viewModel.SearchText = "";
        
        // Assert 2
        Assert.Equal(2, _viewModel.DiagnosticCodesView.Cast<DTCItem>().Count());
    }

    [Fact]
    public async Task ReadDTCsAsync_PopulatesCollection()
    {
        // Arrange
        var dtcs = new List<DiagnosticTroubleCode> 
        { 
            new DiagnosticTroubleCode { Code = "P0101" },
            new DiagnosticTroubleCode { Code = "P0300" }
        };
        _mockObdService.Setup(s => s.IsConnected).Returns(true);
        _mockObdService.Setup(s => s.ReadDTCsAsync(It.IsAny<CancellationToken>())).ReturnsAsync(dtcs);

        // Act
        await _viewModel.ReadDTCsCommand.ExecuteAsync(null);

        // Assert
        Assert.Equal(2, _viewModel.DiagnosticCodes.Count);
        Assert.Equal(2, _viewModel.DiagnosticCodesView.Cast<DTCItem>().Count());
    }

    [Fact]
    public void ViewCommunityFixCommand_NavigatesToWikiFixView()
    {
        // Arrange
        var item = new DTCItem { Code = "P0101" };
        
        // Act
        _viewModel.ViewCommunityFixCommand.Execute(item);
        
        // Assert
        _mockRegionManager.Verify(rm => rm.RequestNavigate(
            "MainRegion", 
            "WikiFixView", 
            It.Is<NavigationParameters>(p => p.GetValue<string>("dtc") == "P0101")
        ), Times.Once);
    }
}
