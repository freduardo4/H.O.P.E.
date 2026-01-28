using HOPE.Core.Models;
using HOPE.Core.Services.Community;
using HOPE.Desktop.ViewModels;
using Moq;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Xunit;

namespace HOPE.Desktop.Tests.ViewModels;

public class WikiFixViewModelTests
{
    private readonly Mock<IWikiFixService> _mockWikiFixService;
    private readonly WikiFixViewModel _viewModel;

    public WikiFixViewModelTests()
    {
        _mockWikiFixService = new Mock<IWikiFixService>();
        _viewModel = new WikiFixViewModel(_mockWikiFixService.Object);
    }

    [Fact]
    public void Constructor_InitializesWithEmptyCollections()
    {
        Assert.Empty(_viewModel.Posts);
        Assert.Null(_viewModel.SelectedPost);
        Assert.False(_viewModel.IsLoading);
        Assert.Equal(string.Empty, _viewModel.SearchQuery);
        Assert.Equal(string.Empty, _viewModel.DtcFilter);
    }

    [Fact]
    public async Task SearchAsync_LoadsPosts_AndManagesLoadingState()
    {
        // Arrange
        var posts = new List<WikiPost>
        {
            new WikiPost { Id = Guid.NewGuid(), Title = "Misfire Fix" },
            new WikiPost { Id = Guid.NewGuid(), Title = "O2 Sensor Error" }
        };
        
        var tcs = new TaskCompletionSource<IEnumerable<WikiPost>>();
        _mockWikiFixService.Setup(s => s.SearchPostsAsync(It.IsAny<string>(), It.IsAny<string>()))
            .Returns(tcs.Task);

        _viewModel.SearchQuery = "misfire";
        _viewModel.DtcFilter = "P0300";

        // Act
        var searchTask = _viewModel.SearchCommand.ExecuteAsync(null);
        
        // Assert Loading state while task is running
        Assert.True(_viewModel.IsLoading);
        
        // Complete the task
        tcs.SetResult(posts);
        await searchTask;

        // Assert final state
        Assert.False(_viewModel.IsLoading);
        Assert.Equal(2, _viewModel.Posts.Count);
        Assert.Equal("Misfire Fix", _viewModel.Posts[0].Title);
        _mockWikiFixService.Verify(s => s.SearchPostsAsync("misfire", "P0300"), Times.Once);
    }

    [Fact]
    public async Task UpvoteAsync_CallsService_AndUpdatesOptimistically()
    {
        // Arrange
        var post = new WikiPost { Id = Guid.NewGuid(), Title = "Test Post", Upvotes = 10 };
        _mockWikiFixService.Setup(s => s.UpvotePostAsync(post.Id))
            .Returns(Task.CompletedTask);

        // Act
        await _viewModel.UpvoteCommand.ExecuteAsync(post);

        // Assert
        Assert.Equal(11, post.Upvotes);
        _mockWikiFixService.Verify(s => s.UpvotePostAsync(post.Id), Times.Once);
    }

    [Fact]
    public async Task LoadDtcFixesAsync_SetsFilter_AndTriggersSearch()
    {
        // Arrange
        _mockWikiFixService.Setup(s => s.SearchPostsAsync(string.Empty, "P0171"))
            .ReturnsAsync(new List<WikiPost>());

        // Act
        await _viewModel.LoadDtcFixesAsync("P0171");

        // Assert
        Assert.Equal("P0171", _viewModel.DtcFilter);
        Assert.Equal(string.Empty, _viewModel.SearchQuery);
        _mockWikiFixService.Verify(s => s.SearchPostsAsync(string.Empty, "P0171"), Times.Once);
    }
}
