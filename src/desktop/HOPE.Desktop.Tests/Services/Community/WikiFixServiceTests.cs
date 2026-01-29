using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.Community;
using HOPE.Core.Services.Logging;
using Moq;
using Moq.Protected;
using Xunit;

namespace HOPE.Desktop.Tests.Services.Community;

public class WikiFixServiceTests
{
    private readonly Mock<HttpMessageHandler> _handlerMock;
    private readonly Mock<ILoggingService> _loggerMock;
    private readonly WikiFixService _service;

    public WikiFixServiceTests()
    {
        _handlerMock = new Mock<HttpMessageHandler>();
        _loggerMock = new Mock<ILoggingService>();
        var httpClient = new HttpClient(_handlerMock.Object) { BaseAddress = new Uri("http://localhost") };
        _service = new WikiFixService(httpClient, _loggerMock.Object);
    }

    [Fact]
    public async Task SearchPostsAsync_ReturnsItemsOnSuccess()
    {
        // Arrange
        var mockResponse = new { items = new[] { new WikiPost { Title = "Test Fix" } }, total = 1 };
        
        _handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.IsAny<HttpRequestMessage>(),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = JsonContent.Create(mockResponse)
            });

        // Act
        var results = await _service.SearchPostsAsync("leak", "P0300");

        // Assert
        Assert.Single(results);
        Assert.Equal("Test Fix", results.First().Title);
    }

    [Fact]
    public async Task SearchKnowledgeAsync_CallsGraphQLEndpoint()
    {
        // Arrange
        var mockResponse = new { data = new Dictionary<string, List<KnowledgeNode>> { 
            ["searchKnowledge"] = new List<KnowledgeNode> { new KnowledgeNode { Name = "Intake Manifold" } } 
        } };

        _handlerMock.Protected()
            .Setup<Task<HttpResponseMessage>>(
                "SendAsync",
                ItExpr.Is<HttpRequestMessage>(req => req.RequestUri.AbsolutePath == "/graphql"),
                ItExpr.IsAny<CancellationToken>())
            .ReturnsAsync(new HttpResponseMessage
            {
                StatusCode = HttpStatusCode.OK,
                Content = JsonContent.Create(mockResponse)
            });

        // Act
        var results = await _service.SearchKnowledgeAsync("vacuum");

        // Assert
        Assert.Single(results);
        Assert.Equal("Intake Manifold", results.First().Name);
    }
}
