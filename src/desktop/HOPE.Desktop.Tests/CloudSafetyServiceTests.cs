using System.Net;
using System.Net.Http;
using HOPE.Core.Services.Safety;
using Xunit;

namespace HOPE.Desktop.Tests;

public class CloudSafetyServiceTests
{
    private ConfigurableSafetyHandler _handler;
    private CloudSafetyService _service;

    private void SetupService(Func<HttpRequestMessage, Task<HttpResponseMessage>> handlerFunc)
    {
        _handler = new ConfigurableSafetyHandler(handlerFunc);
        _service = new CloudSafetyService(new HttpClient(_handler));
    }

    [Fact]
    public async Task ValidateFlashOperationAsync_ReturnsTrue_WhenCloudApproves()
    {
        // Arrange
        SetupService(async req => new HttpResponseMessage
        {
            StatusCode = HttpStatusCode.OK,
            Content = new StringContent("{\"allowed\": true}")
        });

        // Act
        var result = await _service.ValidateFlashOperationAsync("ECU123", 12.5);

        // Assert
        Assert.True(result);
    }

    [Fact]
    public async Task ValidateFlashOperationAsync_ReturnsFalse_WhenCloudDenies()
    {
        // Arrange
        SetupService(async req => new HttpResponseMessage
        {
            StatusCode = HttpStatusCode.OK,
            Content = new StringContent("{\"allowed\": false}")
        });

        // Act
        var result = await _service.ValidateFlashOperationAsync("ECU123", 12.5);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task ValidateFlashOperationAsync_ReturnsFalse_WhenCloudErrors()
    {
        // Arrange
        SetupService(async req => new HttpResponseMessage
        {
            StatusCode = HttpStatusCode.InternalServerError
        });

        // Act
        var result = await _service.ValidateFlashOperationAsync("ECU123", 12.5);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task ValidateFlashOperationAsync_ReturnsFalse_WhenNetworkFails()
    {
        // Arrange
        SetupService(async req => throw new HttpRequestException("Network error"));

        // Act
        var result = await _service.ValidateFlashOperationAsync("ECU123", 12.5);

        // Assert
        Assert.False(result);
    }

    [Fact]
    public async Task LogSafetyEventAsync_SendsRequest_ToCorrectEndpoint()
    {
        // Arrange
        HttpRequestMessage? capturedRequest = null;
        SetupService(async req => 
        {
            capturedRequest = req;
            return new HttpResponseMessage { StatusCode = HttpStatusCode.OK };
        });

        var evt = new SafetyEvent
        {
            EventType = "TEST",
            EcuId = "ECU1",
            Timestamp = DateTime.UtcNow
        };

        // Act
        await _service.LogSafetyEventAsync(evt);

        // Assert
        Assert.NotNull(capturedRequest);
        Assert.Equal(HttpMethod.Post, capturedRequest.Method);
        Assert.EndsWith("/safety/telemetry", capturedRequest.RequestUri?.ToString());
    }
}

public class ConfigurableSafetyHandler : HttpMessageHandler
{
    private readonly Func<HttpRequestMessage, Task<HttpResponseMessage>> _handler;

    public ConfigurableSafetyHandler(Func<HttpRequestMessage, Task<HttpResponseMessage>> handler)
    {
        _handler = handler;
    }

    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        return _handler(request);
    }
}
