using System.IO;
using Microsoft.Data.Sqlite;
using HOPE.Core.Models;
using HOPE.Core.Services.Database;

namespace HOPE.Desktop.Tests;

public class DatabaseServiceTests : IAsyncLifetime
{
    private SqliteDatabaseService _service = null!;
    private string _testDbPath = null!;

    public async Task InitializeAsync()
    {
        // Use a unique temp file for each test run to avoid conflicts
        _testDbPath = Path.Combine(Path.GetTempPath(), $"hope_test_{Guid.NewGuid()}.db");
        _service = new SqliteDatabaseService(_testDbPath);
        await _service.InitializeAsync();
    }

    public Task DisposeAsync()
    {
        // Clear connection pool to release file handles before deletion
        SqliteConnection.ClearAllPools();

        // Clean up the test database
        if (File.Exists(_testDbPath))
        {
            File.Delete(_testDbPath);
        }
        return Task.CompletedTask;
    }

    [Fact]
    public async Task InitializeAsync_CreatesDatabase()
    {
        // Assert - database should exist after initialization
        Assert.True(File.Exists(_testDbPath));
    }

    [Fact]
    public async Task StartSessionAsync_ReturnsValidGuid()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();

        // Act
        var sessionId = await _service.StartSessionAsync(vehicleId);

        // Assert
        Assert.NotEqual(Guid.Empty, sessionId);
    }

    [Fact]
    public async Task GetSessionsAsync_ReturnsCreatedSessions()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        await _service.StartSessionAsync(vehicleId);
        await _service.StartSessionAsync(vehicleId);

        // Act
        var sessions = await _service.GetSessionsAsync();

        // Assert
        Assert.Equal(2, sessions.Count);
    }

    [Fact]
    public async Task EndSessionAsync_SetsEndTime()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var sessionId = await _service.StartSessionAsync(vehicleId);

        // Act
        await _service.EndSessionAsync(sessionId);
        var sessions = await _service.GetSessionsAsync();

        // Assert
        var session = sessions.First(s => s.Id == sessionId);
        Assert.NotNull(session.EndTime);
    }

    [Fact]
    public async Task LogReadingAsync_StoresReading()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var sessionId = await _service.StartSessionAsync(vehicleId);
        var reading = new OBD2Reading
        {
            SessionId = sessionId,
            PID = OBD2PIDs.EngineRPM,
            Name = "Engine RPM",
            Value = 3500,
            Unit = "RPM",
            Timestamp = DateTime.UtcNow
        };

        // Act
        await _service.LogReadingAsync(reading);
        var data = await _service.GetSessionDataAsync(sessionId);

        // Assert
        Assert.Single(data);
        Assert.Equal(OBD2PIDs.EngineRPM, data[0].PID);
        Assert.Equal(3500, data[0].Value);
    }

    [Fact]
    public async Task LogReadingsAsync_StoresMultipleReadings()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var sessionId = await _service.StartSessionAsync(vehicleId);
        var readings = new List<OBD2Reading>
        {
            new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.EngineRPM,
                Name = "Engine RPM",
                Value = 3500,
                Unit = "RPM",
                Timestamp = DateTime.UtcNow
            },
            new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.VehicleSpeed,
                Name = "Vehicle Speed",
                Value = 60,
                Unit = "km/h",
                Timestamp = DateTime.UtcNow
            },
            new OBD2Reading
            {
                SessionId = sessionId,
                PID = OBD2PIDs.CoolantTemp,
                Name = "Coolant Temperature",
                Value = 90,
                Unit = "°C",
                Timestamp = DateTime.UtcNow
            }
        };

        // Act
        await _service.LogReadingsAsync(readings);
        var data = await _service.GetSessionDataAsync(sessionId);

        // Assert
        Assert.Equal(3, data.Count);
    }

    [Fact]
    public async Task GetSessionDataAsync_ReturnsEmptyForNonExistentSession()
    {
        // Act
        var data = await _service.GetSessionDataAsync(Guid.NewGuid());

        // Assert
        Assert.Empty(data);
    }

    [Fact]
    public async Task GetSessionsAsync_ReturnsSessionsInDescendingOrder()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var session1 = await _service.StartSessionAsync(vehicleId);
        await Task.Delay(10); // Ensure different timestamps
        var session2 = await _service.StartSessionAsync(vehicleId);

        // Act
        var sessions = await _service.GetSessionsAsync();

        // Assert - most recent first
        Assert.Equal(session2, sessions[0].Id);
        Assert.Equal(session1, sessions[1].Id);
    }

    [Fact]
    public async Task LogReadingAsync_HandlesNullRawResponse()
    {
        // Arrange
        var vehicleId = Guid.NewGuid();
        var sessionId = await _service.StartSessionAsync(vehicleId);
        var reading = new OBD2Reading
        {
            SessionId = sessionId,
            PID = OBD2PIDs.EngineRPM,
            Name = "Engine RPM",
            Value = 3500,
            Unit = "RPM",
            RawResponse = null,
            Timestamp = DateTime.UtcNow
        };

        // Act
        await _service.LogReadingAsync(reading);
        var data = await _service.GetSessionDataAsync(sessionId);

        // Assert
        Assert.Single(data);
        Assert.Null(data[0].RawResponse);
    }
}
