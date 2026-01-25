using Microsoft.Data.Sqlite;
using HOPE.Core.Models;
using System.IO;

namespace HOPE.Core.Services.Database;

public class SqliteDatabaseService : IDatabaseService
{
    private readonly string _dbPath;
    private readonly string _connectionString;

    public SqliteDatabaseService() : this(null)
    {
    }

    public SqliteDatabaseService(string? customDbPath)
    {
        if (customDbPath != null)
        {
            _dbPath = customDbPath;
        }
        else
        {
            string appData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string hopeDir = Path.Combine(appData, "HOPE");
            if (!Directory.Exists(hopeDir)) Directory.CreateDirectory(hopeDir);
            _dbPath = Path.Combine(hopeDir, "hope.db");
        }

        _connectionString = $"Data Source={_dbPath}";
    }

    public async Task InitializeAsync()
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        var createSessionsTable = @"
            CREATE TABLE IF NOT EXISTS Sessions (
                Id GUID PRIMARY KEY,
                VehicleId GUID,
                StartTime DATETIME,
                EndTime DATETIME NULL,
                Notes TEXT
            )";

        var createReadingsTable = @"
            CREATE TABLE IF NOT EXISTS Readings (
                Id INTEGER PRIMARY KEY AUTOINCREMENT,
                Timestamp DATETIME,
                SessionId GUID,
                PID TEXT,
                Name TEXT,
                Value REAL,
                Unit TEXT,
                RawResponse TEXT,
                FOREIGN KEY(SessionId) REFERENCES Sessions(Id)
            )";

        using var command = connection.CreateCommand();
        command.CommandText = createSessionsTable;
        await command.ExecuteNonQueryAsync();
        
        command.CommandText = createReadingsTable;
        await command.ExecuteNonQueryAsync();
    }

    public async Task<Guid> StartSessionAsync(Guid vehicleId)
    {
        var sessionId = Guid.NewGuid();
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        using var command = connection.CreateCommand();
        command.CommandText = "INSERT INTO Sessions (Id, VehicleId, StartTime) VALUES ($id, $vId, $start)";
        command.Parameters.AddWithValue("$id", sessionId);
        command.Parameters.AddWithValue("$vId", vehicleId);
        command.Parameters.AddWithValue("$start", DateTime.UtcNow);
        
        await command.ExecuteNonQueryAsync();
        return sessionId;
    }

    public async Task EndSessionAsync(Guid sessionId)
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        using var command = connection.CreateCommand();
        command.CommandText = "UPDATE Sessions SET EndTime = $end WHERE Id = $id";
        command.Parameters.AddWithValue("$end", DateTime.UtcNow);
        command.Parameters.AddWithValue("$id", sessionId);
        
        await command.ExecuteNonQueryAsync();
    }

    public async Task LogReadingAsync(OBD2Reading reading)
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        using var command = connection.CreateCommand();
        command.CommandText = @"
            INSERT INTO Readings (Timestamp, SessionId, PID, Name, Value, Unit, RawResponse)
            VALUES ($time, $sid, $pid, $name, $val, $unit, $raw)";
        
        command.Parameters.AddWithValue("$time", reading.Timestamp);
        command.Parameters.AddWithValue("$sid", reading.SessionId);
        command.Parameters.AddWithValue("$pid", reading.PID);
        command.Parameters.AddWithValue("$name", reading.Name);
        command.Parameters.AddWithValue("$val", reading.Value);
        command.Parameters.AddWithValue("$unit", reading.Unit);
        command.Parameters.AddWithValue("$raw", (object?)reading.RawResponse ?? DBNull.Value);

        await command.ExecuteNonQueryAsync();
    }

    public async Task LogReadingsAsync(IEnumerable<OBD2Reading> readings)
    {
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();
        using var transaction = connection.BeginTransaction();

        try
        {
            foreach (var reading in readings)
            {
                using var command = connection.CreateCommand();
                command.Transaction = transaction;
                command.CommandText = @"
                    INSERT INTO Readings (Timestamp, SessionId, PID, Name, Value, Unit, RawResponse)
                    VALUES ($time, $sid, $pid, $name, $val, $unit, $raw)";
                
                command.Parameters.AddWithValue("$time", reading.Timestamp);
                command.Parameters.AddWithValue("$sid", reading.SessionId);
                command.Parameters.AddWithValue("$pid", reading.PID);
                command.Parameters.AddWithValue("$name", reading.Name);
                command.Parameters.AddWithValue("$val", reading.Value);
                command.Parameters.AddWithValue("$unit", reading.Unit);
                command.Parameters.AddWithValue("$raw", (object?)reading.RawResponse ?? DBNull.Value);

                await command.ExecuteNonQueryAsync();
            }
            await transaction.CommitAsync();
        }
        catch
        {
            await transaction.RollbackAsync();
            throw;
        }
    }

    public async Task<List<DiagnosticSession>> GetSessionsAsync()
    {
        var sessions = new List<DiagnosticSession>();
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM Sessions ORDER BY StartTime DESC";
        
        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            sessions.Add(new DiagnosticSession
            {
                Id = reader.GetGuid(0),
                VehicleId = reader.GetGuid(1),
                StartTime = reader.GetDateTime(2),
                EndTime = reader.IsDBNull(3) ? null : reader.GetDateTime(3),
                Notes = reader.IsDBNull(4) ? "" : reader.GetString(4)
            });
        }
        return sessions;
    }

    public async Task<List<OBD2Reading>> GetSessionDataAsync(Guid sessionId)
    {
        var data = new List<OBD2Reading>();
        using var connection = new SqliteConnection(_connectionString);
        await connection.OpenAsync();

        using var command = connection.CreateCommand();
        command.CommandText = "SELECT * FROM Readings WHERE SessionId = $sid ORDER BY Timestamp ASC";
        command.Parameters.AddWithValue("$sid", sessionId);
        
        using var reader = await command.ExecuteReaderAsync();
        while (await reader.ReadAsync())
        {
            data.Add(new OBD2Reading
            {
                Timestamp = reader.GetDateTime(1),
                SessionId = reader.GetGuid(2),
                PID = reader.GetString(3),
                Name = reader.GetString(4),
                Value = reader.GetDouble(5),
                Unit = reader.GetString(6),
                RawResponse = reader.IsDBNull(7) ? null : reader.GetString(7)
            });
        }
        return data;
    }
}
