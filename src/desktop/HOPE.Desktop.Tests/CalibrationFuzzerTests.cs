using System.IO;
using System.Text.Json;
using HOPE.Core.Services.ECU;
using Xunit;

namespace HOPE.Desktop.Tests;

public class CalibrationFuzzerTests : IAsyncLifetime, IDisposable
{
    private readonly string _testRepoPath;
    private readonly CalibrationRepository _repository;

    public CalibrationFuzzerTests()
    {
        _testRepoPath = Path.Combine(Path.GetTempPath(), "HOPE_Fuzz_Repo_" + Guid.NewGuid());
        _repository = new CalibrationRepository(_testRepoPath);
    }

    public async Task InitializeAsync()
    {
        await _repository.InitializeAsync();
    }

    public Task DisposeAsync()
    {
        return Task.CompletedTask;
    }

    public void Dispose()
    {
        if (Directory.Exists(_testRepoPath))
            Directory.Delete(_testRepoPath, true);
    }

    [Theory]
    [InlineData("{\"EcuId\": \"TEST\", \"Blocks\": []}", false)] // Empty blocks
    [InlineData("{\"EcuId\": null, \"Blocks\": []}", true)] // Null EcuId (should catch)
    [InlineData("not a json", true)] // Invalid JSON
    [InlineData("{\"EcuId\": \"TEST\", \"Blocks\": [{\"Name\": \"B1\", \"StartAddress\": -1, \"Data\": \"\"}]}", true)] // Negative address
    public async Task LoadCalibration_CorruptedJson_ThrowsOrHandles(string json, bool shouldThrow)
    {
        // Arrange
        var filePath = Path.Combine(_testRepoPath, "corrupted.json");
        await File.WriteAllTextAsync(filePath, json);

        // Act & Assert
        if (shouldThrow)
        {
            await Assert.ThrowsAsync<JsonException>(async () => 
            {
                var content = await File.ReadAllTextAsync(filePath);
                JsonSerializer.Deserialize<CalibrationFile>(content);
            });
        }
        else
        {
            var content = await File.ReadAllTextAsync(filePath);
            var cal = JsonSerializer.Deserialize<CalibrationFile>(content);
            Assert.NotNull(cal);
        }
    }

    [Fact]
    public async Task ValidateChecksum_FailsOnMismatchedData()
    {
        // Arrange
        var cal = new CalibrationFile
        {
            EcuId = "TEST",
            Blocks = new List<CalibrationBlock>
            {
                new CalibrationBlock 
                { 
                    Name = "B1", 
                    StartAddress = 0x1000, 
                    Data = new byte[] { 0x01, 0x02 },
                    Checksum = "wrong" 
                }
            }
        };

        // Act
        // Note: CalibrationRepository.ValidateChecksumAsync is meant to check if 
        // the computed checksum matches the one in the object.
        var isValid = await _repository.ValidateChecksumAsync(cal);

        // Assert
        Assert.False(isValid.IsValid);
    }

    [Fact]
    public void Fuzz_DataMutations_ShouldNotCrash()
    {
        // Create a valid cal
        var cal = new CalibrationFile
        {
            EcuId = "FUZZ",
            Blocks = new List<CalibrationBlock>
            {
                new CalibrationBlock { Name = "B1", StartAddress = 0, Data = new byte[100] }
            }
        };

        var random = new Random(42);
        
        for (int i = 0; i < 100; i++)
        {
            try
            {
                // Mutate
                var mutationType = random.Next(3);
                switch (mutationType)
                {
                    case 0: cal.EcuId = null!; break;
                    case 1: cal.Blocks[0].Data = null!; break;
                    case 2: cal.Blocks[0].StartAddress = (uint)random.Next(); break;
                }
            }
            catch (Exception)
            {
                // Validation exceptions are fine, we just want to see if serialization crashes
            }

            // Test if serialize/deserialize crashes
            try
            {
                var json = JsonSerializer.Serialize(cal);
                var deserialized = JsonSerializer.Deserialize<CalibrationFile>(json);
            }
            catch (Exception)
            {
                // Exceptions are expected for nulls, but it shouldn't crash the process
            }
        }
    }
}
