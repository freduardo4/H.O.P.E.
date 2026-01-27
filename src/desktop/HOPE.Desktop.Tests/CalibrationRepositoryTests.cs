using System.IO;
using HOPE.Core.Services.ECU;
using Xunit;

namespace HOPE.Desktop.Tests;

public class CalibrationRepositoryTests : IDisposable
{
    private readonly string _testRepoPath;
    private readonly CalibrationRepository _repository;

    public CalibrationRepositoryTests()
    {
        _testRepoPath = Path.Combine(Path.GetTempPath(), $"hope_test_repo_{Guid.NewGuid():N}");
        _repository = new CalibrationRepository(_testRepoPath);
    }

    public void Dispose()
    {
        _repository.Dispose();
        if (Directory.Exists(_testRepoPath))
        {
            Directory.Delete(_testRepoPath, recursive: true);
        }
    }

    [Fact]
    public async Task Initialize_CreatesRepositoryStructure()
    {
        // Act
        await _repository.InitializeAsync();

        // Assert
        Assert.True(Directory.Exists(_testRepoPath));
        Assert.True(Directory.Exists(Path.Combine(_testRepoPath, "objects")));
        Assert.True(Directory.Exists(Path.Combine(_testRepoPath, "refs")));
        Assert.True(Directory.Exists(Path.Combine(_testRepoPath, "staging")));
        Assert.True(File.Exists(Path.Combine(_testRepoPath, "HEAD")));
        Assert.True(File.Exists(Path.Combine(_testRepoPath, "index.json")));
    }

    [Fact]
    public async Task Stage_AddsCalibrationToStagingArea()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");

        // Act
        await _repository.StageAsync(calibration);

        // Assert
        var stagingFile = Path.Combine(_testRepoPath, "staging", "ECU001.cal");
        Assert.True(File.Exists(stagingFile));
    }

    [Fact]
    public async Task Commit_CreatesPersistentCommit()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");
        await _repository.StageAsync(calibration);

        // Act
        var commitHash = await _repository.CommitAsync("Initial calibration backup", "Test Author");

        // Assert
        Assert.NotNull(commitHash);
        Assert.Equal(64, commitHash.Length); // SHA-256 hex string

        // Verify staging area is cleared
        var stagingFile = Path.Combine(_testRepoPath, "staging", "ECU001.cal");
        Assert.False(File.Exists(stagingFile));
    }

    [Fact]
    public async Task GetHistory_ReturnsCommitHistory()
    {
        // Arrange
        await _repository.InitializeAsync();

        var calibration1 = CreateTestCalibration("ECU001");
        await _repository.StageAsync(calibration1);
        await _repository.CommitAsync("First commit");

        var calibration2 = CreateTestCalibration("ECU001", new byte[] { 0x01, 0x02, 0x03 });
        await _repository.StageAsync(calibration2);
        await _repository.CommitAsync("Second commit");

        // Act
        var history = await _repository.GetHistoryAsync();

        // Assert
        Assert.Equal(2, history.Count);
        Assert.Equal("Second commit", history[0].Message);
        Assert.Equal("First commit", history[1].Message);
    }

    [Fact]
    public async Task ValidateChecksum_ReturnsTrueForValidCalibration()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");

        // Act
        var result = await _repository.ValidateChecksumAsync(calibration);

        // Assert
        Assert.True(result.IsValid);
        Assert.Equal(result.ExpectedChecksum, result.CalculatedChecksum);
    }

    [Fact]
    public async Task ValidateChecksum_ReturnsFalseForCorruptedCalibration()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");

        // Corrupt the checksum
        calibration.FullChecksum = "invalid_checksum";

        // Act
        var result = await _repository.ValidateChecksumAsync(calibration);

        // Assert
        Assert.False(result.IsValid);
        Assert.NotEqual(result.ExpectedChecksum, result.CalculatedChecksum);
    }

    [Fact]
    public async Task DiffCalibrations_IdentifiesChanges()
    {
        // Arrange
        await _repository.InitializeAsync();

        var calibration1 = CreateTestCalibration("ECU001", new byte[] { 0x00, 0x01, 0x02, 0x03 });
        var calibration2 = CreateTestCalibration("ECU001", new byte[] { 0x00, 0xFF, 0x02, 0x03 });

        // Act
        var diff = await _repository.DiffCalibrationsAsync(calibration1, calibration2);

        // Assert
        Assert.Single(diff.Changes);
        Assert.Equal(ChangeType.Modified, diff.Changes[0].ChangeType);
        Assert.Single(diff.Changes[0].ByteChanges!);
        Assert.Equal((uint)0x1001, diff.Changes[0].ByteChanges![0].Address); // 0x1000 + 1
        Assert.Equal((byte)0x01, diff.Changes[0].ByteChanges![0].OldValue);
        Assert.Equal((byte)0xFF, diff.Changes[0].ByteChanges![0].NewValue);
    }

    [Fact]
    public async Task DiffCalibrations_DetectsAddedBlocks()
    {
        // Arrange
        await _repository.InitializeAsync();

        var calibration1 = CreateTestCalibration("ECU001");
        var calibration2 = CreateTestCalibration("ECU001");
        calibration2.Blocks.Add(new CalibrationBlock
        {
            Name = "NewBlock",
            StartAddress = 0x2000,
            Data = new byte[] { 0xAA, 0xBB },
            Checksum = "dummy"
        });

        // Act
        var diff = await _repository.DiffCalibrationsAsync(calibration1, calibration2);

        // Assert
        Assert.Contains(diff.Changes, c => c.ChangeType == ChangeType.Added && c.BlockName == "NewBlock");
    }

    [Fact]
    public async Task CreateTag_CreatesNamedTag()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");
        await _repository.StageAsync(calibration);
        var commitHash = await _repository.CommitAsync("Release version");

        // Act
        await _repository.CreateTagAsync("v1.0.0", commitHash, "First release");

        // Assert
        var tagPath = Path.Combine(_testRepoPath, "refs", "tags", "v1.0.0");
        Assert.True(File.Exists(tagPath));
    }

    [Fact]
    public async Task ExportToBinary_CreatesValidBinaryFile()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001", new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04 });
        var outputPath = Path.Combine(_testRepoPath, "export.bin");

        // Act
        await _repository.ExportToBinaryAsync(calibration, outputPath);

        // Assert
        Assert.True(File.Exists(outputPath));
        var exportedData = await File.ReadAllBytesAsync(outputPath);
        Assert.Equal(5, exportedData.Length);
        Assert.Equal(new byte[] { 0x00, 0x01, 0x02, 0x03, 0x04 }, exportedData);
    }

    [Fact]
    public async Task CommitCreated_EventFires()
    {
        // Arrange
        await _repository.InitializeAsync();
        var calibration = CreateTestCalibration("ECU001");
        await _repository.StageAsync(calibration);

        CalibrationCommitEventArgs? eventArgs = null;
        _repository.CommitCreated += (sender, args) => eventArgs = args;

        // Act
        await _repository.CommitAsync("Test commit", "Test Author");

        // Assert
        Assert.NotNull(eventArgs);
        Assert.Equal("Test commit", eventArgs.Message);
        Assert.Equal("Test Author", eventArgs.Author);
    }

    [Fact]
    public async Task Commit_WithNoStagedFiles_ThrowsException()
    {
        // Arrange
        await _repository.InitializeAsync();

        // Act & Assert
        await Assert.ThrowsAsync<InvalidOperationException>(
            () => _repository.CommitAsync("Empty commit"));
    }

    #region Helper Methods

    private CalibrationFile CreateTestCalibration(string ecuId, byte[]? data = null)
    {
        data ??= new byte[] { 0x00, 0x01, 0x02, 0x03 };

        var block = new CalibrationBlock
        {
            Name = "MainMap",
            StartAddress = 0x1000,
            Data = data,
            Checksum = ComputeChecksum(data)
        };

        var calibration = new CalibrationFile
        {
            EcuId = ecuId,
            ReadTimestamp = DateTime.UtcNow,
            Metadata = new CalibrationMetadata
            {
                VIN = "TEST123456789",
                EcuPartNumber = "12345678",
                SoftwareVersion = "1.0"
            },
            Blocks = new List<CalibrationBlock> { block }
        };

        calibration.FullChecksum = ComputeFileChecksum(calibration);
        return calibration;
    }

    private static string ComputeChecksum(byte[] data)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        var hash = sha.ComputeHash(data);
        return Convert.ToHexString(hash).ToLower();
    }

    private static string ComputeFileChecksum(CalibrationFile calibration)
    {
        using var sha = System.Security.Cryptography.SHA256.Create();
        using var stream = new MemoryStream();

        foreach (var block in calibration.Blocks.OrderBy(b => b.StartAddress))
        {
            stream.Write(block.Data);
        }

        stream.Position = 0;
        var hash = sha.ComputeHash(stream);
        return Convert.ToHexString(hash).ToLower();
    }

    #endregion
}
