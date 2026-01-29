using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.Database;
using HOPE.Core.Services.Security;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Services.Security;

public class CalibrationLedgerServiceTests
{
    private readonly Mock<IDatabaseService> _dbMock;
    private readonly CalibrationLedgerService _service;

    public CalibrationLedgerServiceTests()
    {
        _dbMock = new Mock<IDatabaseService>();
        _service = new CalibrationLedgerService(_dbMock.Object);
    }

    [Fact]
    public async Task CommmitChangeAsync_CreatesValidChainedBlock()
    {
        // Arrange
        var calId = Guid.NewGuid();
        var data = new byte[] { 0x01, 0x02, 0x03 };
        var author = "TestAuthor";
        var summary = "Initial Commit";

        _dbMock.Setup(d => d.GetLastLedgerEntryAsync()).ReturnsAsync((LedgerEntry)null);

        // Act
        var hash = await _service.CommmitChangeAsync(calId, data, author, summary);

        // Assert
        Assert.NotNull(hash);
        _dbMock.Verify(d => d.AddLedgerEntryAsync(It.Is<LedgerEntry>(e => 
            e.BlockHeight == 1 && 
            e.Author == author &&
            e.PreviousBlockHash == "0000000000000000000000000000000000000000000000000000000000000000")), Times.Once);
    }

    [Fact]
    public async Task VerifyLedgerAsync_DetectsTamper()
    {
        // Arrange
        var entries = new List<LedgerEntry>
        {
            new LedgerEntry { BlockHeight = 1, PreviousBlockHash = "0000000000000000000000000000000000000000000000000000000000000000", BlockHash = "HASH1", Timestamp = DateTime.UtcNow },
            new LedgerEntry { BlockHeight = 2, PreviousBlockHash = "HASH1", BlockHash = "HASH2", Timestamp = DateTime.UtcNow }
        };

        // Injecting correct hashes is hard in a mock without calling the actual logic, 
        // but we can test the failure case easily.
        _dbMock.Setup(d => d.GetLedgerEntriesAsync()).ReturnsAsync(entries);

        // Act
        var result = await _service.VerifyLedgerAsync();

        // Assert
        Assert.False(result.IsValid);
        Assert.Contains("Mismatched hash", result.Message);
    }
}
