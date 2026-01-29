using System;
using System.IO;
using System.Linq;
using System.Reactive.Subjects;
using System.Threading.Tasks;
using HOPE.Core.Hardware;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.Diagnostics;
using Microsoft.Extensions.Logging;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Services;

public class FlightRecorderServiceTests
{
    private readonly Mock<IHardwareAdapter> _hardwareMock;
    private readonly Mock<ILogger<FlightRecorderService>> _loggerMock;
    private readonly Subject<byte[]> _messageStream;
    private readonly FlightRecorderService _service;

    public FlightRecorderServiceTests()
    {
        _hardwareMock = new Mock<IHardwareAdapter>();
        _messageStream = new Subject<byte[]>();
        _hardwareMock.Setup(h => h.StreamMessages()).Returns(_messageStream);
        
        _loggerMock = new Mock<ILogger<FlightRecorderService>>();
        
        // Small buffer for testing overflow (Size 3)
        _service = new FlightRecorderService(_hardwareMock.Object, _loggerMock.Object, bufferSize: 3);
    }

    [Fact]
    public async Task DumpToFileAsync_WritesBufferedMessages()
    {
        // Arrange
        var path = Path.Combine(Path.GetTempPath(), $"FlightRecord_{Guid.NewGuid()}.csv");
        _service.StartRecording();
        
        _messageStream.OnNext(new byte[] { 0x11, 0x22 });
        _messageStream.OnNext(new byte[] { 0x33, 0x44 });
        
        // Act
        await _service.DumpToFileAsync(path);
        
        // Assert
        Assert.True(File.Exists(path), "File should be created");
        var lines = await File.ReadAllLinesAsync(path);
        // Header + 2 rows = 3 lines
        Assert.Equal(3, lines.Length); 
        Assert.Contains("1122", lines[1]);
        Assert.Contains("3344", lines[2]);
        
        // Cleanup
        if (File.Exists(path)) File.Delete(path);
    }

    [Fact]
    public async Task Buffer_OverflowsCorrectly()
    {
        // Arrange
        var path = Path.Combine(Path.GetTempPath(), $"FlightRecord_Overflow_{Guid.NewGuid()}.csv");
        _service.StartRecording();
        
        // Push 4 messages (Buffer size is 3)
        // 1, 2, 3, 4
        // Logic should keep: 2, 3, 4
        
        _messageStream.OnNext(new byte[] { 0x01 });
        _messageStream.OnNext(new byte[] { 0x02 });
        _messageStream.OnNext(new byte[] { 0x03 });
        _messageStream.OnNext(new byte[] { 0x04 });
        
        // Act
        await _service.DumpToFileAsync(path);
        
        // Assert
        var lines = await File.ReadAllLinesAsync(path);
        
        // Header + 3 rows (Size 3) = 4 lines
        Assert.Equal(4, lines.Length);
        
        var content = string.Join(Environment.NewLine, lines);
        
        // Check contents
        Assert.DoesNotContain(",01", content); // 01 should be dropped
        Assert.Contains(",02", content); // 02 should be first
        Assert.Contains(",04", content); // 04 should be last
        
        // Cleanup
        if (File.Exists(path)) File.Delete(path);
    }
}
