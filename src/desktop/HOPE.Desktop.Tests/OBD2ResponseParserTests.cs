using Xunit;
using HOPE.Core.Services.OBD;
using HOPE.Core.Models;

namespace HOPE.Desktop.Tests;

/// <summary>
/// Tests for OBD2ResponseParser - parsing Mode 01 live data, Mode 03 DTCs, and VIN responses.
/// </summary>
public class OBD2ResponseParserTests
{
    #region Mode 01 - Live Data Response Tests
    
    [Fact]
    public void ParseMode01Response_EngineRPM_ReturnsCorrectValue()
    {
        // Arrange - "41 0C 0F A0" = RPM of ((0x0F * 256) + 0xA0) / 4 = (15 * 256 + 160) / 4 = 1000
        string response = "41 0C 0F A0";
        string pid = "0C";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("0C", result.PID);
        Assert.Equal("Engine RPM", result.Name);
        Assert.Equal("RPM", result.Unit);
        Assert.Equal(1000, result.Value, 0.1);
    }

    [Fact]
    public void ParseMode01Response_CoolantTemp_ReturnsCorrectValue()
    {
        // Arrange - "41 05 7B" = Coolant temp of 0x7B - 40 = 123 - 40 = 83°C
        string response = "41 05 7B";
        string pid = "05";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Coolant Temperature", result.Name);
        Assert.Equal("°C", result.Unit);
        Assert.Equal(83, result.Value, 0.1);
    }

    [Fact]
    public void ParseMode01Response_VehicleSpeed_ReturnsCorrectValue()
    {
        // Arrange - "41 0D 64" = Speed of 0x64 = 100 km/h
        string response = "41 0D 64";
        string pid = "0D";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Vehicle Speed", result.Name);
        Assert.Equal("km/h", result.Unit);
        Assert.Equal(100, result.Value, 0.1);
    }

    [Fact]
    public void ParseMode01Response_ThrottlePosition_ReturnsCorrectValue()
    {
        // Arrange - "41 11 80" = Throttle at 0x80 * 100 / 255 = 128 * 100 / 255 = ~50.2%
        string response = "41 11 80";
        string pid = "11";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Throttle Position", result.Name);
        Assert.Equal("%", result.Unit);
        Assert.Equal(50.2, result.Value, 0.5);
    }

    [Fact]
    public void ParseMode01Response_EngineLoad_ReturnsCorrectValue()
    {
        // Arrange - "41 04 FF" = Engine load at 0xFF * 100 / 255 = 100%
        string response = "41 04 FF";
        string pid = "04";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Engine Load", result.Name);
        Assert.Equal(100, result.Value, 0.1);
    }

    [Fact]
    public void ParseMode01Response_MAFSensor_ReturnsCorrectValue()
    {
        // Arrange - "41 10 03 E8" = MAF of ((0x03 * 256) + 0xE8) / 100 = 1000 / 100 = 10 g/s
        string response = "41 10 03 E8";
        string pid = "10";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Equal("MAF Air Flow Rate", result.Name);
        Assert.Equal("g/s", result.Unit);
        Assert.Equal(10, result.Value, 0.1);
    }

    [Fact]
    public void ParseMode01Response_NullResponse_ReturnsNull()
    {
        // Arrange
        string? response = null;
        string pid = "0C";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response!, pid);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void ParseMode01Response_EmptyResponse_ReturnsNull()
    {
        // Arrange
        string response = "";
        string pid = "0C";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void ParseMode01Response_InvalidMode_ReturnsNull()
    {
        // Arrange - Response doesn't start with "41"
        string response = "42 0C 0F A0";
        string pid = "0C";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void ParseMode01Response_MismatchedPID_ReturnsNull()
    {
        // Arrange - Requested 0C but got response for 0D
        string response = "41 0D 64";
        string pid = "0C";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.Null(result);
    }

    [Fact]
    public void ParseMode01Response_UnknownPID_ReturnsRawValue()
    {
        // Arrange - Unknown PID FF
        string response = "41 FF 42";
        string pid = "FF";

        // Act
        var result = OBD2ResponseParser.ParseMode01Response(response, pid);

        // Assert
        Assert.NotNull(result);
        Assert.Contains("Unknown PID", result.Name);
        Assert.Equal("raw", result.Unit);
        Assert.Equal(0x42, result.Value, 0.1);
    }

    #endregion

    #region Mode 03 - DTC Response Tests

    [Fact]
    public void ParseDTCs_SingleDTC_ReturnsCorrectCode()
    {
        // Arrange - "43 01 03" = P0103 (high nibble 0 = P0, remaining = 103)
        string response = "43 01 03";

        // Act
        var result = OBD2ResponseParser.ParseDTCs(response);

        // Assert
        Assert.Single(result);
        Assert.Equal("P0103", result[0].Code);
    }

    [Fact]
    public void ParseDTCs_MultipleDTCs_ReturnsAllCodes()
    {
        // Arrange - Two DTCs
        string response = "43 01 03 03 00";

        // Act
        var result = OBD2ResponseParser.ParseDTCs(response);

        // Assert
        Assert.Equal(2, result.Count);
    }

    [Fact]
    public void ParseDTCs_NoDTCs_ReturnsEmptyList()
    {
        // Arrange - No DTCs (00 00)
        string response = "43 00 00";

        // Act
        var result = OBD2ResponseParser.ParseDTCs(response);

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void ParseDTCs_EmptyResponse_ReturnsEmptyList()
    {
        // Act
        var result = OBD2ResponseParser.ParseDTCs("");

        // Assert
        Assert.Empty(result);
    }

    [Fact]
    public void ParseDTCs_InvalidMode_ReturnsEmptyList()
    {
        // Arrange - Response doesn't start with "43"
        string response = "41 01 03";

        // Act
        var result = OBD2ResponseParser.ParseDTCs(response);

        // Assert
        Assert.Empty(result);
    }

    #endregion

    #region Supported PIDs Response Tests

    [Fact]
    public void ParseSupportedPIDs_ValidResponse_ReturnsCorrectPIDs()
    {
        // Arrange - "41 00 BE 1F A8 13" = Various PIDs supported
        string response = "41 00 BE 1F A8 13";
        string basePID = "00";

        // Act
        var result = OBD2ResponseParser.ParseSupportedPIDs(response, basePID);

        // Assert
        Assert.NotEmpty(result);
        // BE = 1011 1110 -> PIDs 01, 03, 04, 05, 06, 07 are supported
        Assert.Contains("01", result);
        Assert.Contains("03", result);
        Assert.Contains("04", result);
    }

    [Fact]
    public void ParseSupportedPIDs_EmptyResponse_ReturnsEmptyList()
    {
        // Act
        var result = OBD2ResponseParser.ParseSupportedPIDs("", "00");

        // Assert
        Assert.Empty(result);
    }

    #endregion

    #region VIN Response Tests

    [Fact]
    public void ParseVIN_EmptyResponse_ReturnsNull()
    {
        // Act
        var result = OBD2ResponseParser.ParseVIN("");

        // Assert
        Assert.Null(result);
    }

    #endregion
}
