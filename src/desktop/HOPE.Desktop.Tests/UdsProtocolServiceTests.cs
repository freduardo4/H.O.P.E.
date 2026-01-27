using HOPE.Core.Interfaces;
using HOPE.Core.Services.Protocols;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests;

public class UdsProtocolServiceTests : IDisposable
{
    private readonly Mock<IHardwareAdapter> _mockAdapter;
    private readonly UdsProtocolService _service;

    public UdsProtocolServiceTests()
    {
        _mockAdapter = new Mock<IHardwareAdapter>();
        _mockAdapter.Setup(a => a.IsConnected).Returns(true);
        _mockAdapter.Setup(a => a.Type).Returns(HardwareType.J2534);

        _service = new UdsProtocolService(_mockAdapter.Object);
    }

    public void Dispose()
    {
        _service.Dispose();
    }

    [Fact]
    public async Task DiagnosticSessionControl_DefaultSession_ReturnsPositiveResponse()
    {
        // Arrange
        var expectedResponse = new byte[] { 0x50, 0x01 }; // Positive response to 0x10
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.DiagnosticSessionControl),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedResponse);

        // Act
        var response = await _service.DiagnosticSessionControlAsync(UdsSession.Default);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(UdsSession.Default, _service.CurrentSession);
    }

    [Fact]
    public async Task DiagnosticSessionControl_ExtendedSession_ChangesCurrentSession()
    {
        // Arrange
        var expectedResponse = new byte[] { 0x50, 0x03 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.IsAny<byte[]>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedResponse);

        // Act
        await _service.DiagnosticSessionControlAsync(UdsSession.ExtendedDiagnostic);

        // Assert
        Assert.Equal(UdsSession.ExtendedDiagnostic, _service.CurrentSession);
    }

    [Fact]
    public async Task DiagnosticSessionControl_NegativeResponse_ReturnsError()
    {
        // Arrange - Negative response (0x7F, ServiceId, NRC)
        var negativeResponse = new byte[] { 0x7F, 0x10, 0x22 }; // Conditions not correct
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.IsAny<byte[]>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(negativeResponse);

        // Act
        var response = await _service.DiagnosticSessionControlAsync(UdsSession.Programming);

        // Assert
        Assert.False(response.IsPositive);
        Assert.Equal(UdsNrc.ConditionsNotCorrect, response.NegativeResponseCode);
    }

    [Fact]
    public async Task ReadDataByIdentifier_ValidDID_ReturnsData()
    {
        // Arrange
        ushort testDid = 0xF190; // VIN DID
        var expectedResponse = new byte[] { 0x62, 0xF1, 0x90, 0x41, 0x42, 0x43 }; // "ABC"
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.ReadDataByIdentifier),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(expectedResponse);

        // Act
        var response = await _service.ReadDataByIdentifierAsync(testDid);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(testDid, response.DataIdentifier);
        Assert.Equal(new byte[] { 0x41, 0x42, 0x43 }, response.Data);
    }

    [Fact]
    public async Task SecurityAccess_RequestSeed_ReturnsSeed()
    {
        // Arrange
        var seedResponse = new byte[] { 0x67, 0x01, 0xAA, 0xBB, 0xCC, 0xDD }; // Seed bytes
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.SecurityAccess && b[1] == 0x01),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(seedResponse);

        // Act
        var response = await _service.RequestSecuritySeedAsync(1);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(4, response.Seed.Length);
        Assert.Equal(new byte[] { 0xAA, 0xBB, 0xCC, 0xDD }, response.Seed);
    }

    [Fact]
    public async Task SecurityAccess_SendValidKey_UnlocksAccess()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x67, 0x02 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.SecurityAccess && b[1] == 0x02),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.SendSecurityKeyAsync(1, new byte[] { 0x11, 0x22, 0x33, 0x44 });

        // Assert
        Assert.True(response.IsPositive);
        Assert.True(_service.IsSecurityUnlocked);
    }

    [Fact]
    public async Task SecurityAccess_InvalidKey_DeniesAccess()
    {
        // Arrange
        var negativeResponse = new byte[] { 0x7F, 0x27, 0x35 }; // Invalid key
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.IsAny<byte[]>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(negativeResponse);

        // Act
        var response = await _service.SendSecurityKeyAsync(1, new byte[] { 0xFF, 0xFF, 0xFF, 0xFF });

        // Assert
        Assert.False(response.IsPositive);
        Assert.Equal(UdsNrc.InvalidKey, response.NegativeResponseCode);
        Assert.False(_service.IsSecurityUnlocked);
    }

    [Fact]
    public async Task WriteDataByIdentifier_ValidData_ReturnsPositive()
    {
        // Arrange
        ushort testDid = 0xF199;
        var positiveResponse = new byte[] { 0x6E, 0xF1, 0x99 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.WriteDataByIdentifier),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.WriteDataByIdentifierAsync(testDid, new byte[] { 0x01, 0x02 });

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task InputOutputControl_ReturnControlToEcu_Succeeds()
    {
        // Arrange
        ushort actuatorDid = 0x0100;
        var positiveResponse = new byte[] { 0x6F, 0x01, 0x00, 0x00 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.InputOutputControlByIdentifier),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.InputOutputControlByIdentifierAsync(
            actuatorDid,
            IoControlParameter.ReturnControlToEcu);

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task RoutineControl_StartRoutine_ReturnsRoutineInfo()
    {
        // Arrange
        ushort routineId = 0x0203;
        var positiveResponse = new byte[] { 0x71, 0x01, 0x02, 0x03, 0xAA, 0xBB };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.RoutineControl),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.StartRoutineAsync(routineId);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(routineId, response.RoutineId);
        Assert.Equal(new byte[] { 0xAA, 0xBB }, response.RoutineInfo);
    }

    [Fact]
    public async Task ReadDtcInformation_ReturnsDtcList()
    {
        // Arrange
        // Response: 59 02 FF [DTC1: P0101] [Status1] [DTC2: P0300] [Status2]
        var positiveResponse = new byte[]
        {
            0x59, 0x02, 0xFF,
            0x01, 0x01, 0x00, 0x09, // P0101, Active
            0x03, 0x00, 0x00, 0x08  // P0300, Stored
        };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.ReadDtcInformation),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.ReadDtcInformationAsync(DtcReportType.ReportDTCByStatusMask);

        // Assert
        Assert.True(response.IsPositive);
        Assert.Equal(2, response.DTCs.Count);
    }

    [Fact]
    public async Task ClearDiagnosticInformation_ClearsAllDtcs()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x54 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.ClearDiagnosticInformation),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.ClearDiagnosticInformationAsync();

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task RequestDownload_ReturnsMaxBlockSize()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x74, 0x20, 0x0F, 0xF0 }; // Max block 0xFF0
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.RequestDownload),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.RequestDownloadAsync(0x10000, 0x1000);

        // Assert
        Assert.True(response.IsPositive);
        Assert.True(response.MaxBlockLength > 0);
    }

    [Fact]
    public async Task TransferData_TransfersBlock()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x76, 0x01 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.TransferData),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.TransferDataAsync(1, new byte[] { 0xAA, 0xBB, 0xCC });

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task EcuReset_HardReset_Succeeds()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x51, 0x01 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.EcuReset),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.EcuResetAsync(EcuResetType.HardReset);

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task TesterPresent_KeepsSessionAlive()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x7E, 0x00 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.Is<byte[]>(b => b[0] == UdsServiceId.TesterPresent),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        // Act
        var response = await _service.TesterPresentAsync(suppressPositiveResponse: false);

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public async Task ResponsePending_WaitsForFinalResponse()
    {
        // Arrange
        var pendingResponse = new byte[] { 0x7F, 0x22, 0x78 }; // Response pending
        var finalResponse = new byte[] { 0x62, 0xF1, 0x90, 0x41 }; // Final positive

        var callCount = 0;
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.IsAny<byte[]>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(() =>
            {
                callCount++;
                return callCount == 1 ? pendingResponse : finalResponse;
            });

        // Act
        var response = await _service.ReadDataByIdentifierAsync(0xF190);

        // Assert
        Assert.True(response.IsPositive);
    }

    [Fact]
    public void SessionChanged_EventFires_OnSessionChange()
    {
        // Arrange
        var positiveResponse = new byte[] { 0x50, 0x03 };
        _mockAdapter.Setup(a => a.SendMessageAsync(
            It.IsAny<byte[]>(),
            It.IsAny<int>(),
            It.IsAny<CancellationToken>()))
            .ReturnsAsync(positiveResponse);

        UdsSessionChangedEventArgs? eventArgs = null;
        _service.SessionChanged += (sender, args) => eventArgs = args;

        // Act
        _service.DiagnosticSessionControlAsync(UdsSession.ExtendedDiagnostic).Wait();

        // Assert
        Assert.NotNull(eventArgs);
        Assert.Equal(UdsSession.Default, eventArgs.PreviousSession);
        Assert.Equal(UdsSession.ExtendedDiagnostic, eventArgs.NewSession);
    }
}
