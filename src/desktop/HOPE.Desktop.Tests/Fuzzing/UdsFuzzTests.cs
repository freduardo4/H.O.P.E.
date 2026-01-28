using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HOPE.Core.Interfaces;
using HOPE.Core.Models;
using HOPE.Core.Services.Protocols;
using Moq;
using Xunit;

namespace HOPE.Desktop.Tests.Fuzzing;

public class UdsFuzzTests
{
    private readonly Mock<IHardwareAdapter> _adapterMock;
    private readonly UdsProtocolService _service;
    private readonly Random _random;

    public UdsFuzzTests()
    {
        _adapterMock = new Mock<IHardwareAdapter>();
        _service = new UdsProtocolService(_adapterMock.Object);
        _random = new Random(12345); // Fixed seed for reproducibility
    }

    [Fact]
    public async Task Fuzz_ReadMultipleDataByIdentifier_ShouldNotCrash()
    {
        // Fuzzing parameters
        int iterations = 1000;
        var dids = new ushort[] { 0xF190, 0xF187, 0xF189 };

        for (int i = 0; i < iterations; i++)
        {
            // Generate random response
            int length = _random.Next(0, 100);
            byte[] fuzzData = new byte[length];
            _random.NextBytes(fuzzData);

            // Ensure first byte indicates positive response if we want to test parsing logic
            if (length > 0 && _random.NextDouble() > 0.1) 
            {
                fuzzData[0] = 0x62; // Positive response for ReadDataByIdentifier
            }

            _adapterMock.Reset();
            _adapterMock.Setup(x => x.SendMessageAsync(It.IsAny<byte[]>(), It.IsAny<int>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(fuzzData);

            try
            {
                await _service.ReadMultipleDataByIdentifierAsync(dids);
            }
            catch (Exception ex)
            {
                // We expect UdsException or potentially generic exceptions handled gracefully.
                // We DO NOT want crashes (IndexOutOfRange, NullReference, etc. unhandled).
                // Ideally, the service should wrap all parsing errors.
                // For now, let's allow it, but failing on unexpected system exceptions would be the goal of a strict fuzzer.
                
                // Assert that exception is not critical
                Assert.False(ex is IndexOutOfRangeException, $"IndexOutOfRangeException at iteration {i}. Data: {BitConverter.ToString(fuzzData)}");
                Assert.False(ex is NullReferenceException, $"NullReferenceException at iteration {i}. Data: {BitConverter.ToString(fuzzData)}");
            }
        }
    }

    [Fact]
    public async Task Fuzz_ReadDtcInformation_ShouldNotCrash()
    {
        int iterations = 1000;

        for (int i = 0; i < iterations; i++)
        {
            int length = _random.Next(0, 50);
            byte[] fuzzData = new byte[length];
            _random.NextBytes(fuzzData);

            if (length > 0 && _random.NextDouble() > 0.1)
            {
                fuzzData[0] = 0x59; // Positive response checking
            }

            _adapterMock.Reset();
            _adapterMock.Setup(x => x.SendMessageAsync(It.IsAny<byte[]>(), It.IsAny<int>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(fuzzData);

            try
            {
                await _service.ReadDtcInformationAsync(DtcReportType.ReportDTCByStatusMask);
            }
            catch (Exception ex)
            {
                Assert.False(ex is IndexOutOfRangeException, $"IndexOutOfRangeException at iteration {i}. Data: {BitConverter.ToString(fuzzData)}");
                Assert.False(ex is NullReferenceException, $"NullReferenceException at iteration {i}. Data: {BitConverter.ToString(fuzzData)}");
            }
        }
    }
}
