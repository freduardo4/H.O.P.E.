using HOPE.Core.Models;
using HOPE.Core.Protocols;

namespace HOPE.Core.Services.ECU;

public class ECUService : IECUService
{
    private readonly IDiagnosticProtocol _protocol;
    private readonly Random _random = new();

    public ECUService(IDiagnosticProtocol protocol)
    {
        _protocol = protocol;
    }

    public async Task<ECUCalibration> ReadCalibrationAsync(CancellationToken cancellationToken = default)
    {
        // 1. Start Diagnostic Session
        await _protocol.StartSessionAsync(0x03); // Extended Session
        
        // 2. Unlock ECU
        await _protocol.SecurityAccessAsync(0x01);

        // 3. Read some dummy address for the demo
        byte[] data = await _protocol.ReadMemoryAsync(0x1000, 1024);

        return new ECUCalibration
        {
            CalibrationId = Guid.NewGuid(),
            FileSizeBytes = data.Length,
            Version = "Stock (V1.0)",
            UploadedAt = DateTime.UtcNow,
            IsChecksumValid = true
        };
    }

    public async Task<double[,]> ReadMapAsync(string mapName, CancellationToken cancellationToken = default)
    {
        // For the demo, return a mock 8x8 fuel map
        int rows = 8;
        int cols = 8;
        double[,] map = new double[rows, cols];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // Simulate some base fuel values
                map[i, j] = 12.5 + (i * 0.5) - (j * 0.2) + (_random.NextDouble() * 0.2);
            }
        }

        return map;
    }

    public async Task<bool> WriteValueAsync(long address, double value, CancellationToken cancellationToken = default)
    {
        // Simulate successful write
        await Task.Delay(100, cancellationToken);
        return true;
    }
}
