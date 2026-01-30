using System;
using System.IO;
using System.Threading.Tasks;
using Apache.Arrow;
using Apache.Arrow.Ipc;
using Apache.Arrow.Types;

namespace HOPE.Core.Services.Infra;

public class ArrowDataService
{
    public async Task ExportSensorDataAsync(string filePath, float[] timestamps, float[] values)
    {
        // Define Schema
        var schema = new Schema.Builder()
            .Field(new Field("timestamp", new FloatType(), false))
            .Field(new Field("value", new FloatType(), false))
            .Build();

        // Create Record Batch
        int length = timestamps.Length;
        var timestampData = new FloatArray.Builder().Append(timestamps).Build();
        var valueData = new FloatArray.Builder().Append(values).Build();

        var recordBatch = new RecordBatch(schema, [timestampData, valueData], length);

        try
        {
            using var stream = File.OpenWrite(filePath);
            var writer = new ArrowFileWriter(stream, schema);
            await writer.WriteRecordBatchAsync(recordBatch);
            await writer.WriteEndAsync();
        }
        catch (Exception ex)
        {
            throw new IOException($"Failed to write Arrow file: {ex.Message}", ex);
        }
    }
}
