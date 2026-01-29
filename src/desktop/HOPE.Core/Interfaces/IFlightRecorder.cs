using System;
using System.Threading.Tasks;
using HOPE.Core.Hardware;

namespace HOPE.Core.Interfaces;

/// <summary>
/// Interface for a black box flight recorder that logs CAN traffic in a circular buffer.
/// </summary>
public interface IFlightRecorder
{
    /// <summary>
    /// Starts recording traffic to the internal circular buffer.
    /// </summary>
    void StartRecording();

    /// <summary>
    /// Stops recording.
    /// </summary>
    void StopRecording();

    /// <summary>
    /// Dumps the current buffer to a file.
    /// </summary>
    Task DumpToFileAsync(string path);

    /// <summary>
    /// Observable that fires when the recorder triggers an auto-dump (e.g., on error).
    /// </summary>
    IObservable<HardwareErrorEventArgs> TriggerEvents { get; }
}
