using Serilog;
using System;
using System.IO;
using System.Collections.Generic;

namespace HOPE.Core.Services.Logging;

public interface ILoggingService
{
    void Information(string message);
    void Warning(string message);
    void Error(string message, Exception? ex = null);
}

public class SerilogLoggingService : ILoggingService
{
    public SerilogLoggingService()
    {
        var logPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HOPE", "logs", "log-.txt");
        
        Log.Logger = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.Console()
            .WriteTo.File(logPath, rollingInterval: RollingInterval.Day)
            .CreateLogger();
    }

    public void Information(string message) => Log.Information(message);
    public void Warning(string message) => Log.Warning(message);
    public void Error(string message, Exception? ex = null) => Log.Error(ex, message);
}
