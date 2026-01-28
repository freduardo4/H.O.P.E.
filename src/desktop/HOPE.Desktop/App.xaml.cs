using System.IO;
using System.Windows;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Protocols;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.AI;
using HOPE.Core.Services.Export;
using HOPE.Core.Services.Logging;
using HOPE.Core.Services.Infra;
using HOPE.Core.Services.Safety;
using HOPE.Desktop.Views;
using Prism.DryIoc;
using Prism.Ioc;
using Sentry;

namespace HOPE.Desktop;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : PrismApplication
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // Initialize Sentry
        SentrySdk.Init(options =>
        {
            options.Dsn = "https://example@sentry.io/123"; // Replace with real DSN
            options.Debug = true;
            options.TracesSampleRate = 1.0;
        });

        base.OnStartup(e);
    }

    protected override Window CreateShell()
    {
        return Container.Resolve<MainWindow>();
    }

    protected override void RegisterTypes(IContainerRegistry containerRegistry)
    {
        // Register Logging First
        containerRegistry.RegisterSingleton<ILoggingService, SerilogLoggingService>();
        
        // Register Services
        containerRegistry.RegisterSingleton<IDatabaseService, SqliteDatabaseService>();
        containerRegistry.RegisterSingleton<CloudSafetyService>();
        containerRegistry.RegisterSingleton<PreFlightService>();
        
        // Protocol & ECU
        containerRegistry.RegisterSingleton<IDiagnosticProtocol, UDSProtocol>();
        containerRegistry.RegisterSingleton<IECUService, ECUService>();
        
        // AI Services - use ONNX-based anomaly detection for production
        containerRegistry.RegisterSingleton<IAnomalyService, OnnxAnomalyService>();

        // Tuning Optimizer Service - genetic algorithm-based ECU tuning
        containerRegistry.RegisterSingleton<ITuningOptimizerService, TuningOptimizerService>();

        // RUL Predictor Service - remaining useful life prediction
        containerRegistry.RegisterSingleton<IRULPredictorService, RULPredictorService>();
        
        // Export Services
        containerRegistry.RegisterSingleton<IExportService, ExportService>();
        
        // For development, we'll use the Mock service
        // containerRegistry.RegisterSingleton<IOBD2Service, OBD2Service>();
        containerRegistry.RegisterSingleton<IOBD2Service, MockOBD2Service>();

        // Register Calibration Repository & Backup
        var repoPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "HOPE", "CalibrationRepo");
        containerRegistry.RegisterInstance(new CalibrationRepository(repoPath));
        containerRegistry.RegisterSingleton<IBackupService>(() => new BackupService(repoPath));

        // Register Views for Navigation
        containerRegistry.RegisterForNavigation<DashboardView>();
        containerRegistry.RegisterForNavigation<MapVisualizationView>();
        containerRegistry.RegisterForNavigation<MapDiffViewer>();
        containerRegistry.RegisterForNavigation<DTCView>();
        containerRegistry.RegisterForNavigation<SettingsView>();
        containerRegistry.RegisterForNavigation<SessionHistoryView>();
    }

    protected override async void OnInitialized()
    {
        base.OnInitialized();

        // Initialize Database
        var dbService = Container.Resolve<IDatabaseService>();
        await dbService.InitializeAsync();

        // Load ONNX anomaly detection model
        var anomalyService = Container.Resolve<IAnomalyService>();
        if (anomalyService is OnnxAnomalyService onnxService)
        {
            var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "anomaly_detector.onnx");
            if (File.Exists(modelPath))
            {
                await onnxService.LoadModelAsync(modelPath);
            }
        }

        var regionManager = Container.Resolve<Prism.Regions.IRegionManager>();
        regionManager.RequestNavigate("MainRegion", "DashboardView");
    }
}

