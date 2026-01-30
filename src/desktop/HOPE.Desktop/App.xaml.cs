using System.IO;
using System.Windows;
using System.Net.Http;
using HOPE.Core.Services.Marketplace;
using HOPE.Core.Interfaces;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Protocols;
using HOPE.Core.Services.Protocols;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.AI;
using HOPE.Core.Services.Security;
using HOPE.Core.Security;
using HOPE.Core.Hardware;
using HOPE.Core.Services.Reports;
using HOPE.Core.Services.Export;
using HOPE.Core.Services.Logging;
using HOPE.Core.Services.Infra;
using HOPE.Core.Services.Safety;
using HOPE.Core.Services.Simulation;
using HOPE.Core.Services.Community;
using HOPE.Core.Services.Cloud;
using HOPE.Desktop.ViewModels;
using HOPE.Desktop.Views;
using Prism.DryIoc;
using Prism.Ioc;
using Sentry;
using Squirrel;

namespace HOPE.Desktop;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : PrismApplication
{
    protected override void OnStartup(StartupEventArgs e)
    {
        try 
        {
            // Initialize Sentry
            // SentrySdk.Init(options =>
            // {
            //     options.Dsn = ""; // Disable Sentry
            //     options.TracesSampleRate = 1.0;
            // });

            // Handle Squirrel Installation Events
            // SquirrelAwareApp.HandleEvents(
            //     onInitialInstall: (v, t) => t.CreateShortcutForThisExe(),
            //     onAppUpdate: (v, t) => t.CreateShortcutForThisExe(),
            //     onAppUninstall: (v, t) => t.RemoveShortcutForThisExe()
            // );

            // Check for updates in background
            Task.Run(async () => await CheckForUpdatesAsync());

            // Global Exception Handling
            DispatcherUnhandledException += OnDispatcherUnhandledException;
            TaskScheduler.UnobservedTaskException += OnUnobservedTaskException;
            AppDomain.CurrentDomain.UnhandledException += OnUnhandledException;

            base.OnStartup(e);
        }
        catch (Exception ex)
        {
             File.WriteAllText(@"c:\Users\Test\Documents\H.O.P.E\desktop_startup_error.txt", ex.ToString());
             MessageBox.Show($"Startup failed: {ex.Message}", "Fatal Error");
             Shutdown();
        }
    }

    protected override Window CreateShell()
    {
        try
        {
            return Container.Resolve<MainWindow>();
        }
        catch (Exception ex)
        {
            File.AppendAllText(@"c:\Users\Test\Documents\H.O.P.E\desktop_startup_error.txt", $"CreateShell Error: {ex}\n");
            throw;
        }
    }

    protected override void RegisterTypes(IContainerRegistry containerRegistry)
    {
        // Register Logging First
        containerRegistry.RegisterSingleton<ILoggingService, SerilogLoggingService>();
        
        // Register Services
        containerRegistry.RegisterSingleton<IDatabaseService, SqliteDatabaseService>();
        containerRegistry.RegisterSingleton<IVoltageMonitor, VoltageMonitor>();
        containerRegistry.RegisterSingleton<PreFlightService>();
        containerRegistry.RegisterSingleton<ISsoService, SsoService>();
        containerRegistry.RegisterSingleton<IFingerprintService, HardwareFingerprintService>();
        containerRegistry.RegisterSingleton<HttpClient>();
        
        // AI Services - use ONNX-based anomaly detection for production
        containerRegistry.RegisterSingleton<IAnomalyService, OnnxAnomalyService>();

        // AI & Report Services for Analytics & Diagnosis
        containerRegistry.RegisterSingleton<ILlmService, MockLlmService>();
        containerRegistry.RegisterSingleton<IReportService, GenerativeReportService>();
        containerRegistry.RegisterSingleton<DiagnosticNarrativeService>();
        containerRegistry.RegisterSingleton<ExplainableAnomalyService>();

        // RUL Predictor Service - remaining useful life prediction
        containerRegistry.RegisterSingleton<IRULPredictorService, RULPredictorService>();
        
        // Community & Wiki-Fix
        containerRegistry.RegisterSingleton<IWikiFixService, WikiFixService>();

        // Real OBD2 Hardware Interface (Generic ELM327)
        containerRegistry.RegisterSingleton<IOBD2Service, OBD2Service>();

        // Register Views for Navigation
        containerRegistry.RegisterForNavigation<DashboardView>();
        containerRegistry.RegisterForNavigation<DTCView>();
        containerRegistry.RegisterForNavigation<SettingsView>();
        containerRegistry.RegisterForNavigation<SessionHistoryView>();
        
        // Wiki-Fix Navigation
        containerRegistry.Register<WikiFixViewModel>();
        containerRegistry.RegisterForNavigation<WikiFixView, WikiFixViewModel>();
        containerRegistry.RegisterForNavigation<LoginView, LoginViewModel>();
    }

    protected override async void OnInitialized()
    {
        try
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
            regionManager.RequestNavigate("MainRegion", "LoginView");
        }
        catch (Exception ex)
        {
            File.AppendAllText(@"c:\Users\Test\Documents\H.O.P.E\desktop_startup_error.txt", $"OnInitialized Error: {ex}\n");
            MessageBox.Show($"Initialization failed: {ex.Message}", "Fatal Error");
        }
    }
    private void OnDispatcherUnhandledException(object sender, System.Windows.Threading.DispatcherUnhandledExceptionEventArgs e)
    {
        SentrySdk.CaptureException(e.Exception);
        // Optional: Resolve logger and log locally
        // var logger = Container.Resolve<ILoggingService>();
        // logger.Error("Unhandled Dispatcher Exception", e.Exception);
        
        // Prevent default crash if possible
        e.Handled = true;
        MessageBox.Show($"An unexpected error occurred: {e.Exception.Message}\n\nThe application will attempt to continue.", "Unexpected Error", MessageBoxButton.OK, MessageBoxImage.Error);
    }

    private void OnUnobservedTaskException(object? sender, UnobservedTaskExceptionEventArgs e)
    {
        SentrySdk.CaptureException(e.Exception);
        e.SetObserved(); // Prevent process termination
    }

    private void OnUnhandledException(object sender, UnhandledExceptionEventArgs e)
    {
        if (e.ExceptionObject is Exception ex)
        {
            SentrySdk.CaptureException(ex);
            SentrySdk.FlushAsync(TimeSpan.FromSeconds(2)).Wait();
        }
    }

    private async Task CheckForUpdatesAsync()
    {
        try
        {
            // TODO: Replace with real update URL (e.g. S3 bucket or GitHub Releases)
            string updateUrl = "https://github.com/Hopetuning/HOPE"; 
            
            using var mgr = await UpdateManager.GitHubUpdateManager(updateUrl);
            var releaseEntry = await mgr.UpdateApp();
            
            if (releaseEntry != null)
            {
                // Optionally notify user
            }
        }
        catch (Exception ex)
        {
            // Log update checks failures nicely (updates often fail due to network, ignore noisy errors)
            SentrySdk.CaptureException(ex);
        }
    }
}

