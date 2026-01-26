using System.IO;
using System.Windows;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Protocols;
using HOPE.Core.Services.ECU;
using HOPE.Core.Services.AI;
using HOPE.Core.Services.Export;
using HOPE.Desktop.Views;
using Prism.DryIoc;
using Prism.Ioc;

namespace HOPE.Desktop;

/// <summary>
/// Interaction logic for App.xaml
/// </summary>
public partial class App : PrismApplication
{
    protected override Window CreateShell()
    {
        return Container.Resolve<MainWindow>();
    }

    protected override void RegisterTypes(IContainerRegistry containerRegistry)
    {
        // Register Services
        containerRegistry.RegisterSingleton<IDatabaseService, SqliteDatabaseService>();
        
        // Protocol & ECU
        containerRegistry.RegisterSingleton<IDiagnosticProtocol, UDSProtocol>();
        containerRegistry.RegisterSingleton<IECUService, ECUService>();
        
        // AI Services - use ONNX-based anomaly detection for production
        containerRegistry.RegisterSingleton<IAnomalyService, OnnxAnomalyService>();
        
        // Export Services
        containerRegistry.RegisterSingleton<IExportService, ExportService>();
        
        // For development, we'll use the Mock service
        // containerRegistry.RegisterSingleton<IOBD2Service, OBD2Service>();
        containerRegistry.RegisterSingleton<IOBD2Service, MockOBD2Service>();

        // Register Views for Navigation
        containerRegistry.RegisterForNavigation<DashboardView>();
        containerRegistry.RegisterForNavigation<MapVisualizationView>();
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

