using System.Windows;
using HOPE.Core.Services.OBD;
using HOPE.Core.Services.Database;
using HOPE.Core.Protocols;
using HOPE.Core.Services.ECU;
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
        
        // For development, we'll use the Mock service
        // containerRegistry.RegisterSingleton<IOBD2Service, OBD2Service>();
        containerRegistry.RegisterSingleton<IOBD2Service, MockOBD2Service>();

        // Register Views for Navigation
        containerRegistry.RegisterForNavigation<DashboardView>();
        containerRegistry.RegisterForNavigation<MapVisualizationView>();
    }

    protected override async void OnInitialized()
    {
        base.OnInitialized();

        // Initialize Database
        var dbService = Container.Resolve<IDatabaseService>();
        await dbService.InitializeAsync();

        var regionManager = Container.Resolve<Prism.Regions.IRegionManager>();
        regionManager.RequestNavigate("MainRegion", "DashboardView");
    }
}

