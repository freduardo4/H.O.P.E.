using Prism.Events;
using Prism.Regions;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Desktop.Events;

namespace HOPE.Desktop.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly IRegionManager _regionManager;

    [ObservableProperty]
    private bool _isSidebarVisible = false; // Hide by default

    public MainWindowViewModel(IRegionManager regionManager, IEventAggregator eventAggregator)
    {
        _regionManager = regionManager;
        
        eventAggregator.GetEvent<UserLoggedInEvent>().Subscribe(OnUserLoggedIn);
    }

    private void OnUserLoggedIn(string username)
    {
        IsSidebarVisible = true;
    }

    [RelayCommand]
    private void Navigate(string viewName)
    {
        _regionManager.RequestNavigate("MainRegion", viewName);
    }
}
