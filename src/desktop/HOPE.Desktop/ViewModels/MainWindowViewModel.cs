using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Prism.Regions;

namespace HOPE.Desktop.ViewModels;

public partial class MainWindowViewModel : ObservableObject
{
    private readonly IRegionManager _regionManager;

    public MainWindowViewModel(IRegionManager regionManager)
    {
        _regionManager = regionManager;
    }

    [RelayCommand]
    private void Navigate(string viewName)
    {
        _regionManager.RequestNavigate("MainRegion", viewName);
    }
}
