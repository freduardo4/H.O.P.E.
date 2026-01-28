using System.Windows.Input;
using System.Threading.Tasks;
using HOPE.Core.Services.Cloud;
using Prism.Commands;
using Prism.Mvvm;
using Prism.Regions;

namespace HOPE.Desktop.ViewModels
{
    public class LoginViewModel : BindableBase
    {
        private readonly ISsoService _ssoService;
        private readonly IRegionManager _regionManager;

        public LoginViewModel(ISsoService ssoService, IRegionManager regionManager)
        {
            _ssoService = ssoService;
            _regionManager = regionManager;

            LoginWithGoogleCommand = new DelegateCommand(async () => await LoginWithGoogle());
            LoginWithGithubCommand = new DelegateCommand(async () => await LoginWithGithub());
        }

        public ICommand LoginWithGoogleCommand { get; }
        public ICommand LoginWithGithubCommand { get; }

        private async Task LoginWithGoogle()
        {
            try
            {
                var response = await _ssoService.LoginWithGoogleAsync();
                if (response != null)
                {
                    NavigateToDashboard();
                }
            }
            catch (Exception ex)
            {
                // Log and show error
            }
        }

        private async Task LoginWithGithub()
        {
            try
            {
                var response = await _ssoService.LoginWithGithubAsync();
                if (response != null)
                {
                    NavigateToDashboard();
                }
            }
            catch (Exception ex)
            {
                // Log and show error
            }
        }

        private void NavigateToDashboard()
        {
            _regionManager.RequestNavigate("MainRegion", "DashboardView");
        }
    }
}
