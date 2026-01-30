using System.Windows.Input;
using System.Threading.Tasks;
using HOPE.Core.Services.Cloud;
using Prism.Commands;
using Prism.Mvvm;
using Prism.Regions;
using Prism.Events;
using HOPE.Desktop.Events;

namespace HOPE.Desktop.ViewModels
{
    public class LoginViewModel : BindableBase
    {
        private readonly ISsoService _ssoService;
        private readonly IRegionManager _regionManager;
        private readonly IEventAggregator _eventAggregator;

        private string _email;
        public string Email
        {
            get => _email;
            set => SetProperty(ref _email, value);
        }

        public LoginViewModel(ISsoService ssoService, IRegionManager regionManager, IEventAggregator eventAggregator)
        {
            _ssoService = ssoService;
            _regionManager = regionManager;
            _eventAggregator = eventAggregator;

            LoginWithGoogleCommand = new DelegateCommand(async () => await LoginWithGoogle());
            LoginWithGithubCommand = new DelegateCommand(async () => await LoginWithGithub());
            LoginWithEmailCommand = new DelegateCommand<object>(LoginWithEmail);
        }

        public ICommand LoginWithEmailCommand { get; }

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

        private void LoginWithEmail(object passwordBox)
        {
            // Dummy authentication
            // In a real app, bind PasswordBox using an Attached Property or pass it as here
            var password = (passwordBox as System.Windows.Controls.PasswordBox)?.Password;

            if (string.IsNullOrWhiteSpace(Email) || string.IsNullOrWhiteSpace(password))
            {
                System.Windows.MessageBox.Show("Please enter email and password.", "Login Failed");
                return;
            }

            // Dummy Account Check
            if (Email == "admin" && password == "admin") 
            {
                NavigateToDashboard();
            }
            else
            {
                 System.Windows.MessageBox.Show("Invalid Credentials. Try 'admin' / 'admin'", "Login Failed");
            }
        }

        private void NavigateToDashboard()
        {
            _eventAggregator.GetEvent<UserLoggedInEvent>().Publish("User");
            _regionManager.RequestNavigate("MainRegion", "DashboardView");
        }
    }
}
