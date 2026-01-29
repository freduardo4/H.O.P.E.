using System;
using System.Threading.Tasks;
using HOPE.Core.Services.Cloud;
using HOPE.Desktop.ViewModels;
using Moq;
using Prism.Regions;
using Xunit;

namespace HOPE.Desktop.Tests.ViewModels
{
    public class LoginViewModelTests
    {
        private readonly Mock<ISsoService> _mockSsoService;
        private readonly Mock<IRegionManager> _mockRegionManager;
        private readonly LoginViewModel _viewModel;

        public LoginViewModelTests()
        {
            _mockSsoService = new Mock<ISsoService>();
            _mockRegionManager = new Mock<IRegionManager>();
            _viewModel = new LoginViewModel(_mockSsoService.Object, _mockRegionManager.Object);
        }

        [Fact]
        public async Task LoginWithGoogle_Success_NavigatesToDashboard()
        {
            // Arrange
            _mockSsoService.Setup(s => s.LoginWithGoogleAsync())
                .ReturnsAsync(new AuthResponse { AccessToken = "valid_token" });

            // Act
            _viewModel.LoginWithGoogleCommand.Execute(null);

            // Assert
            _mockRegionManager.Verify(r => r.RequestNavigate("MainRegion", "DashboardView"), Times.Once);
        }

        [Fact]
        public async Task LoginWithGithub_Success_NavigatesToDashboard()
        {
            // Arrange
            _mockSsoService.Setup(s => s.LoginWithGithubAsync())
                .ReturnsAsync(new AuthResponse { AccessToken = "valid_token" });

            // Act
            _viewModel.LoginWithGithubCommand.Execute(null);

            // Assert
            _mockRegionManager.Verify(r => r.RequestNavigate("MainRegion", "DashboardView"), Times.Once);
        }

        [Fact]
        public async Task LoginWithGoogle_Failure_DoesNotNavigate()
        {
            // Arrange
            _mockSsoService.Setup(s => s.LoginWithGoogleAsync())
                .ReturnsAsync((AuthResponse?)null);

            // Act
            _viewModel.LoginWithGoogleCommand.Execute(null);

            // Assert
            _mockRegionManager.Verify(r => r.RequestNavigate(It.IsAny<string>(), It.IsAny<string>()), Times.Never);
        }
    }
}
