using Xunit;
using Moq;
using HOPE.Desktop.ViewModels;
using HOPE.Core.Services.Marketplace;
using HOPE.Core.Models;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

namespace HOPE.Desktop.Tests.ViewModels
{
    public class MarketplaceViewModelTests
    {
        private readonly Mock<IMarketplaceService> _mockService;


        public MarketplaceViewModelTests()
        {
            _mockService = new Mock<IMarketplaceService>();
        }

        [Fact]
        public async Task LoadCommand_ShouldPopulateListings_WhenServiceReturnsItems()
        {
            // Arrange
            var mockListings = new List<CalibrationListing>
            {
                new CalibrationListing { Id = "1", Title = "Test Tune 1", Price = 50, Version = "1.0", Description = "Desc" },
                new CalibrationListing { Id = "2", Title = "Test Tune 2", Price = 100, Version = "2.0", Description = "Desc 2" }
            };

            _mockService.Setup(s => s.GetListingsAsync())
                .ReturnsAsync(mockListings);

            var viewModel = new MarketplaceViewModel(_mockService.Object);

            // Act
            // Direct async call to ensure completion for test
            await viewModel.LoadListingsAsync();
            
            // Wait for async execution (ViewModel commands are usually fire-and-forget in Prism, 
            // but this simple VM might need a small delay or check)
            // Ideally VM should expose a task or IsLoading check. 
            // Since it's DelegateCommand (Prism), it's sync from caller perspective but async inside.
            // We can wait a bit or inspect IsLoading.
            
            await Task.Delay(100); // Small consistency delay for the async void

            // Assert
            Assert.Equal(2, viewModel.Listings.Count);
            Assert.Equal("Test Tune 1", viewModel.Listings.First().Title);
            _mockService.Verify(s => s.GetListingsAsync(), Times.AtLeastOnce);
        }

        [Fact]
        public async Task PurchaseCommand_ShouldCallService_WhenExecuted()
        {
            // Arrange
            var item = new ListingItem { Id = "1", Title = "Tune" };
            _mockService.Setup(s => s.PurchaseAndDownloadAsync("1"))
                .ReturnsAsync(true);

            var viewModel = new MarketplaceViewModel(_mockService.Object);

            // Act
            viewModel.PurchaseCommand.Execute(item);
            await Task.Delay(100);

            // Assert
            _mockService.Verify(s => s.PurchaseAndDownloadAsync("1"), Times.Once);
        }
    }
}
