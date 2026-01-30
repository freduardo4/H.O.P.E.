using System;
using System.Collections.ObjectModel;
using System.Windows.Input;
using HOPE.Core.Services.Marketplace;
using Prism.Commands;
using Prism.Mvvm;

using Microsoft.Extensions.Logging;

namespace HOPE.Desktop.ViewModels
{
    public class MarketplaceViewModel : BindableBase
    {
        private readonly IMarketplaceService _marketplaceService;
        private readonly ILogger<MarketplaceViewModel> _logger;
        private bool _isLoading;

        public MarketplaceViewModel(IMarketplaceService marketplaceService, ILogger<MarketplaceViewModel> logger)
        {
            _marketplaceService = marketplaceService;
            _logger = logger;
            Listings = new ObservableCollection<ListingItem>();
            LoadCommand = new DelegateCommand(async () => await LoadListingsAsync());
            PurchaseCommand = new DelegateCommand<ListingItem>(async (item) => await PurchaseListing(item));

            // Load initial data
            LoadCommand.Execute(null);
        }

        public ObservableCollection<ListingItem> Listings { get; }
        public ICommand LoadCommand { get; }
        public ICommand PurchaseCommand { get; }

        public bool IsLoading
        {
            get => _isLoading;
            set => SetProperty(ref _isLoading, value);
        }

        public async Task LoadListingsAsync()
        {
            IsLoading = true;
            try
            {
                _logger.LogInformation("Loading marketplace listings...");
                Listings.Clear();
                var items = await _marketplaceService.GetListingsAsync();
                foreach (var item in items)
                {
                    Listings.Add(new ListingItem
                    {
                        Id = item.Id,
                        Title = item.Title,
                        Description = item.Description,
                        Price = item.Price,
                        Version = item.Version
                    });
                }
                _logger.LogInformation("Loaded {Count} listings.", items.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to load listings");
            }
            finally
            {
                IsLoading = false;
            }
        }

        private async Task PurchaseListing(ListingItem item)
        {
            if (item == null) return;
            
            try 
            {
                _logger.LogInformation("Purchasing listing {ListingId}", item.Id);
                var success = await _marketplaceService.PurchaseAndDownloadAsync(item.Id);
                if (success)
                {
                    _logger.LogInformation("Purchase successful for {ListingId}", item.Id);
                    // Show success message or navigate
                }
                else
                {
                    _logger.LogWarning("Purchase failed for {ListingId}", item.Id);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error purchasing listing {ListingId}", item.Id);
            }
        }
    }

    public class ListingItem
    {
        public required string Id { get; set; }
        public required string Title { get; set; }
        public required string Description { get; set; }
        public double Price { get; set; }
        public required string Version { get; set; }
    }
}
