using System;
using System.Collections.ObjectModel;
using System.Windows.Input;
using HOPE.Core.Services.Marketplace;
using Prism.Commands;
using Prism.Mvvm;

namespace HOPE.Desktop.ViewModels
{
    public class MarketplaceViewModel : BindableBase
    {
        private readonly IMarketplaceService _marketplaceService;
        private bool _isLoading;

        public MarketplaceViewModel(IMarketplaceService marketplaceService)
        {
            _marketplaceService = marketplaceService;
            Listings = new ObservableCollection<ListingItem>();
            LoadCommand = new DelegateCommand(async () => await LoadListings());
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

        private async Task LoadListings()
        {
            IsLoading = true;
            try
            {
                // In a real app, fetch from IMarketplaceService
                // Mocking for now
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
            }
            finally
            {
                IsLoading = false;
            }
        }

        private async Task PurchaseListing(ListingItem item)
        {
            if (item == null) return;
            var success = await _marketplaceService.PurchaseAndDownloadAsync(item.Id);
            if (success)
            {
                // Show success message or navigate
            }
        }
    }

    public class ListingItem
    {
        public string Id { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public double Price { get; set; }
        public string Version { get; set; }
    }
}
