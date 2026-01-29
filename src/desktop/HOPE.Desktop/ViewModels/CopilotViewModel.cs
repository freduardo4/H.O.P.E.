using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using Prism.Commands;
using Prism.Mvvm;
using Prism.Regions;
using HOPE.Core.Services.AI;

namespace HOPE.Desktop.ViewModels
{
    public class CopilotViewModel : BindableBase, INavigationAware
    {
        private readonly ITuningCopilotService _copilotService;
        private readonly IRegionManager _regionManager;

        private string _currentQuery = string.Empty;
        public string CurrentQuery
        {
            get => _currentQuery;
            set => SetProperty(ref _currentQuery, value);
        }

        private bool _isBusy;
        public bool IsBusy
        {
            get => _isBusy;
            set => SetProperty(ref _isBusy, value);
        }

        public ObservableCollection<ChatMessage> ChatHistory { get; } = new ObservableCollection<ChatMessage>();

        public DelegateCommand SendQueryCommand { get; }

        public CopilotViewModel(ITuningCopilotService copilotService, IRegionManager regionManager)
        {
            _copilotService = copilotService;
            _regionManager = regionManager;

            SendQueryCommand = new DelegateCommand(async () => await SendQueryAsync(), () => !IsBusy)
                .ObservesProperty(() => IsBusy);

            // Initial Greeting
            ChatHistory.Add(new ChatMessage
            {
                IsUser = false,
                Message = "Hello! I'm your Tuning Copilot. Ask me anything about diagnostics, ECU maps, or protocols."
            });
        }

        private async Task SendQueryAsync()
        {
            if (string.IsNullOrWhiteSpace(CurrentQuery)) return;

            var userText = CurrentQuery;
            CurrentQuery = string.Empty; // Clear input

            // Add User Message
            ChatHistory.Add(new ChatMessage { IsUser = true, Message = userText });
            IsBusy = true;

            try
            {
                var response = await _copilotService.AskAsync(userText);

                // Add Bot Message
                ChatHistory.Add(new ChatMessage { IsUser = false, Message = response.Answer });
            }
            catch (Exception ex)
            {
                ChatHistory.Add(new ChatMessage { IsUser = false, Message = $"Error: {ex.Message}" });
            }
            finally
            {
                IsBusy = false;
            }
        }

        public void OnNavigatedTo(NavigationContext navigationContext)
        {
        }

        public bool IsNavigationTarget(NavigationContext navigationContext)
        {
            return true;
        }

        public void OnNavigatedFrom(NavigationContext navigationContext)
        {
        }
    }

    public class ChatMessage
    {
        public bool IsUser { get; set; }
        public string Message { get; set; } = string.Empty;
        public DateTime Timestamp { get; } = DateTime.Now;
    }
}
