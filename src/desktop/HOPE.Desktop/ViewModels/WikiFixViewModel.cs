using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using HOPE.Core.Models;
using HOPE.Core.Services.Community;
using System.Collections.ObjectModel;

namespace HOPE.Desktop.ViewModels;

public partial class WikiFixViewModel : ObservableObject
{
    private readonly IWikiFixService _wikiFixService;

    [ObservableProperty]
    private ObservableCollection<WikiPost> _posts = new();

    [ObservableProperty]
    private ObservableCollection<KnowledgeNode> _relatedNodes = new();

    [ObservableProperty]
    private string _searchQuery = string.Empty;

    [ObservableProperty]
    private string _dtcFilter = string.Empty;

    [ObservableProperty]
    private bool _isLoading;

    [ObservableProperty]
    private WikiPost? _selectedPost;

    public WikiFixViewModel(IWikiFixService wikiFixService)
    {
        _wikiFixService = wikiFixService;
    }

    [RelayCommand]
    public async Task SearchAsync()
    {
        IsLoading = true;
        try
        {
            var results = await _wikiFixService.SearchPostsAsync(SearchQuery, DtcFilter);
            Posts = new ObservableCollection<WikiPost>(results);

            // Also search knowledge graph
            if (!string.IsNullOrEmpty(SearchQuery) || !string.IsNullOrEmpty(DtcFilter))
            {
                var graphResults = await _wikiFixService.SearchKnowledgeAsync(string.IsNullOrEmpty(DtcFilter) ? SearchQuery : DtcFilter);
                RelatedNodes = new ObservableCollection<KnowledgeNode>(graphResults);
            }
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand]
    public async Task UpvoteAsync(WikiPost post)
    {
        if (post == null) return;
        
        await _wikiFixService.UpvotePostAsync(post.Id);
        post.Upvotes++; // Optimistic UI
    }

    public async Task LoadDtcFixesAsync(string dtc)
    {
        DtcFilter = dtc;
        SearchQuery = string.Empty;
        await SearchAsync();
    }
}
