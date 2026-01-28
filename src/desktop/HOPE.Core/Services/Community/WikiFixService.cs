using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading.Tasks;
using HOPE.Core.Models;
using HOPE.Core.Services.Logging;

namespace HOPE.Core.Services.Community;

public class WikiFixService : IWikiFixService
{
    private readonly HttpClient _httpClient;
    private readonly ILoggingService _logger;

    public WikiFixService(HttpClient httpClient, ILoggingService logger)
    {
        _httpClient = httpClient;
        _logger = logger;
    }

    public async Task<IEnumerable<WikiPost>> SearchPostsAsync(string query = "", string dtc = "")
    {
        try
        {
            var url = $"/wiki-fix/posts?query={Uri.EscapeDataString(query)}&dtc={Uri.EscapeDataString(dtc)}";
            var response = await _httpClient.GetFromJsonAsync<WikiFixSearchResponse>(url);
            return response?.Items ?? new List<WikiPost>();
        }
        catch (Exception ex)
        {
            _logger.Error("Failed to search Wiki-Fix posts", ex);
            return new List<WikiPost>();
        }
    }

    public async Task<WikiPost?> GetPostAsync(Guid id)
    {
        try
        {
            return await _httpClient.GetFromJsonAsync<WikiPost>($"/wiki-fix/posts/{id}");
        }
        catch (Exception ex)
        {
            _logger.Error($"Failed to fetch Wiki-Fix post {id}", ex);
            return null;
        }
    }

    public async Task UpvotePostAsync(Guid id)
    {
        try
        {
            await _httpClient.PostAsync($"/wiki-fix/posts/{id}/upvote", null);
        }
        catch (Exception ex)
        {
            _logger.Error($"Failed to upvote Wiki-Fix post {id}", ex);
        }
    }

    public async Task<IEnumerable<KnowledgeNode>> SearchKnowledgeAsync(string query)
    {
        try
        {
            var graphQuery = new
            {
                query = @"query Search($q: String!) {
                    searchKnowledge(query: $q) {
                        id
                        name
                        type
                        description
                    }
                }",
                variables = new { q = query }
            };

            var response = await _httpClient.PostAsJsonAsync("/graphql", graphQuery);
            if (!response.IsSuccessStatusCode) return new List<KnowledgeNode>();

            var result = await response.Content.ReadFromJsonAsync<GraphResponse<List<KnowledgeNode>>>();
            return result?.Data?["searchKnowledge"] ?? new List<KnowledgeNode>();
        }
        catch (Exception ex)
        {
            _logger.Error("Failed to search knowledge graph", ex);
            return new List<KnowledgeNode>();
        }
    }

    public async Task<IEnumerable<KnowledgeNode>> GetRelatedNodesAsync(Guid nodeId)
    {
        try
        {
            var graphQuery = new
            {
                query = @"query Related($id: ID!) {
                    relatedFixes(nodeId: $id) {
                        id
                        name
                        type
                        description
                    }
                }",
                variables = new { id = nodeId }
            };

            var response = await _httpClient.PostAsJsonAsync("/graphql", graphQuery);
            if (!response.IsSuccessStatusCode) return new List<KnowledgeNode>();

            var result = await response.Content.ReadFromJsonAsync<GraphResponse<List<KnowledgeNode>>>();
            return result?.Data?["relatedFixes"] ?? new List<KnowledgeNode>();
        }
        catch (Exception ex)
        {
            _logger.Error($"Failed to get related nodes for {nodeId}", ex);
            return new List<KnowledgeNode>();
        }
    }

    private class GraphResponse<T>
    {
        public Dictionary<string, T> Data { get; set; } = new();
    }

    private class WikiFixSearchResponse
    {
        public List<WikiPost> Items { get; set; } = new();
        public int Total { get; set; }
    }
}
