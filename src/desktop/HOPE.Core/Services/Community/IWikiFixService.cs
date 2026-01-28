using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using HOPE.Core.Models;

namespace HOPE.Core.Services.Community;

public interface IWikiFixService
{
    Task<IEnumerable<WikiPost>> SearchPostsAsync(string query = "", string dtc = "");
    Task<WikiPost?> GetPostAsync(Guid id);
    Task UpvotePostAsync(Guid id);

    // Knowledge Graph
    Task<IEnumerable<KnowledgeNode>> SearchKnowledgeAsync(string query);
    Task<IEnumerable<KnowledgeNode>> GetRelatedNodesAsync(Guid nodeId);
}
