using System;
using System.Collections.Generic;

namespace HOPE.Core.Models;

public class WikiPost
{
    public Guid Id { get; set; }
    public string Title { get; set; } = string.Empty;
    public string Content { get; set; } = string.Empty;
    public List<string> Tags { get; set; } = new();
    public int Upvotes { get; set; }
    public int Downvotes { get; set; }
    public string AuthorName { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public List<RepairPattern> Patterns { get; set; } = new();
}

public class RepairPattern
{
    public Guid Id { get; set; }
    public Guid PostId { get; set; }
    public string DTC { get; set; } = string.Empty;
    public double? MinAnomalyScore { get; set; }
    public double ConfidenceScore { get; set; }
}
