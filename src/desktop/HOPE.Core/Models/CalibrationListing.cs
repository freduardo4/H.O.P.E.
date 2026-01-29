using System;

namespace HOPE.Core.Models
{
    public class CalibrationListing
    {
        public string Id { get; set; } = string.Empty;
        public string Title { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public double Price { get; set; }
        public string Version { get; set; } = string.Empty;
        public string FileUrl { get; set; } = string.Empty;
    }
}
