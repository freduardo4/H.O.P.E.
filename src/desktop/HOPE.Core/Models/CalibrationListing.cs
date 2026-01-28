using System;

namespace HOPE.Core.Models
{
    public class CalibrationListing
    {
        public string Id { get; set; }
        public string Title { get; set; }
        public string Description { get; set; }
        public double Price { get; set; }
        public string Version { get; set; }
        public string FileUrl { get; set; }
    }
}
