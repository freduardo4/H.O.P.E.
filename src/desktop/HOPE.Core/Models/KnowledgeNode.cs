using System;

namespace HOPE.Core.Models
{
    public enum NodeType
    {
        DTC,
        SYMPTOM,
        PART,
        REPAIR_STEP,
        VEHICLE_SYSTEM
    }

    public class KnowledgeNode
    {
        public Guid Id { get; set; }
        public string Name { get; set; } = string.Empty;
        public NodeType Type { get; set; }
        public string? Description { get; set; }
        public double RelevanceWeight { get; set; } = 1.0;
    }
}
