namespace HOPE.Core.Models
{
    public class MapSwitchProfile
    {
        public byte Id { get; set; }
        public string Name { get; set; }
        public string Description { get; set; }
        
        /// <summary>
        /// Rev limit in RPM
        /// </summary>
        public int RevLimit { get; set; }
        
        /// <summary>
        /// Target boost in PSI or Bar (unit depends on ECU, assuming PSI for display)
        /// </summary>
        public double BoostTarget { get; set; }

        public static MapSwitchProfile Default => new MapSwitchProfile 
        { 
            Id = 1, 
            Name = "Normal", 
            Description = "Standard daily driving calibration.", 
            RevLimit = 6500, 
            BoostTarget = 14.7 
        };
    }
}
