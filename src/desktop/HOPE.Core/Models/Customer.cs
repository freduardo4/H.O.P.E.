namespace HOPE.Core.Models;

/// <summary>
/// Represents a customer in the HOPE system.
/// </summary>
public class Customer
{
    public Guid CustomerId { get; set; }
    public Guid ShopId { get; set; }

    public string FirstName { get; set; } = string.Empty;
    public string LastName { get; set; } = string.Empty;

    public string? Email { get; set; }
    public string Phone { get; set; } = string.Empty;

    /// <summary>
    /// Customer address
    /// </summary>
    public Address? Address { get; set; }

    /// <summary>
    /// Customer preferences (notifications, report format, etc.)
    /// </summary>
    public CustomerPreferences Preferences { get; set; } = new();

    /// <summary>
    /// Customer notes (technician observations, preferences)
    /// </summary>
    public string? Notes { get; set; }

    /// <summary>
    /// Customer tags (VIP, Fleet, etc.)
    /// </summary>
    public List<string> Tags { get; set; } = new();

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Navigation property for vehicles
    /// </summary>
    public List<Vehicle> Vehicles { get; set; } = new();

    /// <summary>
    /// Full name of the customer
    /// </summary>
    public string FullName => $"{FirstName} {LastName}".Trim();

    public override string ToString() => $"{FullName} - {Phone}";
}

/// <summary>
/// Customer address information
/// </summary>
public class Address
{
    public string Street { get; set; } = string.Empty;
    public string? Street2 { get; set; }
    public string City { get; set; } = string.Empty;
    public string State { get; set; } = string.Empty;
    public string PostalCode { get; set; } = string.Empty;
    public string Country { get; set; } = "USA";

    public override string ToString() =>
        $"{Street}, {City}, {State} {PostalCode}, {Country}";
}

/// <summary>
/// Customer preferences and settings
/// </summary>
public class CustomerPreferences
{
    /// <summary>
    /// Preferred contact method
    /// </summary>
    public ContactMethod PreferredContactMethod { get; set; } = ContactMethod.Email;

    /// <summary>
    /// Whether to send email notifications
    /// </summary>
    public bool EmailNotifications { get; set; } = true;

    /// <summary>
    /// Whether to send SMS notifications
    /// </summary>
    public bool SMSNotifications { get; set; } = false;

    /// <summary>
    /// Preferred report format
    /// </summary>
    public ReportFormat PreferredReportFormat { get; set; } = ReportFormat.PDF;

    /// <summary>
    /// Language preference
    /// </summary>
    public string Language { get; set; } = "en-US";

    /// <summary>
    /// Whether customer wants performance recommendations
    /// </summary>
    public bool SendPerformanceRecommendations { get; set; } = true;

    /// <summary>
    /// Whether customer wants predictive maintenance alerts
    /// </summary>
    public bool SendMaintenanceAlerts { get; set; } = true;
}

/// <summary>
/// Contact methods
/// </summary>
public enum ContactMethod
{
    Email,
    Phone,
    SMS,
    NoPreference
}

/// <summary>
/// Report format options
/// </summary>
public enum ReportFormat
{
    PDF,
    Email,
    Both
}

/// <summary>
/// Shop/business information
/// </summary>
public class Shop
{
    public Guid ShopId { get; set; }
    public string Name { get; set; } = string.Empty;
    public string? LegalName { get; set; }

    public Address? Address { get; set; }
    public string Phone { get; set; } = string.Empty;
    public string? Email { get; set; }
    public string? Website { get; set; }

    /// <summary>
    /// Subscription tier
    /// </summary>
    public SubscriptionTier Subscription { get; set; } = SubscriptionTier.Basic;

    /// <summary>
    /// Logo URL/path for branding
    /// </summary>
    public string? LogoUrl { get; set; }

    /// <summary>
    /// Shop branding colors (hex codes)
    /// </summary>
    public ShopBranding Branding { get; set; } = new();

    /// <summary>
    /// Shop settings
    /// </summary>
    public ShopSettings Settings { get; set; } = new();

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? UpdatedAt { get; set; }

    /// <summary>
    /// Navigation property for users (technicians)
    /// </summary>
    public List<User> Users { get; set; } = new();

    /// <summary>
    /// Navigation property for customers
    /// </summary>
    public List<Customer> Customers { get; set; } = new();

    public override string ToString() => Name;
}

public class ShopBranding
{
    public string PrimaryColor { get; set; } = "#1E40AF";
    public string SecondaryColor { get; set; } = "#64748B";
    public string AccentColor { get; set; } = "#F59E0B";
}

public class ShopSettings
{
    public string TimeZone { get; set; } = "America/New_York";
    public string Currency { get; set; } = "USD";
    public bool EnableOnlineBooking { get; set; }
    public bool EnableCustomerPortal { get; set; }
}

/// <summary>
/// Subscription tiers
/// </summary>
public enum SubscriptionTier
{
    Trial,
    Basic,
    Professional,
    Enterprise
}

/// <summary>
/// User/technician information
/// </summary>
public class User
{
    public Guid UserId { get; set; }
    public Guid ShopId { get; set; }

    public string Email { get; set; } = string.Empty;
    public string FirstName { get; set; } = string.Empty;
    public string LastName { get; set; } = string.Empty;

    public UserRole Role { get; set; } = UserRole.Technician;

    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? LastLoginAt { get; set; }

    public string FullName => $"{FirstName} {LastName}".Trim();

    /// <summary>
    /// Navigation property to shop
    /// </summary>
    public Shop? Shop { get; set; }

    public override string ToString() => $"{FullName} ({Role})";
}

/// <summary>
/// User roles for RBAC
/// </summary>
public enum UserRole
{
    Admin,
    ShopOwner,
    ShopManager,
    Technician,
    ReadOnly,
    Customer
}
