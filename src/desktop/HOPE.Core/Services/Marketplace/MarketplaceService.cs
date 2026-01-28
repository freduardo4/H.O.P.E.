using System;
using System.IO;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using HOPE.Core.Services.Security;

namespace HOPE.Core.Services.Marketplace
{
    public interface IMarketplaceService
    {
        Task<System.Collections.Generic.List<HOPE.Core.Models.CalibrationListing>> GetListingsAsync();
        Task<bool> PurchaseAndDownloadAsync(string listingId);
        Task<bool> ValidateAndDecryptAsync(string filePath, string licenseKey);
        string MarketplaceCachePath { get; }
    }

    public class MarketplaceService : IMarketplaceService
    {
        private readonly HttpClient _httpClient;
        private readonly IFingerprintService _fingerprintService;
        private readonly string _baseUrl = "http://localhost:3000/graphql";
        public string MarketplaceCachePath => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "HOPE", "Marketplace");

        public MarketplaceService(HttpClient httpClient, IFingerprintService fingerprintService)
        {
            _httpClient = httpClient;
            _fingerprintService = fingerprintService;

            if (!Directory.Exists(MarketplaceCachePath))
            {
                Directory.CreateDirectory(MarketplaceCachePath);
            }
        }

        public async Task<System.Collections.Generic.List<HOPE.Core.Models.CalibrationListing>> GetListingsAsync()
        {
            var query = new
            {
                query = @"query {
                    calibrationListings {
                        id
                        title
                        description
                        price
                        version
                    }
                }"
            };

            try 
            {
                var response = await _httpClient.PostAsync(_baseUrl, 
                    new StringContent(JsonSerializer.Serialize(query), Encoding.UTF8, "application/json"));
                
                if (!response.IsSuccessStatusCode) return new System.Collections.Generic.List<HOPE.Core.Models.CalibrationListing>();

                var json = await response.Content.ReadAsStringAsync();
                var doc = JsonDocument.Parse(json);
                var listingsFunc = doc.RootElement.GetProperty("data").GetProperty("calibrationListings");
                
                var result = new System.Collections.Generic.List<HOPE.Core.Models.CalibrationListing>();
                foreach (var item in listingsFunc.EnumerateArray())
                {
                    result.Add(new HOPE.Core.Models.CalibrationListing
                    {
                        Id = item.GetProperty("id").GetString(),
                        Title = item.GetProperty("title").GetString(),
                        Description = item.GetProperty("description").GetString(),
                        Price = item.GetProperty("price").GetDouble(),
                        Version = item.GetProperty("version").GetString()
                    });
                }
                return result;
            }
            catch 
            {
                return new System.Collections.Generic.List<HOPE.Core.Models.CalibrationListing>();
            }
        }

        public async Task<bool> PurchaseAndDownloadAsync(string listingId)
        {
            var fingerprint = _fingerprintService.GetHardwareFingerprint();
            
            // 1. Purchase Mutation
            var purchaseQuery = new
            {
                query = @"mutation Purchase($listingId: String!, $hardwareId: String!) {
                    purchaseCalibration(listingId: $listingId, hardwareId: $hardwareId) {
                        id
                        licenseKey
                        listing {
                            title
                            version
                        }
                    }
                }",
                variables = new { listingId, hardwareId = fingerprint }
            };

            var response = await _httpClient.PostAsync(_baseUrl, 
                new StringContent(JsonSerializer.Serialize(purchaseQuery), Encoding.UTF8, "application/json"));

            if (!response.IsSuccessStatusCode) return false;

            var json = await response.Content.ReadAsStringAsync();
            var doc = JsonDocument.Parse(json);
            if (doc.RootElement.TryGetProperty("errors", out _)) return false;

            var licenseData = doc.RootElement.GetProperty("data").GetProperty("purchaseCalibration");
            var licenseKey = licenseData.GetProperty("licenseKey").GetString();
            var title = licenseData.GetProperty("listing").GetProperty("title").GetString();
            // Sanitize filename
            var safeTitle = string.Join("_", title.Split(Path.GetInvalidFileNameChars()));

            // 2. Download encrypted binary using the new Controller Endpoint
            // GET /marketplace/download/:licenseKey?hardwareId=...
            
            var downloadUrl = $"http://localhost:3000/marketplace/download/{licenseKey}?hardwareId={fingerprint}";
            var fileResponse = await _httpClient.GetAsync(downloadUrl);

            if (!fileResponse.IsSuccessStatusCode) return false;

            var fileBytes = await fileResponse.Content.ReadAsByteArrayAsync();
            var fileName = $"{safeTitle}_{licenseKey.Substring(0, 8)}.bin";
            var filePath = Path.Combine(MarketplaceCachePath, fileName);

            await File.WriteAllBytesAsync(filePath, fileBytes);

            return true; 
        }

        public async Task<bool> ValidateAndDecryptAsync(string filePath, string licenseKey)
        {
            var fingerprint = _fingerprintService.GetHardwareFingerprint();

            // 1. Verify License with Backend
            var verifyQuery = new
            {
                query = @"query Verify($licenseKey: String!, $hardwareId: String!) {
                    verifyLicense(licenseKey: $licenseKey, hardwareId: $hardwareId)
                }",
                variables = new { licenseKey, hardwareId = fingerprint }
            };

            var response = await _httpClient.PostAsync(_baseUrl,
                new StringContent(JsonSerializer.Serialize(verifyQuery), Encoding.UTF8, "application/json"));

            // If verified, normally we'd decrypt here
            // The actual decryption would use the hardware fingerprint as part of the key derivation
            return response.IsSuccessStatusCode;
        }
    }
}
