using System;
using System.Diagnostics;
using System.Net;
using System.Net.Http;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace HOPE.Core.Services.Cloud
{
    public interface ISsoService
    {
        Task<AuthResponse?> LoginWithGoogleAsync();
    }

    public class SsoService : ISsoService
    {
        private const string BackendUrl = "http://localhost:3000";
        private readonly HttpClient _httpClient;

        public SsoService()
        {
            _httpClient = new HttpClient();
        }

        public async Task<AuthResponse?> LoginWithGoogleAsync()
        {
            return await InitiateOAuthFlow("google");
        }



        private async Task<AuthResponse?> InitiateOAuthFlow(string provider)
        {
            // Simulate a browser-based login flow
            // In a production app, we would listen on a local port (e.g. http://localhost:5000/callback)
            
            string url = $"{BackendUrl}/auth/{provider}";
            OpenBrowser(url);

            // Simulation: Wait for "user" to complete login in browser
            await Task.Delay(2000);

            // Return a mocked success response
            return new AuthResponse
            {
                AccessToken = "mock_jwt_token_header.payload.signature",
                RefreshToken = "mock_refresh_token",
                User = new UserInfo 
                {
                    Id = Guid.NewGuid().ToString(),
                    Email = "demo.user@hope-project.org",
                    FirstName = "Demo",
                    LastName = "Mechanic",
                    Role = "Admin"
                }
            };
        }

        private void OpenBrowser(string url)
        {
            try
            {
                Process.Start(url);
            }
            catch
            {
                // hack because of this: https://github.com/dotnet/corefx/issues/10361
                if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                {
                    url = url.Replace("&", "^&");
                    Process.Start(new ProcessStartInfo("cmd", $"/c start {url}") { CreateNoWindow = true });
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                {
                    Process.Start("xdg-open", url);
                }
                else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                {
                    Process.Start("open", url);
                }
                else
                {
                    throw;
                }
            }
        }
    }

    public class AuthResponse
    {
        public string AccessToken { get; set; }
        public string RefreshToken { get; set; }
        public UserInfo User { get; set; }
    }

    public class UserInfo
    {
        public string Id { get; set; }
        public string Email { get; set; }
        public string FirstName { get; set; }
        public string LastName { get; set; }
        public string Role { get; set; }
    }
}
