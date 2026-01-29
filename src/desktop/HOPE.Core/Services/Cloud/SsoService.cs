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
        Task<AuthResponse?> LoginWithGithubAsync();
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

        public async Task<AuthResponse?> LoginWithGithubAsync()
        {
            return await InitiateOAuthFlow("github");
        }

        private async Task<AuthResponse?> InitiateOAuthFlow(string provider)
        {
            // In a real desktop app, we'd start a local HTTP listener to receive the callback
            // For this implementation, we'll simulate the flow by opening the browser to the backend SSO endpoint
            
            string url = $"{BackendUrl}/auth/{provider}";
            OpenBrowser(url);

            // This is a simplified implementation. In a real app, we'd wait for the callback 
            // via a local server (e.g. http://localhost:5000/callback) or a custom protocol handler.
            // For now, we'll return a placeholder or wait for user to copy-paste (simplified).
            
            throw new NotImplementedException("Full OAuth2 callback handling in Desktop requires a local HTTP listener or custom protocol handler.");
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
