using System;
using System.Text;
using Xunit;
using HOPE.Core.Security;
using HOPE.Core.Interfaces;
using Moq;

namespace HOPE.Desktop.Tests
{
    public class CryptoServiceTests
    {
        [Fact]
        public void EncryptDecrypt_WithCorrectPassword_Succeeds()
        {
            var mockHardware = new Mock<IHardwareProvider>();
            mockHardware.Setup(x => x.GetHardwareId()).Returns("DEFAULT_ID");
            var service = new CryptoService(mockHardware.Object);
            string password = "SecretPassword123";
            byte[] data = Encoding.UTF8.GetBytes("Hello World");

            byte[] encrypted = service.EncryptFile(data, password);
            byte[] decrypted = service.DecryptFile(encrypted, password);

            Assert.Equal(data, decrypted);
        }

        [Fact]
        public void EncryptDecrypt_WithWrongPassword_ThrowsException()
        {
            var mockHardware = new Mock<IHardwareProvider>();
            mockHardware.Setup(x => x.GetHardwareId()).Returns("DEFAULT_ID");
            var service = new CryptoService(mockHardware.Object);
            string password = "SecretPassword123";
            byte[] data = Encoding.UTF8.GetBytes("Hello World");

            byte[] encrypted = service.EncryptFile(data, password);

            Assert.Throws<System.Security.Cryptography.CryptographicException>(() =>
            {
                service.DecryptFile(encrypted, "WrongPassword");
            });
        }

        [Fact]
        public void HardwareLock_PreventsDecryptionOnDifferentHardware()
        {
            var mockHardware = new Mock<IHardwareProvider>();
            mockHardware.Setup(x => x.GetHardwareId()).Returns("DEFAULT_ID");
            var service = new CryptoService(mockHardware.Object);
            string password = "SecretPassword123";
            byte[] data = Encoding.UTF8.GetBytes("Locked Content");
            string hardwareIdA = "DEVICE_A_ID";
            string hardwareIdB = "DEVICE_B_ID";

            // Encrypt locked to A
            byte[] encrypted = service.EncryptFile(data, password, hardwareIdA);

            // Decrypt with A (Success)
            byte[] decryptedA = service.DecryptFile(encrypted, password, hardwareIdA);
            Assert.Equal(data, decryptedA);

            // Decrypt with B (Fail)
            Assert.Throws<System.Security.Cryptography.CryptographicException>(() =>
            {
                service.DecryptFile(encrypted, password, hardwareIdB);
            });
        }
    }
}
