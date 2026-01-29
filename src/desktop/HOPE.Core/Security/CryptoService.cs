using System;
using System.Security.Cryptography;
using System.Text;
using System.IO;

namespace HOPE.Core.Security
{
    public class CryptoService
    {
        private readonly IHardwareProvider _hardwareProvider;

        // Key derivation parameters
        private const int SaltSize = 16; // 128 bit
        private const int KeySize = 32; // 256 bit
        private const int NonceSize = 12; // 96 bit (standard for GCM)
        private const int TagSize = 16; // 128 bit (standard for GCM)
        private const int Iterations = 100000; // PBKDF2 iterations

        public CryptoService(IHardwareProvider hardwareProvider)
        {
            _hardwareProvider = hardwareProvider ?? throw new ArgumentNullException(nameof(hardwareProvider));
        }

        /// <summary>
        /// Encrypts data using AES-256-GCM.
        /// Derives key from password and optional hardware ID.
        /// </summary>
        public byte[] EncryptFile(byte[] data, string password, string? hardwareId = null)
        {
            // If hardwareId is null, we use the default from provider
            hardwareId ??= _hardwareProvider.GetHardwareId();

            // 1. Generate random salt
            byte[] salt = RandomNumberGenerator.GetBytes(SaltSize);

            // 2. Derive key from password + hardwareId + salt
            byte[] key = DeriveKey(password, hardwareId, salt);

            // 3. Generate random nonce (IV)
            byte[] nonce = RandomNumberGenerator.GetBytes(NonceSize);

            // 4. Encrypt
            using var aes = new AesGcm(key);
            
            byte[] ciphertext = new byte[data.Length];
            byte[] tag = new byte[TagSize];
            
            aes.Encrypt(nonce, data, ciphertext, tag);

            // 5. Combine into single blob
            using var ms = new MemoryStream();
            using var writer = new BinaryWriter(ms);
            
            writer.Write(salt);
            writer.Write(nonce);
            writer.Write(tag);
            writer.Write(ciphertext);
            
            return ms.ToArray();
        }

        /// <summary>
        /// Decrypts data using AES-256-GCM.
        /// </summary>
        public byte[] DecryptFile(byte[] encryptedData, string password, string? currentHardwareId = null)
        {
            using var ms = new MemoryStream(encryptedData);
            using var reader = new BinaryReader(ms);

            // 1. Read headers
            if (ms.Length < SaltSize + NonceSize + TagSize)
                throw new CryptographicException("Invalid encrypted data format (too short)");

            byte[] salt = reader.ReadBytes(SaltSize);
            byte[] nonce = reader.ReadBytes(NonceSize);
            byte[] tag = reader.ReadBytes(TagSize);
            byte[] ciphertext = reader.ReadBytes((int)(ms.Length - ms.Position));

            // 2. Derive key
            // If currentHardwareId is null, we use the default from provider
            currentHardwareId ??= _hardwareProvider.GetHardwareId();
            byte[] key = DeriveKey(password, currentHardwareId, salt);

            // 3. Decrypt
            try 
            {
                using var aes = new AesGcm(key);
                byte[] plaintext = new byte[ciphertext.Length];
                aes.Decrypt(nonce, ciphertext, tag, plaintext);
                return plaintext;
            }
            catch (CryptographicException)
            {
                // Here we could implement the migration mode check
                throw new CryptographicException("Decryption failed. Invalid password or hardware mismatch.");
            }
        }

        /// <summary>
        /// Re-encrypts data for a new hardware ID if the old one is known (e.g., via a migration flow).
        /// </summary>
        public byte[] MigrateEncryptedData(byte[] encryptedData, string password, string oldHardwareId)
        {
            try
            {
                // 1. Decrypt with old hardware ID
                byte[] plaintext = DecryptFile(encryptedData, password, oldHardwareId);

                // 2. Re-encrypt with CURRENT hardware ID
                return EncryptFile(plaintext, password);
            }
            catch (Exception ex)
            {
                throw new CryptographicException("Migration failed. Verify old hardware ID and password.", ex);
            }
        }

        private byte[] DeriveKey(string password, string? hardwareId, byte[] salt)
        {
            string combinedInput = password;
            if (!string.IsNullOrEmpty(hardwareId))
            {
                combinedInput += $"|{hardwareId}";
            }

            using var pbkdf2 = new Rfc2898DeriveBytes(combinedInput, salt, Iterations, HashAlgorithmName.SHA256);
            return pbkdf2.GetBytes(KeySize);
        }
    }
}
