# PKI & Calibration Signing Strategy

To ensure calibration integrity and prevent unauthorized engine damage from malicious or corrupt files, HOPE employs a Public Key Infrastructure (PKI).

## 1. Hierarchy

- **Root CA**: Offline, ultra-secure private key used to sign Intermediate CAs.
- **Intermediate CA (Signing Agent)**: Hosted by HOPE Central. Responsible for signing Tuner public keys and verifying signature requests.
- **Tuner Keys**: Each authorized tuner possesses a unique asymmetric key pair (RSA-4096 or ECDSA P-384).

## 2. Signing Flow

1. **Upload**: Tuner uploads a `.hcal` (HOPE Calibration) file.
2. **Sign**: The backend hashes the file content and signs it using the Tuner's private key (stored in an HSM or encrypted Vault).
3. **Envelope**: The file is packaged with a CMS (Cryptographic Message Syntax) signature.
4. **Validation**: The Desktop Client verifies:
   - File Integrity (Hash Match).
   - Signer Authenticity (Tuner Certificate chained to HOPE Root).
   - Revocation Status (CRL/OCSP check).

## 3. Revocation & Trust

If a tuner is found to be providing dangerous or illegal files:
- Their certificate is added to a **Revocation List**.
- The Desktop Hardware Interface (J2534 Adapter) will refuse to flash any files signed by the revoked certificate.

## 4. Hardware Enforcement

The `IHardwareAdapter` implementation will eventually include `VerifySignature(byte[] file, byte[] signature)` as a pre-condition for the `FlashECU` command.
