# Code Signing and Artifact Security

To ensure the integrity and safety of vehicle calibrations and application binaries, H.O.P.E. employs a strict code signing process.

## Application Binaries

### Desktop App
- All production releases of the Desktop App (`.exe` / `.msi`) must be signed using a trusted Code Signing Certificate.
- Windows SmartScreen reputation is established through this certificate.
- **CI Process**: The release pipeline unlocks the signing key from a secure vault (e.g., Azure Key Vault / AWS KMS) only during the `release` stage of tagged builds.

## Calibration Files

ECU Calibrations (`.bin`, `.hex`, `.cal`) are critical safety components.

### Signing Flow
1.  **Generation**: The AI Tuner generates a candidate calibration.
2.  **Validation**: Automated safety checks verify limits (e.g., max boost, timing advance).
3.  **Signing**:
    - Validated calibrations are hashed (SHA-256).
    - The hash is signed with the H.O.P.E. Private Key using RSA-4096 or ECDSA.
    - The signature is appended to the file header or stored in a sidecar manifest.
4.  **Verification**:
    - The Desktop Flasher Tool verifies the signature against the embedded Public Key before writing to any ECU.
    - If verification fails, the flash is aborted immediately.

## Key Management

- **Root Key**: Stored offline in a Hardware Security Module (HSM) or air-gapped cold storage.
- **Signing Keys**: Rotated quarterly. access restricted to Release Engineering team and validated CI pipelines.
