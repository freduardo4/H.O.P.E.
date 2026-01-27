# Marketplace Security Design

The H.O.P.E. Marketplace allows trusted tuners to sell encrypted calibrations.

## Principles
1.  **IP Protection**: Tuners' "secret sauce" (maps) must not be viewable by the buyer, only flashable.
2.  **Hardware Locking**: A purchased file is locked to a specific VIN or Hardware ID.

## Workflow

### 1. Listing & Purchase
- Tuner uploads a "Master Calibration" (encrypted with Server Key).
- Buyer purchases license for Vehicle `VIN_123`.

### 2. File Generation
- Server decrypts Master.
- Server re-encrypts calibration using a derived key `K_device = KDF(VIN_123, User_Secret)`.
- Server signs the payload.

### 3. Flashing
- Desktop App downloads the encrypted package.
- Desktop App connects to vehicle, verifies VIN `VIN_123`.
- If VIN matches, App decrypts payload *in memory* and streams to ECU.
- **Critical**: Decrypted data is never written to disk.

## Auditing
- All transactions generate a blockchain-like hash entry in the Audit Log, ensuring proof of purchase and origin.
