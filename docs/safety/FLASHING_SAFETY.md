# Flashing / Tuning Safety & Legal Requirements

## Liability Disclaimer
**WARNING:** Modifying ECU calibrations can cause permanent engine damage, void warranties, and violate emissions regulations. H.O.P.E. provides tools for diagnostics and calibration management but assumes NO LIABILITY for the files flashed to vehicles.

## Safety Interlocks
To prevent catastrophic failure during flashing:

1.  **Battery Voltage Check**: Flash prohibited if voltage < 12.5V.
2.  **Engine State**: Flash prohibited if RPM > 0 (Engine must be off).
3.  **Seed-Key Exchange**: Must successfully complete security access level 0x27.
4.  **Checksum Validation**: All memory blocks must be validated before the flash commit.

## Pre-Flash Validation
Every calibration must pass the following automated checks before being unlocked for flashing:
- **Global Limits**: No table values exceeding physical sensor limits.
- **Sanity Check**: Spark advance tables must be monotonic (mostly).
- **Signature**: File must be signed by a trusted key (see [Code Signing](../security/signing.md)).
