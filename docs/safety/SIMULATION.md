# Simulation Environment & Safety

H.O.P.E. provides a robust simulation environment to ensure that calibrations and diagnostic procedures are safe before they are applied to physical hardware.

## 1. Simulated Hardware Adapter
The `MockHardwareAdapter` allows developers to test the application logic without a physical J2534 or ELM327 device. 
- **Voltage Simulation**: Simulate low-battery conditions to verify that flashing is correctly blocked.
- **Protocol Simulation**: Returns standard responses for OBD2, UDS, and KWP2000 queries.

## 2. Digital Twin (BeamNG.drive)
The [HiL Testing Tier](../plans/walkthrough_hiL.md) leverages the high-fidelity physics of BeamNG.drive to simulate:
- **Engine Thermals**: Detects if a tune causes over-heating under sustained load.
- **Mechanical Stress**: Simulates piston/rod failure if ignition timing or boost limits are exceeded.
- **Fault Injection**: Manually inject sensor failures (MAF drift, Packet Loss) to see how the ECU and HOPE app respond.

## 3. Mandatory Pre-Flight Validation
For professional installations, a calibration MUST pass a "Lap of Truth" in simulation before the flash unlock is granted by the [Safety Policy Service](../safety/FLASHING_SAFETY.md).
