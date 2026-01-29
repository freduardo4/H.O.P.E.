# H.O.P.E. Onboarding Guide

Welcome to the future of vehicle diagnostics and optimization. This guide will walk you through setting up your environment and performing your first diagnostic session.

## 1. Environment Setup

H.O.P.E. is a multi-stack project. Ensure you have the following installed:
- **Languages**: Node.js (v20+), .NET (v8.0), Python (v3.10+)
- **Database**: PostgreSQL (v15+)
- **Hardware**: J2534-compliant adapter (or use our [Hardware Simulator](docs/safety/SIMULATION.md))

## 2. Fast-Track Deployment

Open a PowerShell terminal and run:
```powershell
# 1. Clone & Setup
git clone https://github.com/freduardo4/H.O.P.E.git
cd H.O.P.E
.\scripts\hope.ps1 setup

# 2. Run Backend
.\scripts\hope.ps1 start backend

# 3. Run Desktop
# (In a separate terminal)
.\scripts\hope.ps1 start desktop
```

## 3. Your First Session

1.  **Login**: Use the default credentials (if seeded) or register a new technician account.
2.  **Connect**: Click the **Connect** icon in the Desktop app. Choose "Simulated Hardware" if you don't have a physical adapter connected.
3.  **Scan**: Execute a "Full System Scan". Watch the AI Anomaly Detector analyze live telemetry in the **Insights** tab.
4.  **Safety Check**: Navigate to the **Safety** dashboard to verify the vehicle's battery voltage and module health status.

## 4. Next Steps

- **Tuning**: Learn how to use the [Map Diff Tool](docs/features/MAP_DIFF.md) and [Calibration Ledger](docs/security/signing.md).
- **Simulation**: Conduct virtual pre-flight tests using the [HiL Testing Tier](docs/plans/walkthrough_hiL.md).
- **Quality**: Review the [Verification Report](docs/plans/walkthrough_verification.md) for current system stability.
- **AI Training**: Check the [MLOps Guide](docs/ml/MLOPS.md) to retrain models on your own telemetry.

---
*For support, visit our [Wiki-Fix Community](docs/community/WIKI.md).*
