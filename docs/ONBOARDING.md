# H.O.P.E. Onboarding Guide

Welcome to the future of vehicle diagnostics and analytics. This guide will walk you through setting up your environment and performing your first diagnostic session.

## 1. Environment Setup

H.O.P.E. is a multi-stack project. Ensure you have the following installed:
- **Languages**: Node.js (v20+), .NET (v8.0), Python (v3.10+)
- **Database**: PostgreSQL (v15+)
- **Hardware**: Generic OBD2 ELM327-compatible adapter

## 2. Fast-Track Deployment

Open a PowerShell terminal and run:
```powershell
# 1. Clone & Setup
git clone https://github.com/freduardo4/H.O.P.E.git
cd H.O.P.E

# 2. Start Backend
cd src/backend
npm install
npm run start:dev

# 3. Start Desktop
# Open src/desktop/HOPE.Desktop.sln in Visual Studio and Run
```

## 3. Your First Session

1.  **Login**: Use Google OAuth or create a local technician account.
2.  **Connect**: Plug in your OBD2 adapter and click **Connect** in the Desktop app.
3.  **Scan**: Execute a "Full System Scan". Watch the AI Anomaly Detector analyze live telemetry in the **Insights** tab.
4.  **Analyze**: Review the RUL (Remaining Useful Life) predictions for critical components.

## 4. Next Steps

- **AI Training**: Check the [MLOps Guide](ml/MLOPS.md) to learn about the anomaly detection pipeline.
- **Wiki-Fix**: Contribute repair patterns to the [Wiki-Fix Community](community/WIKI.md).

---
*For support, visit our documentation index.*
