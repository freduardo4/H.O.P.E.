# H.O.P.E. (Heuristic Optimization & Predictive Ecosystem)

![H.O.P.E. Logo](docs/assets/image001.png)

## Overview

H.O.P.E. is an advanced, AI-powered vehicle diagnostics and ECU calibration platform. It bridges the gap between traditional tuning tools and modern predictive maintenance using machine learning.

## Quick Start

### Prerequisites
- .NET 8.0 SDK
- Node.js 20.x
- Python 3.10+
- PostgreSQL

### Running the Stack

1.  **Backend**:
    ```bash
    cd src/backend
    npm install
    npm run start:dev
    ```

2.  **Desktop App**:
    - Open `src/desktop/HOPE.Desktop.sln` in Visual Studio.
    - Build and Run.

3.  **AI Training**:
    ```bash
    cd src/ai-training
    pip install -r requirements.txt
    python train_anomaly_detector.py
    ```

## Documentation

- **[Full Documentation](docs/README_full.md)**: Detailed architecture, features, and deep dives.
- **[Documentation Index](docs/index.md)**: Navigation for all project docs.
- **[Contributing](CONTRIBUTING.md)**: How to get involved.

## License

See [LICENSE](LICENSE) file.
