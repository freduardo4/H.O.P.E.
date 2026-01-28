# H.O.P.E. (High-Output Performance Engineering)

![H.O.P.E. Logo](docs/assets/image001.png)

## Overview

H.O.P.E. is an advanced, AI-powered vehicle diagnostics and ECU calibration platform. It bridges the gap between traditional tuning tools and modern predictive maintenance using machine learning.

## Quick Start

### Prerequisites
- .NET 8.0 SDK
- Node.js 20.x
- Python 3.10+
- PostgreSQL

## Quick Start

The easiest way to get started is using the **HOPE Developer CLI**.

1.  **Dependencies**: Ensure .NET 8.0, Node.js 20+, and Python 3.10+ are installed.
2.  **Setup**: Run `.\scripts\hope.ps1 setup` to install all module dependencies.
3.  **Start Services**:
    - `.\scripts\hope.ps1 start backend` (API server)
    - `.\scripts\hope.ps1 start desktop` (UI App)
4.  **Test Everything**: `.\scripts\hope.ps1 test all`

## Project Status

- **Status**: ✅ All core components verified.
- **Diagnostics**: UDS/KWP2000 protocol handlers implemented and tested.
- **Safety**: Voltage-aware HAL and simulated hardware integration complete.
- **AI**: LSTM Anomaly Detection pipeline functional with 96% test coverage.
- **Tests**: 440+ automated tests passing across Desktop, Backend, and AI.

## Documentation & Recipes

- **[Developer Recipes](recipes/README.md)**: Living examples for common tasks (OBD sessions, tuning, AI).
- **[Full Documentation](docs/README_full.md)**: Detailed architecture and deep dives.
- **[Documentation Index](docs/index.md)**: Navigation for all project docs.
- **[Contributing](CONTRIBUTING.md)**: How to get involved.

## License

See [LICENSE](LICENSE) file.
