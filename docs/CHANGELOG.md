# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- **Desktop**: Overlay Comparison ("Hit Tracing") feature in `MapVisualizationViewModel`.
- **Desktop**: `LogReplayService` for CSV log playback and testing.
- **Tests**: Comprehensive unit tests for `VoltageMonitor`, `CalibrationRepository` (Rollback), and `AxisEditorViewModel`.
- **AI**: MLOps pipeline improvements: Centralized `requirements.txt`, standardized `/configs/`, and unified `hope_ai_cli.py`.
- **AI**: Added `docs/ai-pipeline.md` detailing the training and export workflows.
- **AI**: Implemented RAG-based **"Tuning Copilot"** using TF-IDF and Prism-based Chat UI.
- **Desktop**: Real-time DTC Filtering & Search in `DTCViewModel` (by Code, Description, Category).
- **Desktop**: Enhanced Error Handling and Status Bar feedback in `MultiViewEditorViewModel`.
- **Tests**: `MapVisualizationViewModel` test for active cell tracking logic.
- **Tests**: `DTCViewModel` tests for search and filtering logic.
- **AI**: Stabilized `TuneOptimizer` and `RLGuidedOptimizer` integration with Backend CLI.
- **AI**: Fixed inference test suite `ModuleNotFoundError` and LSTM state warnings.
- **Backend**: Resolved `TuningService` timeouts by adjusting Jest configuration.
- **Desktop**: Integrated `Clowd.Squirrel` for auto-updates and `Sentry` for crash reporting.
- **Backend**: Implemented API Governance: Global `ValidationPipe`, `AllExceptionsFilter`, and Swagger/OpenAPI (`/api`) documentation.
- **Tests**: Added unit tests for `RolesGuard`, `JwtAuthGuard`, and `WikiFixController` to improve backend coverage.
- **Testing**: Completed Project-wide Test Gap Remediation:
  - **Backend**: 272 tests passing (Marketplace, Vehicles, Infra Services).
  - **Desktop**: 241 tests passing (CopilotViewModel, MultiViewEditorViewModel, PreFlightIntegration).
  - **AI**: Verified RL Optimization with Emissions Guardrail tests.
- **Environment**: Verified environment parity (Dev/Staging/Prod) configurations.

### Changed
- Refactored `MapVisualizationViewModel` to support live data overlay.
- Updated documentation (`README_full.md`, `task.md`, `walkthrough.md`) to reflect recent progress.
