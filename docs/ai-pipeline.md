# H.O.P.E. AI/ML Pipeline Documentation

This document describes the AI/ML pipeline for the H.O.P.E. project, covering data ingestion, training, evaluation, and deployment.

## Overview

The AI components of H.O.P.E. include:
1. **Anomaly Detection**: LSTM Autoencoder to identify sensor drift and vehicle faults.
2. **Virtual Sensors (PINN)**: Physics-Informed Neural Networks to estimate metrics like Exhaust Gas Temperature (EGT).
3. **RUL Forecasting**: Remaining Useful Life prediction for vehicle components.
4. **Tuning Optimizer**: Genetic Algorithm to optimize ECU calibrations (Integrated via generalized `TuneOptimizer` for CLI/Backend support).

## Pipeline Structure

```
src/ai-training/
├── configs/            # Centralized JSON configurations
├── hope_ai/            # Core library (models, datasets, training logic)
├── models/             # Saved model weights and ONNX exports
├── tests/              # Regression and unit tests
└── hope_ai_cli.py      # Unified entrypoint
```

## Configuration

Configurations are stored in `src/ai-training/configs/` as JSON files:
- `features.json`: Defines which OBD2 PIDs are used by the models.
- `hyperparameters.json`: Training and architecture hyperparameters.

## Training

Use the unified CLI to train models:

```bash
# Train Anomaly Detector
python hope_ai_cli.py train-anomaly --epochs 100

# Train PINN Virtual Sensor
python hope_ai_cli.py train-pinn --epochs 50

# Train RUL Forecaster
python hope_ai_cli.py train-rul --epochs 50
```

## Explainability (XAI)

H.O.P.E. uses SHAP and LIME to provide "Diagnostic Narratives":

```bash
python hope_ai_cli.py explain --input test_data.json --method SHAP
```

## Model Traceability

Training runs are tracked using **MLflow**. Metrics such as loss, validation error, and hyperparameters are logged automatically during training.

## Deployment

Models are exported to **ONNX** format for high-performance inference in the .NET Desktop application.
Exports are located in `src/ai-training/models/onnx/`.

## Quality Assurance

Regression tests ensure model stability and accuracy:

```bash
pytest src/ai-training/tests/test_model_regression.py
```
