# Recipe 03: AI Training Local

## Overview
Train the LSTM anomaly detector on your local machine using custom OBD logs.

## Prerequisites
- Python 3.9+
- CUDA Toolkit (optional, for GPU acceleration)
- `scripts/hope.ps1` CLI tool

## Steps

### 1. Setup Environment
Ensure your python environment is ready.
```powershell
.\scripts\hope.ps1 setup
```

### 2. Prepare Data
Place your CSV logs in `src/ai-training/data/raw`.
Format must match: `timestamp,rpm,load,tps,maf,iat,clt,afr`.

```powershell
# Example: Copy a log file
Copy-Item "C:\MyCarLogs\run1.csv" "src\ai-training\data\raw\training_set.csv"
```

### 3. Run Training
Use the training script directly or via python.

```powershell
# Activate venv if manually running
cd src/ai-training
.\venv\Scripts\Activate.ps1

# Run training script
python scripts/train_anomaly_detector.py --epochs 50 --batch-size 32
```

### 4. Evaluate Model
The script will output:
- `models/anomaly_detector_vX.onnx`
- `reports/training_loss.png`
- `reports/reconstruction_error.png`

Check the loss curve in `reports/` to ensure the model converged.

### 5. Export to Desktop App
Copy the resulting `.onnx` file to the Desktop application's assets folder.

```powershell
Copy-Item "models/anomaly_detector_v1.onnx" "../../desktop/HOPE.Desktop/Assets/Models/"
```
