# MLOps & AI Pipeline

H.O.P.E. uses an automated MLOps pipeline to ensure reproducible and verifiable AI models for vehicle diagnostics.

## 1. Data Provenance (DVC)
We use Data Version Control (DVC) to track datasets stored in S3.
- **Pull Data**: `dvc pull`
- **Track Changes**: `dvc add data/telemetry.csv`

## 2. Training Pipeline
The anomaly detection model is trained using the LSTM Autoencoder architecture.
- **Standard Training**: `python src/ai-training/scripts/train_anomaly_detector.py`
- **Output**: An ONNX model file exported to `src/desktop/HOPE.Core/Assets/Models/`.

## 3. Performance Metrics
A `MODEL_CARD.md` is generated for every production model, detailing:
- Training set distribution (RPM ranges, engine loads).
- Accuracy vs. False Positive rate.
- Safety guardrails (AFR limits).

## 4. Integration
The Desktop App uses the **ONNX Runtime** to run inference locally at 10Hz, ensuring real-time response without cloud latency.
