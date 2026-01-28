# Model Card: LSTM Autoencoder Anomaly Detector

## Model Details
- **Name**: LSTM Autoencoder Anomaly Detector
- **Version**: 1.0.0
- **Type**: Recurrent Neural Network (LSTM)
- **Framework**: PyTorch 2.10.0
- **Format**: ONNX
- **Description**: An unsupervised anomaly detection model designed to identify irregular patterns in automotive diagnostic telemetry (e.g., voltage fluctuations, message timing jitter).

## Intended Use
- **Primary Use Case**: Real-time safety monitoring during ECU flashing.
- **Target Users**: H.O.P.E. Desktop application users (Calibration engineers).
- **Out-of-Scope Use Cases**: Diagnosis of mechanical hardware failure unrelated to electronic telemetry.

## Factors
- **Environmental Factors**: Battery voltage level, CAN bus load.
- **ECU Variability**: Different ECU models may have varying telemetry signatures.

## Metrics
- **Reconstruction Error (MSE)**: Used to derive the anomaly score.
- **Inference Latency**: < 50ms per batch on standard CPU.

## Training Data
- **Dataset**: Synthetic and captured CAN bus telemetry from stable H.O.P.E. test environments.
- **Preprocessing**: Robust scaling, sequence windowing (size=10).

## Quantitative Analyses
- **Verification Status**: 
  - [x] ONNX Export Valid
  - [x] Unit Tests Passed (13/13)
  - [ ] Performance Regression Benchmarked

## Ethical Considerations
- **Safety**: The model is a "fail-safe" advisor. It flags anomalies but allows override by authorized personnel if manual verification is performed.

## Caveats and Recommendations
- **False Positives**: Initial connection establishment may trigger transient anomalies.
- **Retraining**: Model should be retrained periodically as new ECU variants are introduced to the H.O.P.E. ecosystem.
