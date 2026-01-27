"""
HOPE AI Inference Script

This script provides inference capabilities for the trained anomaly detection model.
It can be used for:
- Batch inference on historical data
- Real-time inference on streaming data
- Model evaluation and testing

Usage:
    python inference.py --model_path ../models/lstm_autoencoder_latest --data_path ../data/test.csv
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

from train_anomaly_detector import DEVICE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detector interface for HOPE desktop application."""

    def __init__(self, model_path: str):
        """Initialize the detector with a trained model."""
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.is_onnx = False

        self._load_model()

    def _load_model(self):
        """Load the model from disk."""
        # Check for ONNX model first (preferred for production)
        onnx_path = self.model_path / 'onnx' / 'anomaly_detector.onnx'
        if onnx_path.exists():
            self._load_onnx_model(onnx_path)
        else:
            self._load_pytorch_model()

    def _load_onnx_model(self, onnx_path: Path):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(str(onnx_path))
            self.is_onnx = True

            # Load config
            import json
            config_path = self.model_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)

            # Load scaler
            import joblib
            scaler_path = self.model_path / 'scaler.joblib'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None

            logger.info(f"Loaded ONNX model from {onnx_path}")

        except ImportError:
            logger.warning("onnxruntime not available, falling back to PyTorch")
            self._load_pytorch_model()

    def _load_pytorch_model(self):
        """Load PyTorch model for inference."""
        # Import here to avoid loading TensorFlow if ONNX is available
        from train_anomaly_detector import AnomalyDetector as TrainingDetector

        training_detector = TrainingDetector.load(self.model_path)
        self.model = training_detector.model
        self.config = {
            'sequence_length': self.model.sequence_length,
            'n_features': self.model.n_features,
            'threshold': training_detector.threshold,
        }
        self.scaler = training_detector.scaler

        self.scaler = training_detector.scaler

        logger.info(f"Loaded PyTorch model from {self.model_path}")

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        """Preprocess input data for inference."""
        if self.scaler is not None:
            original_shape = data.shape
            flat = data.reshape(-1, self.config['n_features'])
            scaled = self.scaler.transform(flat)
            return scaled.reshape(original_shape)
        return data

    def predict(self, sequences: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on input sequences.

        Args:
            sequences: Input data of shape (n_sequences, sequence_length, n_features)

        Returns:
            Tuple of (reconstructions, anomaly_scores)
        """
        # Preprocess
        sequences_scaled = self.preprocess(sequences)

        if self.is_onnx:
            # ONNX inference
            input_name = self.session.get_inputs()[0].name
            reconstructions = self.session.run(
                None,
                {input_name: sequences_scaled.astype(np.float32)}
            )[0]
        else:
            # PyTorch inference
            self.model.eval()
            with torch.no_grad():
                inputs = torch.FloatTensor(sequences_scaled).to(DEVICE)
                reconstructions = self.model(inputs).cpu().numpy()

        # Calculate anomaly scores (MSE)
        mse = np.mean(np.power(sequences_scaled - reconstructions, 2), axis=(1, 2))

        return reconstructions, mse

    def detect_anomalies(
        self,
        sequences: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies in input sequences.

        Args:
            sequences: Input data
            threshold: Custom threshold (uses model threshold if None)

        Returns:
            Tuple of (is_anomaly, anomaly_scores, reconstructions)
        """
        reconstructions, scores = self.predict(sequences)

        if threshold is None:
            threshold = self.config.get('threshold', 0.1)

        is_anomaly = scores > threshold

        return is_anomaly, scores, reconstructions

    def analyze_anomaly(
        self,
        sequence: np.ndarray,
        feature_names: List[str] = None,
    ) -> Dict:
        """
        Analyze a single anomalous sequence to identify contributing features.

        Args:
            sequence: Single sequence of shape (sequence_length, n_features)
            feature_names: Names of features

        Returns:
            Dictionary containing analysis results
        """
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(sequence.shape[-1])]

        # Get reconstruction
        sequence_batch = sequence[np.newaxis, ...]
        reconstruction, score = self.predict(sequence_batch)

        # Calculate per-feature reconstruction error
        sequence_scaled = self.preprocess(sequence_batch)
        feature_errors = np.mean(
            np.power(sequence_scaled[0] - reconstruction[0], 2),
            axis=0
        )

        # Rank features by contribution to error
        feature_ranking = np.argsort(feature_errors)[::-1]

        analysis = {
            'anomaly_score': float(score[0]),
            'threshold': self.config.get('threshold', 0.1),
            'is_anomaly': bool(score[0] > self.config.get('threshold', 0.1)),
            'feature_errors': {
                feature_names[i]: float(feature_errors[i])
                for i in range(len(feature_names))
            },
            'top_contributing_features': [
                {
                    'name': feature_names[i],
                    'error': float(feature_errors[i]),
                    'rank': rank + 1,
                }
                for rank, i in enumerate(feature_ranking[:5])
            ],
        }

        return analysis


def create_sequences(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """Create sequences from continuous data."""
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return np.array(sequences)


def load_test_data(file_path: Path, sequence_length: int) -> np.ndarray:
    """Load test data from file."""
    if file_path.suffix == '.npy':
        data = np.load(file_path)
    elif file_path.suffix == '.csv':
        df = pd.read_csv(file_path)
        # Assume numeric columns are features
        data = df.select_dtypes(include=[np.number]).values
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    return create_sequences(data, sequence_length)


def main():
    parser = argparse.ArgumentParser(description='Run anomaly detection inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data_path', type=str,
                        help='Path to test data file')
    parser.add_argument('--threshold', type=float,
                        help='Custom anomaly threshold')
    parser.add_argument('--output', type=str,
                        help='Output file for results')
    args = parser.parse_args()

    # Initialize detector
    detector = AnomalyDetector(args.model_path)

    logger.info(f"Model config: {detector.config}")

    if args.data_path:
        # Load and process test data
        data_path = Path(args.data_path)
        sequences = load_test_data(data_path, detector.config['sequence_length'])

        logger.info(f"Loaded {len(sequences)} sequences from {data_path}")

        # Run detection
        is_anomaly, scores, _ = detector.detect_anomalies(
            sequences,
            threshold=args.threshold
        )

        # Print results
        n_anomalies = is_anomaly.sum()
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(sequences)*100:.1f}%)")
        logger.info(f"Score range: [{scores.min():.6f}, {scores.max():.6f}]")
        logger.info(f"Mean score: {scores.mean():.6f}")

        # Save results if output specified
        if args.output:
            results = pd.DataFrame({
                'sequence_idx': range(len(sequences)),
                'anomaly_score': scores,
                'is_anomaly': is_anomaly,
            })
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")

        # Analyze top anomalies
        if n_anomalies > 0:
            logger.info("\nTop 5 anomalies:")
            top_indices = np.argsort(scores)[-5:][::-1]
            for idx in top_indices:
                if is_anomaly[idx]:
                    analysis = detector.analyze_anomaly(sequences[idx])
                    logger.info(f"  Sequence {idx}: score={analysis['anomaly_score']:.6f}")
                    for feat in analysis['top_contributing_features'][:3]:
                        logger.info(f"    - {feat['name']}: error={feat['error']:.6f}")
    else:
        logger.info("No data path provided. Detector initialized successfully.")
        logger.info("Use --data_path to run inference on test data.")


if __name__ == '__main__':
    main()
