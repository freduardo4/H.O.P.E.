import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.onnx
import numpy as np
from sklearn.model_selection import train_test_split

from .config import (
    EPOCHS,
    BATCH_SIZE,
    SEQUENCE_LENGTH,
    VALIDATION_SPLIT,
    DEVICE
)
from .model import AnomalyDetector
from .dataset import generate_synthetic_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_to_onnx(detector: AnomalyDetector, output_path: Path):
    """Export the model to ONNX format for desktop deployment."""
    try:
        logger.info("Exporting model to ONNX format...")

        # Dummy input for tracing
        dummy_input = torch.randn(1, detector.model.sequence_length, detector.model.n_features).to(DEVICE)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        detector.model.eval()
        torch.onnx.export(
            detector.model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False  # Use legacy exporter for compatibility
        )

        logger.info(f"ONNX model saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to export ONNX model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder for anomaly detection')
    parser.add_argument('--data_dir', type=str, default='../data/processed',
                        help='Directory containing processed training data')
    parser.add_argument('--output_dir', type=str, default='../models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Training batch size')
    parser.add_argument('--generate_synthetic', action='store_true',
                        help='Generate synthetic training data')
    parser.add_argument('--n_vehicles', type=int, default=100,
                        help='Number of vehicles for synthetic data')
    
    # If running as a module (python -m hope_ai.train), sys.argv might need handling
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load data
    if args.generate_synthetic:
        data, labels = generate_synthetic_data(n_vehicles=args.n_vehicles)
    else:
        # Resolve path relative to current working directory or script location
        # Here assuming running from src/ai-training
        data_path = Path(args.data_dir) / 'training_data.npy'
        if data_path.exists():
            logger.info(f"Loading data from {data_path}")
            data = np.load(data_path)
            # labels = np.zeros(len(data)) 
        else:
            logger.warning(f"No data found at {data_path}. Generating synthetic data.")
            data, labels = generate_synthetic_data(n_vehicles=args.n_vehicles)

    # Initialize model
    detector = AnomalyDetector()

    # Prepare sequences
    logger.info("Preparing sequences...")
    sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
    
    # Split into train/validation
    X_train, X_val = train_test_split(
        sequences, test_size=VALIDATION_SPLIT, random_state=42
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Train
    logger.info("Starting training...")
    history = detector.fit(
        X_train,
        X_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = output_dir / f'lstm_autoencoder_{timestamp}'
    detector.save(model_path)

    # Export to ONNX for desktop deployment
    onnx_path = output_dir / 'onnx' / 'anomaly_detector.onnx'
    export_to_onnx(detector, onnx_path)

    # Save training history
    history_path = output_dir / f'training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on validation set
    logger.info("Evaluating model...")
    anomalies, scores = detector.detect_anomalies(X_val)

    logger.info(f"Validation anomaly rate: {anomalies.mean()*100:.2f}%")
    logger.info(f"Mean reconstruction error: {scores.mean():.6f}")
    if detector.threshold:
        logger.info(f"Threshold: {detector.threshold:.6f}")

    logger.info("Training complete!")

if __name__ == '__main__':
    main()
