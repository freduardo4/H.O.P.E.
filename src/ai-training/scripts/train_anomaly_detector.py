"""
HOPE AI Training Script - LSTM Autoencoder for Anomaly Detection

This script trains an LSTM Autoencoder model for detecting anomalies in vehicle
OBD2 data. The model learns normal vehicle behavior patterns and flags deviations
as potential issues.

Architecture:
- Input: 10 OBD2 parameters x 60 timesteps (60 seconds @ 1 Hz)
- LSTM Encoder: 64 units -> Latent Space: 16 dimensions
- LSTM Decoder: 64 units -> Output reconstruction
- Anomaly detection via reconstruction error threshold

Usage:
    python train_anomaly_detector.py --data_dir ../data/processed --output_dir ../models
"""

import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# OBD2 Parameters to use for training
OBD2_FEATURES = [
    'engine_rpm',
    'vehicle_speed',
    'engine_load',
    'coolant_temp',
    'intake_air_temp',
    'maf_flow',
    'throttle_position',
    'fuel_pressure',
    'short_term_fuel_trim',
    'long_term_fuel_trim',
]

# Model hyperparameters
SEQUENCE_LENGTH = 60  # 60 seconds of data at 1 Hz
LATENT_DIM = 16
ENCODER_UNITS = 64
DECODER_UNITS = 64
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2


class LSTMAutoencoder:
    """LSTM Autoencoder for vehicle anomaly detection."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        n_features: int = len(OBD2_FEATURES),
        latent_dim: int = LATENT_DIM,
        encoder_units: int = ENCODER_UNITS,
        decoder_units: int = DECODER_UNITS,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.model = None
        self.encoder = None
        self.threshold = None
        self.scaler = StandardScaler()

    def build_model(self) -> Model:
        """Build the LSTM Autoencoder architecture."""
        # Encoder
        inputs = keras.Input(shape=(self.sequence_length, self.n_features))

        # LSTM Encoder layers
        x = layers.LSTM(
            self.encoder_units,
            activation='tanh',
            return_sequences=True,
            name='encoder_lstm_1'
        )(inputs)
        x = layers.Dropout(0.2)(x)

        x = layers.LSTM(
            self.latent_dim,
            activation='tanh',
            return_sequences=False,
            name='encoder_lstm_2'
        )(x)

        # Latent space
        latent = layers.Dense(self.latent_dim, activation='relu', name='latent')(x)

        # Decoder
        x = layers.RepeatVector(self.sequence_length)(latent)

        x = layers.LSTM(
            self.latent_dim,
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm_1'
        )(x)
        x = layers.Dropout(0.2)(x)

        x = layers.LSTM(
            self.decoder_units,
            activation='tanh',
            return_sequences=True,
            name='decoder_lstm_2'
        )(x)

        # Output layer
        outputs = layers.TimeDistributed(
            layers.Dense(self.n_features),
            name='output'
        )(x)

        # Build model
        self.model = Model(inputs, outputs, name='lstm_autoencoder')

        # Build encoder for latent space extraction
        self.encoder = Model(inputs, latent, name='encoder')

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def prepare_sequences(self, data: np.ndarray) -> np.ndarray:
        """Convert time series data into sequences for LSTM."""
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i + self.sequence_length])
        return np.array(sequences)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        callbacks: List = None,
    ) -> keras.callbacks.History:
        """Train the autoencoder model."""
        if self.model is None:
            self.build_model()

        # Fit scaler on training data
        X_train_flat = X_train.reshape(-1, self.n_features)
        self.scaler.fit(X_train_flat)

        # Transform data
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val) if X_val is not None else None

        validation_data = (X_val_scaled, X_val_scaled) if X_val_scaled is not None else None

        history = self.model.fit(
            X_train_scaled,
            X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )

        # Calculate threshold on training data
        self._calculate_threshold(X_train_scaled)

        return history

    def _scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Scale sequences using the fitted scaler."""
        original_shape = sequences.shape
        flat = sequences.reshape(-1, self.n_features)
        scaled = self.scaler.transform(flat)
        return scaled.reshape(original_shape)

    def _calculate_threshold(self, X: np.ndarray, percentile: float = 95):
        """Calculate anomaly threshold based on reconstruction errors."""
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, percentile)
        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict reconstruction and anomaly scores."""
        X_scaled = self._scale_sequences(X)
        reconstructions = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=(1, 2))
        return reconstructions, mse

    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in the input sequences."""
        _, mse = self.predict(X)
        anomalies = mse > self.threshold
        return anomalies, mse

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Get latent space representation of input sequences."""
        X_scaled = self._scale_sequences(X)
        return self.encoder.predict(X_scaled, verbose=0)

    def save(self, path: str):
        """Save model, scaler, and threshold."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Keras model
        self.model.save(path / 'model.keras')
        self.encoder.save(path / 'encoder.keras')

        # Save scaler and threshold
        joblib.dump(self.scaler, path / 'scaler.joblib')

        with open(path / 'config.json', 'w') as f:
            json.dump({
                'SequenceLength': self.sequence_length,
                'NumFeatures': self.n_features,
                'LatentDim': self.latent_dim,
                'encoder_units': self.encoder_units,
                'decoder_units': self.decoder_units,
                'Threshold': float(self.threshold) if self.threshold else None,
                'Features': OBD2_FEATURES,
                # Scaler parameters for C# service
                'ScalerMean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'ScalerStd': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            }, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'LSTMAutoencoder':
        """Load a saved model."""
        path = Path(path)

        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        instance = cls(
            sequence_length=config['sequence_length'],
            n_features=config['n_features'],
            latent_dim=config['latent_dim'],
            encoder_units=config['encoder_units'],
            decoder_units=config['decoder_units'],
        )

        instance.model = keras.models.load_model(path / 'model.keras')
        instance.encoder = keras.models.load_model(path / 'encoder.keras')
        instance.scaler = joblib.load(path / 'scaler.joblib')
        instance.threshold = config['threshold']

        return instance


def generate_synthetic_data(
    n_vehicles: int = 100,
    n_samples_per_vehicle: int = 3600,  # 1 hour of data per vehicle
    anomaly_rate: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic OBD2 data for training.

    In production, this would be replaced with real vehicle data.
    """
    logger.info(f"Generating synthetic data for {n_vehicles} vehicles...")

    all_data = []
    all_labels = []

    for vehicle_id in range(n_vehicles):
        # Base parameters for this vehicle (slight variations between vehicles)
        base_rpm = np.random.uniform(800, 1000)
        base_speed = 0
        base_load = np.random.uniform(15, 25)
        base_coolant = np.random.uniform(85, 95)

        vehicle_data = []
        vehicle_labels = []

        for t in range(n_samples_per_vehicle):
            # Simulate driving patterns
            driving_phase = (t % 600) / 600  # 10-minute cycles

            # Normal values with realistic variations
            if driving_phase < 0.2:  # Idle
                rpm = base_rpm + np.random.normal(0, 50)
                speed = 0
                load = base_load + np.random.normal(0, 5)
            elif driving_phase < 0.5:  # Acceleration
                rpm = base_rpm + 2000 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 100)
                speed = 100 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 5)
                load = 50 + 30 * (driving_phase - 0.2) / 0.3 + np.random.normal(0, 10)
            elif driving_phase < 0.8:  # Cruising
                rpm = 2500 + np.random.normal(0, 100)
                speed = 100 + np.random.normal(0, 5)
                load = 40 + np.random.normal(0, 5)
            else:  # Deceleration
                rpm = 2500 - 1500 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 100)
                speed = 100 - 100 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 5)
                load = 40 - 25 * (driving_phase - 0.8) / 0.2 + np.random.normal(0, 5)

            # Other parameters
            coolant_temp = base_coolant + min(10, t / 360) + np.random.normal(0, 2)
            intake_temp = 25 + speed * 0.1 + np.random.normal(0, 3)
            maf_flow = load * 0.5 + np.random.normal(0, 2)
            throttle = load * 0.8 + np.random.normal(0, 5)
            fuel_pressure = 350 + np.random.normal(0, 10)
            stft = np.random.normal(0, 3)
            ltft = np.random.normal(0, 2)

            is_anomaly = False

            # Inject anomalies
            if np.random.random() < anomaly_rate:
                is_anomaly = True
                anomaly_type = np.random.choice([
                    'misfire', 'overheating', 'fuel_issue', 'sensor_fault'
                ])

                if anomaly_type == 'misfire':
                    rpm += np.random.uniform(-500, 500)
                    load += np.random.uniform(-20, 20)
                elif anomaly_type == 'overheating':
                    coolant_temp += np.random.uniform(10, 30)
                elif anomaly_type == 'fuel_issue':
                    stft += np.random.uniform(-15, 15)
                    ltft += np.random.uniform(-10, 10)
                elif anomaly_type == 'sensor_fault':
                    idx = np.random.randint(0, len(OBD2_FEATURES))
                    # Will be handled below

            # Clamp values to realistic ranges
            rpm = np.clip(rpm, 0, 8000)
            speed = np.clip(speed, 0, 250)
            load = np.clip(load, 0, 100)
            coolant_temp = np.clip(coolant_temp, -40, 150)
            intake_temp = np.clip(intake_temp, -40, 80)
            maf_flow = np.clip(maf_flow, 0, 200)
            throttle = np.clip(throttle, 0, 100)
            fuel_pressure = np.clip(fuel_pressure, 0, 800)
            stft = np.clip(stft, -25, 25)
            ltft = np.clip(ltft, -25, 25)

            sample = [
                rpm, speed, load, coolant_temp, intake_temp,
                maf_flow, throttle, fuel_pressure, stft, ltft
            ]

            vehicle_data.append(sample)
            vehicle_labels.append(1 if is_anomaly else 0)

        all_data.append(np.array(vehicle_data))
        all_labels.append(np.array(vehicle_labels))

    # Stack all vehicle data
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Generated {len(data)} samples, {labels.sum()} anomalies ({labels.mean()*100:.1f}%)")

    return data, labels


def export_to_onnx(model: LSTMAutoencoder, output_path: str):
    """Export the model to ONNX format for desktop deployment."""
    try:
        import tf2onnx

        logger.info("Exporting model to ONNX format...")

        # Convert to ONNX
        input_signature = [
            tf.TensorSpec(
                shape=(None, model.sequence_length, model.n_features),
                dtype=tf.float32,
                name='input'
            )
        ]

        onnx_model, _ = tf2onnx.convert.from_keras(
            model.model,
            input_signature=input_signature,
            opset=13,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"ONNX model saved to {output_path}")

    except ImportError:
        logger.warning("tf2onnx not installed. Skipping ONNX export.")
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
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate or load data
    if args.generate_synthetic:
        data, labels = generate_synthetic_data(n_vehicles=args.n_vehicles)
    else:
        data_path = Path(args.data_dir) / 'training_data.npy'
        if data_path.exists():
            logger.info(f"Loading data from {data_path}")
            data = np.load(data_path)
            labels = np.zeros(len(data))  # Assume normal data for training
        else:
            logger.warning(f"No data found at {data_path}. Generating synthetic data.")
            data, labels = generate_synthetic_data(n_vehicles=args.n_vehicles)

    # Initialize model
    autoencoder = LSTMAutoencoder()
    autoencoder.build_model()

    logger.info("Model architecture:")
    autoencoder.model.summary()

    # Prepare sequences
    logger.info("Preparing sequences...")
    sequences = autoencoder.prepare_sequences(data)

    # Split into train/validation
    X_train, X_val = train_test_split(
        sequences, test_size=VALIDATION_SPLIT, random_state=42
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")

    # Setup callbacks
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(output_dir / f'checkpoint_{timestamp}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        TensorBoard(
            log_dir=str(output_dir / 'logs' / timestamp),
            histogram_freq=1,
        ),
    ]

    # Train
    logger.info("Starting training...")
    history = autoencoder.fit(
        X_train,
        X_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
    )

    # Save model
    model_path = output_dir / f'lstm_autoencoder_{timestamp}'
    autoencoder.save(model_path)

    # Export to ONNX for desktop deployment
    onnx_path = output_dir / 'onnx' / 'anomaly_detector.onnx'
    export_to_onnx(autoencoder, onnx_path)

    # Save training history
    history_path = output_dir / f'training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
        }, f, indent=2)

    # Evaluate on validation set
    logger.info("Evaluating model...")
    anomalies, scores = autoencoder.detect_anomalies(X_val)

    logger.info(f"Validation anomaly rate: {anomalies.mean()*100:.2f}%")
    logger.info(f"Mean reconstruction error: {scores.mean():.6f}")
    logger.info(f"Threshold: {autoencoder.threshold:.6f}")

    logger.info("Training complete!")

    return autoencoder


if __name__ == '__main__':
    main()
