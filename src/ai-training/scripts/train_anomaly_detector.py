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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
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

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for vehicle anomaly detection."""

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        n_features: int = len(OBD2_FEATURES),
        latent_dim: int = LATENT_DIM,
        encoder_units: int = ENCODER_UNITS,
        decoder_units: int = DECODER_UNITS,
    ):
        super(LSTMAutoencoder, self).__init__()
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.threshold = None
        
        # Encoder
        self.encoder_lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=encoder_units,
            batch_first=True
        )
        self.encoder_dropout = nn.Dropout(0.2)
        self.encoder_lstm2 = nn.LSTM(
            input_size=encoder_units,
            hidden_size=latent_dim,
            batch_first=True
        )
        self.latent_layer = nn.Linear(latent_dim, latent_dim)
        self.latent_activation = nn.ReLU()
        
        # Decoder
        self.decoder_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            batch_first=True
        )
        self.decoder_dropout = nn.Dropout(0.2)
        self.decoder_lstm2 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=decoder_units,
            batch_first=True
        )
        self.output_layer = nn.Linear(decoder_units, n_features)

    def forward(self, x):
        # Encoder
        x, _ = self.encoder_lstm1(x)
        x = self.encoder_dropout(x)
        _, (hidden, _) = self.encoder_lstm2(x)
        
        # Latent space (using hidden state from last timestep)
        # hidden shape: (1, batch, latent_dim) -> (batch, latent_dim)
        latent = hidden[-1]
        latent = self.latent_layer(latent)
        latent = self.latent_activation(latent)
        
        # Decoder
        # Repeat vector: (batch, latent) -> (batch, seq, latent)
        x = latent.unsqueeze(1).repeat(1, self.sequence_length, 1)
        
        x, _ = self.decoder_lstm1(x)
        x = self.decoder_dropout(x)
        x, _ = self.decoder_lstm2(x)
        
        # Output layer applied to each timestep
        outputs = self.output_layer(x)
        
        return outputs

    def get_latent_representation(self, x):
        self.eval()
        with torch.no_grad():
            x, _ = self.encoder_lstm1(x)
            _, (hidden, _) = self.encoder_lstm2(x)
            latent = hidden[-1]
            return self.latent_activation(self.latent_layer(latent))

class AnomalyDetector:
    """Wrapper for training and using the LSTM Autoencoder."""
    
    def __init__(self, model_params: Dict[str, Any] = None):
        self.model_params = model_params or {}
        self.model = LSTMAutoencoder(**self.model_params).to(DEVICE)
        self.scaler = StandardScaler()
        self.threshold = None
        
    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Convert time series data into sequences for LSTM."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray = None,
        epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
    ) -> Dict[str, List[float]]:
        """Train the autoencoder model."""
        
        # Fit scaler on training data
        n_features = X_train.shape[1] if len(X_train.shape) == 2 else X_train.shape[2]
        X_train_flat = X_train.reshape(-1, n_features)
        self.scaler.fit(X_train_flat)
        
        # Transform data
        X_train_scaled = self._scale_sequences(X_train)
        X_val_scaled = self._scale_sequences(X_val) if X_val is not None else None
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_scaled),
            torch.FloatTensor(X_train_scaled) # Target is same as input
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val_scaled is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(X_val_scaled)
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        logger.info(f"Training on {DEVICE}")
        
        for epoch in range(epochs):
            # Training loop
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                
            train_loss /= len(train_loader.dataset)
            history['loss'].append(train_loss)
            
            # Validation loop
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    
            logger.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")
            
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
                
        # Load best model
        if val_loader and os.path.exists('best_model.pth'):
            self.model.load_state_dict(torch.load('best_model.pth'))
            os.remove('best_model.pth')
            
        # Calculate threshold on training data
        self._calculate_threshold(X_train_scaled)
        
        return history

    def _scale_sequences(self, sequences: np.ndarray) -> np.ndarray:
        """Scale sequences using the fitted scaler."""
        original_shape = sequences.shape
        # Handle cases where input is already sequences or raw 2D data
        if len(original_shape) == 3:
            flat = sequences.reshape(-1, original_shape[2])
            scaled = self.scaler.transform(flat)
            return scaled.reshape(original_shape)
        else:
            return self.scaler.transform(sequences)

    def _calculate_threshold(self, X: np.ndarray, percentile: float = 95):
        """Calculate anomaly threshold based on reconstruction errors."""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X).to(DEVICE)
            reconstructions = self.model(inputs).cpu().numpy()
            
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, percentile)
        self.model.threshold = float(self.threshold)
        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")

    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in the input sequences."""
        X_scaled = self._scale_sequences(X)
        
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X_scaled).to(DEVICE)
            reconstructions = self.model(inputs).cpu().numpy()
            
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=(1, 2))
        anomalies = mse > self.threshold
        return anomalies, mse

    def save(self, path: str):
        """Save model, scaler, and threshold."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model
        torch.save(self.model.state_dict(), path / 'model.pth')
        
        # Save scaler
        joblib.dump(self.scaler, path / 'scaler.joblib')

        with open(path / 'config.json', 'w') as f:
            json.dump({
                'SequenceLength': self.model.sequence_length,
                'NumFeatures': self.model.n_features,
                'LatentDim': self.model.latent_dim,
                'encoder_units': self.model.encoder_units,
                'decoder_units': self.model.decoder_units,
                'Threshold': float(self.threshold) if self.threshold else None,
                'Features': OBD2_FEATURES,
                # Scaler parameters for C# service
                'ScalerMean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'ScalerStd': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            }, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load a saved model."""
        path = Path(path)

        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        params = {
            'sequence_length': config.get('SequenceLength', config.get('sequence_length')),
            'n_features': config.get('NumFeatures', config.get('n_features')),
            'latent_dim': config.get('LatentDim', config.get('latent_dim')),
            'encoder_units': config.get('encoder_units', 64),
            'decoder_units': config.get('decoder_units', 64),
        }
        
        instance = cls(params)
        instance.model.load_state_dict(torch.load(path / 'model.pth', map_location=DEVICE))
        instance.scaler = joblib.load(path / 'scaler.joblib')
        instance.threshold = config.get('Threshold', config.get('threshold'))
        instance.model.threshold = instance.threshold

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
                    # Random noise for sensor fault
                    pass 

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


def export_to_onnx(detector: AnomalyDetector, output_path: str):
    """Export the model to ONNX format for desktop deployment."""
    try:
        logger.info("Exporting model to ONNX format...")

        # Dummy input for tracing
        dummy_input = torch.randn(1, detector.model.sequence_length, detector.model.n_features).to(DEVICE)
        
        output_path = Path(output_path)
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
    logger.info(f"Threshold: {detector.threshold:.6f}")

    logger.info("Training complete!")

    return detector


if __name__ == '__main__':
    main()
