import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler

from .config import (
    SEQUENCE_LENGTH,
    OBD2_FEATURES,
    LATENT_DIM,
    ENCODER_UNITS,
    DECODER_UNITS,
    DEVICE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE
)

logger = logging.getLogger(__name__)

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
        self.threshold: Optional[float] = None
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            x, _ = self.encoder_lstm1(x)
            _, (hidden, _) = self.encoder_lstm2(x)
            latent = hidden[-1]
            return self.latent_activation(self.latent_layer(latent))

class AnomalyDetector:
    """Wrapper for training and using the LSTM Autoencoder."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        self.model_params = model_params or {}
        self.model = LSTMAutoencoder(**self.model_params).to(DEVICE)
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        
    def prepare_sequences(self, data: np.ndarray, sequence_length: int) -> np.ndarray:
        """Convert time series data into sequences for LSTM."""
        sequences = []
        for i in range(len(data) - sequence_length + 1):
            sequences.append(data[i:i + sequence_length])
        return np.array(sequences)

    def fit(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
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
        from torch.utils.data import DataLoader, TensorDataset
        
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        history: Dict[str, List[float]] = {'loss': [], 'val_loss': []}
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
        if val_loader and Path('best_model.pth').exists():
            self.model.load_state_dict(torch.load('best_model.pth'))
            Path('best_model.pth').unlink()
            
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

    def _calculate_threshold(self, X: np.ndarray, percentile: float = 95) -> None:
        """Calculate anomaly threshold based on reconstruction errors."""
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X).to(DEVICE)
            reconstructions = self.model(inputs).cpu().numpy()
            
        mse = np.mean(np.power(X - reconstructions, 2), axis=(1, 2))
        self.threshold = float(np.percentile(mse, percentile))
        self.model.threshold = self.threshold
        logger.info(f"Anomaly threshold set to: {self.threshold:.6f}")

    def detect_anomalies(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect anomalies in the input sequences."""
        X_scaled = self._scale_sequences(X)
        
        self.model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(X_scaled).to(DEVICE)
            reconstructions = self.model(inputs).cpu().numpy()
            
        mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=(1, 2))
        
        if self.threshold is None:
             raise ValueError("Threshold not set. Train the model first.")

        anomalies = mse > self.threshold
        return anomalies, mse

    def save(self, path: Path) -> None:
        """Save model, scaler, and threshold."""
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
                'Threshold': self.threshold,
                'Features': OBD2_FEATURES,
                # Scaler parameters for C# service
                'ScalerMean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'ScalerStd': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
            }, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> 'AnomalyDetector':
        """Load a saved model."""
        
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
