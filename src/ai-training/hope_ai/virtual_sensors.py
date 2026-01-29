import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .pinn import PINNModel, PhysicsLoss
from .config import DEVICE, BATCH_SIZE, LEARNING_RATE, PHYSICS_WEIGHT
from .dataset import generate_synthetic_data

logger = logging.getLogger(__name__)

def train_pinn(epochs=50, save_path='models'):
    """Trains the PINN virtual sensor for EGT."""
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Data
    data, _ = generate_synthetic_data(n_vehicles=50)
    X = data[:, :11]
    y = data[:, 11].reshape(-1, 1)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE
    )

    # 3. Model & Training Setup
    model = PINNModel(n_inputs=11, n_outputs=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = PhysicsLoss(lambda_physics=PHYSICS_WEIGHT)
    
    def custom_forward(inputs):
        return model(inputs)
    criterion.forward_with_grad = custom_forward

    # 4. Training Loop
    logger.info(f"Starting PINN training on {DEVICE}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                loss = torch.nn.functional.mse_loss(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    # 5. Save Model and Scalers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = output_dir / f'egt_pinn_{timestamp}.pth'
    torch.save(model.state_dict(), model_save_path)
    joblib.dump(scaler_x, output_dir / f'scaler_x_{timestamp}.joblib')
    joblib.dump(scaler_y, output_dir / f'scaler_y_{timestamp}.joblib')
    
    logger.info(f"EGT PINN model saved to {model_save_path}")
    return model

class VirtualSensorBase(ABC):

    """Abstract base class for virtual sensors."""
    
    @abstractmethod
    def predict(self, inputs: Dict[str, Any]) -> float:
        """Estimate the target parameter based on inputs."""
        pass

class EGTVirtualSensor(VirtualSensorBase):
    """Virtual sensor for Exhaust Gas Temperature (EGT)."""
    
    def __init__(self, model_path: Optional[str] = None, scaler_x_path: Optional[str] = None, scaler_y_path: Optional[str] = None):
        self.model = None
        self.scaler_x = None
        self.scaler_y = None
        if model_path and scaler_x_path and scaler_y_path:
            self.load(model_path, scaler_x_path, scaler_y_path)
            
    def load(self, model_path: str, scaler_x_path: str, scaler_y_path: str):
        """Load the PINN model and scalers."""
        from .config import DEVICE
        import joblib
        
        self.model = PINNModel(n_inputs=11, n_outputs=1).to(DEVICE)
        try:
             self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
             self.model.eval()
             self.scaler_x = joblib.load(scaler_x_path)
             self.scaler_y = joblib.load(scaler_y_path)
             logger.info(f"EGT Model and scalers loaded from {model_path}")
        except Exception as e:
             logger.error(f"Failed to load EGT model or scalers: {e}")

    def predict(self, inputs: Dict[str, Any]) -> float:
        """Estimate EGT from OBD2 parameters."""
        if self.model is None or self.scaler_x is None or self.scaler_y is None:
            # Fallback to simple physical approximation if no model/scalers are loaded
            return self._physical_approximation(inputs)
            
        try:
            features = np.array([
                inputs.get('engine_rpm', 0),
                inputs.get('vehicle_speed', 0),
                inputs.get('engine_load', 0),
                inputs.get('coolant_temp', 0),
                inputs.get('intake_air_temp', 0),
                inputs.get('maf_flow', 0),
                inputs.get('throttle_position', 0),
                inputs.get('fuel_pressure', 0),
                inputs.get('short_term_fuel_trim', 0),
                inputs.get('long_term_fuel_trim', 0),
                inputs.get('ignition_timing', 0)
            ]).reshape(1, -1)
            
            # Scale input
            x_scaled = self.scaler_x.transform(features)
            x_tensor = torch.FloatTensor(x_scaled).to(next(self.model.parameters()).device)
            
            with torch.no_grad():
                y_scaled = self.model(x_tensor).cpu().numpy()
            
            # Inverse scale output
            egt = self.scaler_y.inverse_transform(y_scaled)
            
            return float(egt[0][0])
            
        except Exception as e:
            logger.error(f"EGT prediction failed: {e}")
            return self._physical_approximation(inputs)

    def _physical_approximation(self, inputs: Dict[str, Any]) -> float:
        """Simple thermodynamic approximation for EGT."""
        # See physical model defined in dataset.py
        iat = inputs.get('intake_air_temp', 25)
        load = inputs.get('engine_load', 20)
        rpm = inputs.get('engine_rpm', 800)
        ignition = inputs.get('ignition_timing', 10)
        
        return iat + (load * 6) + (rpm * 0.05) - (ignition * 5) + 300
