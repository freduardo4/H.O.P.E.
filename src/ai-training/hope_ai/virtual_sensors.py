import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import torch
import numpy as np

from .pinn import PINNModel

logger = logging.getLogger(__name__)

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
