import logging
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from .config import DEVICE

logger = logging.getLogger(__name__)

class ComponentType(Enum):
    """Types of vehicle components tracked for RUL prediction."""
    CATALYTIC_CONVERTER = "catalytic_converter"
    O2_SENSOR = "o2_sensor"
    SPARK_PLUGS = "spark_plugs"
    BATTERY = "battery"
    BRAKE_PADS = "brake_pads"
    AIR_FILTER = "air_filter"
    FUEL_FILTER = "fuel_filter"
    TIMING_BELT = "timing_belt"
    COOLANT = "coolant"
    TRANSMISSION_FLUID = "transmission_fluid"

@dataclass
class ComponentHealth:
    """Health status of a vehicle component."""
    component: ComponentType
    health_score: float  # 0.0 (failed) to 1.0 (new)
    estimated_rul_km: float  # Estimated remaining useful life in km
    estimated_rul_days: int  # Estimated remaining useful life in days
    confidence: float  # Confidence in the prediction (0.0 to 1.0)
    degradation_rate: float  # Rate of degradation per 1000 km
    last_service_km: float  # Odometer at last service
    recommended_service_km: float  # Recommended next service odometer
    warning_level: str  # "normal", "warning", "critical"
    contributing_factors: List[str] = field(default_factory=list)

@dataclass
class MaintenancePrediction:
    """Complete maintenance prediction for a vehicle."""
    vehicle_id: str
    odometer_km: float
    prediction_date: datetime
    components: List[ComponentHealth]
    overall_health: float  # Average health score
    next_recommended_service: datetime
    urgent_items: List[str]
    estimated_maintenance_cost: float

class LSTMForecaster(nn.Module):
    """LSTM model for time-series RUL forecasting."""

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        sequence_length: int = 30,
    ):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use the last output
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)

class RULPredictor:
    """
    Remaining Useful Life predictor for vehicle components.
    """

    COMPONENT_PARAMS = {
        ComponentType.CATALYTIC_CONVERTER: {
            'typical_life_km': 150000,
            'features': ['catalyst_efficiency', 'o2_sensor_voltage', 'exhaust_temp'],
            'degradation_threshold': 0.7,
        },
        ComponentType.O2_SENSOR: {
            'typical_life_km': 100000,
            'features': ['o2_response_time', 'fuel_trim_variance', 'voltage_range'],
            'degradation_threshold': 0.6,
        },
        ComponentType.SPARK_PLUGS: {
            'typical_life_km': 50000,
            'features': ['misfire_count', 'ignition_timing_variance', 'idle_stability'],
            'degradation_threshold': 0.5,
        },
        ComponentType.BATTERY: {
            'typical_life_km': 80000,
            'features': ['cold_crank_voltage', 'alternator_voltage', 'parasitic_drain'],
            'degradation_threshold': 0.6,
        },
        ComponentType.BRAKE_PADS: {
            'typical_life_km': 50000,
            'features': ['brake_pressure_variance', 'stopping_distance', 'pad_wear_indicator'],
            'degradation_threshold': 0.3,
        },
        ComponentType.AIR_FILTER: {
            'typical_life_km': 20000,
            'features': ['maf_variance', 'intake_restriction', 'volumetric_efficiency'],
            'degradation_threshold': 0.5,
        },
    }

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
    ):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.models: Dict[ComponentType, LSTMForecaster] = {}
        self.scalers: Dict[ComponentType, MinMaxScaler] = {}
        self.trained_components: List[ComponentType] = []

    def _create_model(self, input_size: int = 1) -> LSTMForecaster:
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            sequence_length=self.sequence_length,
        ).to(DEVICE)

    def prepare_sequences(
        self,
        data: np.ndarray,
        target_col: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(data[i + self.sequence_length, target_col])
        return np.array(X), np.array(y)

    def train(
        self,
        component: ComponentType,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        logger.info(f"Training RUL model for {component.value}")

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1) if data.ndim == 1 else data)
        self.scalers[component] = scaler

        X, y = self.prepare_sequences(data_scaled)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        input_size = X_train.shape[2] if X_train.ndim == 3 else 1
        model = self._create_model(input_size)
        self.models[component] = model

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

        history = {'loss': [], 'val_loss': []}
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            train_loss /= len(train_loader.dataset)
            history['loss'].append(train_loss)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.trained_components.append(component)
        return history

    def predict_rul(
        self,
        component: ComponentType,
        recent_data: np.ndarray,
        current_odometer: float,
        avg_daily_km: float = 50.0,
    ) -> ComponentHealth:
        if component not in self.models:
            return self._estimate_rul_default(component, current_odometer, avg_daily_km, recent_data)

        model = self.models[component]
        scaler = self.scalers[component]

        data_scaled = scaler.transform(recent_data.reshape(-1, 1) if recent_data.ndim == 1 else recent_data)
        if len(data_scaled) < self.sequence_length:
            padding = np.repeat(data_scaled[:1], self.sequence_length - len(data_scaled), axis=0)
            data_scaled = np.concatenate([padding, data_scaled])

        sequence = data_scaled[-self.sequence_length:]
        X = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

        model.eval()
        predictions = []
        current_seq = X.clone()
        with torch.no_grad():
            for _ in range(100):
                pred = model(current_seq)
                predictions.append(pred.item())
                new_val = pred.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()

        current_health = float(recent_data[-1])
        if current_health > 1: current_health /= 100

        params = self.COMPONENT_PARAMS.get(component, {})
        threshold = params.get('degradation_threshold', 0.5)

        rul_steps = 100
        for i, pred in enumerate(predictions):
            if pred < threshold:
                rul_steps = i
                break

        rul_km = rul_steps * avg_daily_km
        rul_days = rul_steps

        if len(recent_data) > 1:
            degradation_rate = (recent_data[0] - recent_data[-1]) / len(recent_data) * 1000
        else:
            degradation_rate = 0.0

        warning_level = "critical" if current_health < threshold else "warning" if current_health < threshold + 0.2 else "normal"
        confidence = max(0.0, min(1.0, 1.0 - np.std(predictions)))

        return ComponentHealth(
            component=component,
            health_score=current_health,
            estimated_rul_km=rul_km,
            estimated_rul_days=rul_days,
            confidence=confidence,
            degradation_rate=degradation_rate,
            last_service_km=current_odometer - rul_km * 0.2, # Placeholder
            recommended_service_km=current_odometer + rul_km * 0.8,
            warning_level=warning_level,
            contributing_factors=self._identify_factors(component, recent_data),
        )

    def _estimate_rul_default(
        self,
        component: ComponentType,
        current_odometer: float,
        avg_daily_km: float,
        recent_data: Optional[np.ndarray] = None,
    ) -> ComponentHealth:
        params = self.COMPONENT_PARAMS.get(component, {'typical_life_km': 100000})
        typical_life = params['typical_life_km']
        estimated_age_km = current_odometer % typical_life
        health_score = max(0.0, 1.0 - (estimated_age_km / typical_life))

        if recent_data is not None and len(recent_data) > 1:
            total_degradation = recent_data[0] - recent_data[-1]
            steps_per_1000km = 1000 / avg_daily_km if avg_daily_km > 0 else 20
            degradation_rate = (total_degradation / len(recent_data)) * steps_per_1000km
            health_score = float(recent_data[-1])
            if health_score > 1 and health_score <= 100: health_score /= 100
        else:
            degradation_rate = typical_life / 1000

        rul_km = max(0, typical_life - estimated_age_km)
        rul_days = int(rul_km / avg_daily_km) if avg_daily_km > 0 else 365

        threshold = params.get('degradation_threshold', 0.5)
        warning_level = "critical" if health_score < threshold else "warning" if health_score < threshold + 0.2 else "normal"

        return ComponentHealth(
            component=component,
            health_score=health_score,
            estimated_rul_km=rul_km,
            estimated_rul_days=rul_days,
            confidence=0.5,
            degradation_rate=degradation_rate,
            last_service_km=current_odometer - estimated_age_km,
            recommended_service_km=current_odometer + rul_km * 0.8,
            warning_level=warning_level,
            contributing_factors=["Default estimation"] if recent_data is None else ["Trend analysis"],
        )

    def _identify_factors(self, component: ComponentType, data: np.ndarray) -> List[str]:
        factors = []
        if len(data) < 2: return factors
        if data[-1] - data[-min(10, len(data))] < -0.1: factors.append("Accelerated degradation detected")
        if np.std(data[-20:]) > 0.1: factors.append("High variability in sensors")
        return factors

    def predict_all_components(
        self,
        vehicle_id: str,
        current_odometer: float,
        telemetry_data: Dict[ComponentType, np.ndarray],
        avg_daily_km: float = 50.0,
    ) -> MaintenancePrediction:
        components = []
        urgent_items = []
        total_cost = 0.0

        for comp_type in ComponentType:
            health = self.predict_rul(comp_type, telemetry_data.get(comp_type), current_odometer, avg_daily_km) if comp_type in telemetry_data else self._estimate_rul_default(comp_type, current_odometer, avg_daily_km)
            components.append(health)
            if health.warning_level == "critical":
                urgent_items.append(f"{comp_type.value}: Urgent replacement")
                total_cost += 500 # Default cost
            elif health.warning_level == "warning":
                total_cost += 250

        overall_health = np.mean([c.health_score for c in components])
        min_rul_days = min(c.estimated_rul_days for c in components)
        next_service = datetime.now() + timedelta(days=max(1, int(min_rul_days * 0.8)))

        return MaintenancePrediction(
            vehicle_id=vehicle_id,
            odometer_km=current_odometer,
            prediction_date=datetime.now(),
            components=components,
            overall_health=overall_health,
            next_recommended_service=next_service,
            urgent_items=urgent_items,
            estimated_maintenance_cost=total_cost,
        )

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for component, model in self.models.items():
            torch.save(model.state_dict(), path / f"{component.value}_model.pth")
        for component, scaler in self.scalers.items():
            joblib.dump(scaler, path / f"{component.value}_scaler.joblib")
        with open(path / 'config.json', 'w') as f:
            json.dump({
                'sequence_length': self.sequence_length,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'trained_components': [c.value for c in self.trained_components],
            }, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'RULPredictor':
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)
        predictor = cls(config['sequence_length'], config['hidden_size'], config['num_layers'])
        for comp_val in config['trained_components']:
            comp = ComponentType(comp_val)
            predictor.scalers[comp] = joblib.load(path / f"{comp_val}_scaler.joblib")
            model = predictor._create_model()
            model.load_state_dict(torch.load(path / f"{comp_val}_model.pth", map_location=DEVICE))
            predictor.models[comp] = model
            predictor.trained_components.append(comp)
        return predictor

def train_rul(epochs=50, save_path='models'):
    """Trains the RUL forecaster for various components."""
    from pathlib import Path
    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    predictor = RULPredictor()
    
    # Simulate degradation data for training
    def generate_synthetic_degradation_data(n_samples=1000):
        base_curve = np.linspace(1.0, 0.3, n_samples)
        noise = np.random.normal(0, 0.05, n_samples)
        return np.clip(base_curve + noise, 0.0, 1.0)

    logger.info("Starting RUL Forecaster training...")
    for component in [
        ComponentType.CATALYTIC_CONVERTER,
        ComponentType.O2_SENSOR,
        ComponentType.BATTERY,
    ]:
        logger.info(f"Training for {component.value}...")
        data = generate_synthetic_degradation_data(1000)
        predictor.train(component, data, epochs=epochs)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = output_dir / f'rul_forecaster_{timestamp}'
    predictor.save(save_dir)
    logger.info(f"RUL Forecaster model saved to {save_dir}")
    return predictor

