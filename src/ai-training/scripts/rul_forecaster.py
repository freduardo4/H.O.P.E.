"""
HOPE RUL (Remaining Useful Life) Forecaster

This script implements time-series forecasting models for predicting component
degradation and remaining useful life in vehicles.

Features:
- LSTM-based time-series forecasting
- Multi-component degradation tracking
- Confidence intervals for predictions
- Early warning system integration

Tracked Components:
- Catalytic converter efficiency
- O2 sensor response time
- Spark plug degradation
- Battery health
- Brake pad wear

Usage:
    python rul_forecaster.py --model_path ../models/rul_model --data_path data.csv
    python rul_forecaster.py --train --data_dir ../data/maintenance
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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

    Uses LSTM forecasting to predict when components will need replacement
    based on historical telemetry and maintenance data.
    """

    # Component-specific parameters
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
            'typical_life_km': 80000,  # or ~4 years
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

        # Models for each component type
        self.models: Dict[ComponentType, LSTMForecaster] = {}
        self.scalers: Dict[ComponentType, MinMaxScaler] = {}
        self.trained_components: List[ComponentType] = []

    def _create_model(self, input_size: int = 1) -> LSTMForecaster:
        """Create a new LSTM model."""
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
        """
        Prepare sequences for training/prediction.

        Args:
            data: Time series data (samples x features)
            target_col: Column index to predict

        Returns:
            Tuple of (X sequences, y targets)
        """
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
        """
        Train the RUL model for a specific component.

        Args:
            component: Component type to train for
            data: Training data (time series of health indicators)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            validation_split: Validation data fraction

        Returns:
            Training history
        """
        logger.info(f"Training RUL model for {component.value}")

        # Scale data
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.reshape(-1, 1) if data.ndim == 1 else data)
        self.scalers[component] = scaler

        # Prepare sequences
        X, y = self.prepare_sequences(data_scaled)

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Create model
        input_size = X_train.shape[2] if X_train.ndim == 3 else 1
        model = self._create_model(input_size)
        self.models[component] = model

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )

        history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)
            history['loss'].append(train_loss)

            # Validation
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE)
                    y_batch = y_batch.to(DEVICE)

                    outputs = model(X_batch).squeeze()
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)

            val_loss /= len(val_loader.dataset)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            if (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        self.trained_components.append(component)
        logger.info(f"Training complete for {component.value}")

        return history

    def predict_rul(
        self,
        component: ComponentType,
        recent_data: np.ndarray,
        current_odometer: float,
        avg_daily_km: float = 50.0,
    ) -> ComponentHealth:
        """
        Predict remaining useful life for a component.

        Args:
            component: Component type
            recent_data: Recent telemetry data (at least sequence_length samples)
            current_odometer: Current odometer reading in km
            avg_daily_km: Average daily driving distance

        Returns:
            ComponentHealth prediction
        """
        if component not in self.models:
            # Use default estimation if model not trained
            return self._estimate_rul_default(component, current_odometer, avg_daily_km, recent_data)

        model = self.models[component]
        scaler = self.scalers[component]

        # Scale input data
        data_scaled = scaler.transform(
            recent_data.reshape(-1, 1) if recent_data.ndim == 1 else recent_data
        )

        # Ensure we have enough data
        if len(data_scaled) < self.sequence_length:
            # Pad with first value
            padding = np.repeat(data_scaled[:1], self.sequence_length - len(data_scaled), axis=0)
            data_scaled = np.concatenate([padding, data_scaled])

        # Use most recent sequence
        sequence = data_scaled[-self.sequence_length:]
        X = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

        # Predict future values
        model.eval()
        predictions = []
        current_seq = X.clone()

        # Predict next 100 time steps
        with torch.no_grad():
            for _ in range(100):
                pred = model(current_seq)
                predictions.append(pred.item())

                # Shift sequence and add prediction
                new_val = pred.unsqueeze(1)
                current_seq = torch.cat([current_seq[:, 1:, :], new_val], dim=1)

        # Convert predictions back to original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = scaler.inverse_transform(predictions).flatten()

        # Calculate health score (current value)
        current_health = float(recent_data[-1])
        if current_health > 1:
            current_health = current_health / 100  # Normalize if percentage

        # Find when health drops below threshold
        params = self.COMPONENT_PARAMS.get(component, {})
        threshold = params.get('degradation_threshold', 0.5)

        rul_steps = 100  # Default
        for i, pred in enumerate(predictions):
            if pred < threshold:
                rul_steps = i
                break

        # Convert steps to km and days
        km_per_step = avg_daily_km  # Assuming 1 step = 1 day
        rul_km = rul_steps * km_per_step
        rul_days = rul_steps

        # Calculate degradation rate
        if len(recent_data) > 1:
            degradation_rate = (recent_data[0] - recent_data[-1]) / len(recent_data) * 1000
        else:
            degradation_rate = 0.0

        # Determine warning level
        if current_health < threshold:
            warning_level = "critical"
        elif current_health < threshold + 0.2:
            warning_level = "warning"
        else:
            warning_level = "normal"

        # Calculate confidence based on prediction variance
        pred_std = np.std(predictions)
        confidence = max(0.0, min(1.0, 1.0 - pred_std))

        return ComponentHealth(
            component=component,
            health_score=current_health,
            estimated_rul_km=rul_km,
            estimated_rul_days=rul_days,
            confidence=confidence,
            degradation_rate=degradation_rate,
            last_service_km=current_odometer - rul_km * 0.2,  # Estimate
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
        """Default RUL estimation when model is not trained."""
        params = self.COMPONENT_PARAMS.get(component, {'typical_life_km': 100000})
        typical_life = params['typical_life_km']

        # Simple linear degradation model
        estimated_age_km = current_odometer % typical_life
        health_score = max(0.0, 1.0 - (estimated_age_km / typical_life))

        # Calculate degradation rate from actual data if available
        if recent_data is not None and len(recent_data) > 1:
            # Calculate degradation per step
            total_degradation = recent_data[0] - recent_data[-1]
            steps = len(recent_data)
            
            # If data is constant (rate ~ 0), ensure it's not negative due to noise
            if abs(total_degradation) < 1e-6:
                degradation_rate = 0.0
            else:
                # Normalize to rate per 1000 km
                # Assuming 1 step ~ 1 day ~ avg_daily_km
                steps_per_1000km = 1000 / avg_daily_km if avg_daily_km > 0 else 20
                degradation_rate = (total_degradation / steps) * steps_per_1000km
                
            # Use actual health score if available
            health_score = float(recent_data[-1])
            if health_score > 1 and health_score <= 100:
                health_score /= 100
        else:
            degradation_rate = typical_life / 1000

        rul_km = max(0, typical_life - estimated_age_km)
        rul_days = int(rul_km / avg_daily_km) if avg_daily_km > 0 else 365

        threshold = params.get('degradation_threshold', 0.5)
        if health_score < threshold:
            warning_level = "critical"
        elif health_score < threshold + 0.2:
            warning_level = "warning"
        else:
            warning_level = "normal"

        return ComponentHealth(
            component=component,
            health_score=health_score,
            estimated_rul_km=rul_km,
            estimated_rul_days=rul_days,
            confidence=0.5,  # Low confidence for default estimation
            degradation_rate=degradation_rate,
            last_service_km=current_odometer - estimated_age_km,
            recommended_service_km=current_odometer + rul_km * 0.8,
            warning_level=warning_level,
            contributing_factors=["Default estimation - no sensor data available"] if recent_data is None else ["Trend analysis (Untrained)"],
        )

    def _identify_factors(
        self,
        component: ComponentType,
        data: np.ndarray,
    ) -> List[str]:
        """Identify factors contributing to degradation."""
        factors = []

        if len(data) < 2:
            return factors

        # Check for rapid degradation
        recent_change = data[-1] - data[-min(10, len(data))]
        if recent_change < -0.1:
            factors.append("Accelerated degradation detected")

        # Check for instability
        if np.std(data[-20:]) > 0.1:
            factors.append("High variability in sensor readings")

        # Component-specific factors
        if component == ComponentType.CATALYTIC_CONVERTER:
            if data[-1] < 0.8:
                factors.append("Reduced catalyst efficiency")
        elif component == ComponentType.O2_SENSOR:
            if np.mean(data[-10:]) < 0.7:
                factors.append("Slow O2 sensor response")
        elif component == ComponentType.BATTERY:
            if data[-1] < 0.7:
                factors.append("Reduced cold cranking capacity")

        return factors

    def predict_all_components(
        self,
        vehicle_id: str,
        current_odometer: float,
        telemetry_data: Dict[ComponentType, np.ndarray],
        avg_daily_km: float = 50.0,
    ) -> MaintenancePrediction:
        """
        Generate complete maintenance prediction for all components.

        Args:
            vehicle_id: Vehicle identifier
            current_odometer: Current odometer in km
            telemetry_data: Dict mapping component to sensor data
            avg_daily_km: Average daily driving distance

        Returns:
            Complete maintenance prediction
        """
        components = []
        urgent_items = []
        total_cost = 0.0

        # Cost estimates by component
        cost_estimates = {
            ComponentType.CATALYTIC_CONVERTER: 1500,
            ComponentType.O2_SENSOR: 200,
            ComponentType.SPARK_PLUGS: 150,
            ComponentType.BATTERY: 200,
            ComponentType.BRAKE_PADS: 300,
            ComponentType.AIR_FILTER: 50,
            ComponentType.FUEL_FILTER: 100,
            ComponentType.TIMING_BELT: 800,
            ComponentType.COOLANT: 100,
            ComponentType.TRANSMISSION_FLUID: 200,
        }

        for component_type in ComponentType:
            if component_type in telemetry_data:
                health = self.predict_rul(
                    component_type,
                    telemetry_data[component_type],
                    current_odometer,
                    avg_daily_km,
                )
            else:
                health = self._estimate_rul_default(
                    component_type,
                    current_odometer,
                    avg_daily_km,
                )

            components.append(health)

            if health.warning_level == "critical":
                urgent_items.append(f"{component_type.value}: Immediate attention required")
                total_cost += cost_estimates.get(component_type, 200)
            elif health.warning_level == "warning":
                total_cost += cost_estimates.get(component_type, 200) * 0.5

        # Calculate overall health
        overall_health = np.mean([c.health_score for c in components])

        # Find next service date
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
        """Save all models and scalers."""
        path.mkdir(parents=True, exist_ok=True)

        # Save model states
        for component, model in self.models.items():
            model_path = path / f"{component.value}_model.pth"
            torch.save(model.state_dict(), model_path)

        # Save scalers
        for component, scaler in self.scalers.items():
            scaler_path = path / f"{component.value}_scaler.joblib"
            joblib.dump(scaler, scaler_path)

        # Save config
        config = {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'trained_components': [c.value for c in self.trained_components],
        }

        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Saved RUL predictor to {path}")

    @classmethod
    def load(cls, path: Path) -> 'RULPredictor':
        """Load predictor from saved files."""
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        predictor = cls(
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
        )

        # Load models and scalers
        for component_value in config['trained_components']:
            component = ComponentType(component_value)

            # Load scaler
            scaler_path = path / f"{component_value}_scaler.joblib"
            if scaler_path.exists():
                predictor.scalers[component] = joblib.load(scaler_path)

            # Load model
            model_path = path / f"{component_value}_model.pth"
            if model_path.exists():
                model = predictor._create_model()
                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                predictor.models[component] = model
                predictor.trained_components.append(component)

        logger.info(f"Loaded RUL predictor from {path}")
        return predictor


def generate_synthetic_degradation_data(
    n_samples: int = 1000,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Generate synthetic component degradation data for testing."""
    # Simulate gradual degradation with noise
    base_curve = np.linspace(1.0, 0.3, n_samples)

    # Add realistic degradation patterns
    # Occasional sudden drops (simulating damage events)
    sudden_drops = np.zeros(n_samples)
    drop_points = np.random.choice(n_samples, size=3, replace=False)
    for point in drop_points:
        sudden_drops[point:] -= np.random.uniform(0.02, 0.05)

    # Add noise
    noise = np.random.normal(0, noise_level, n_samples)

    # Combine
    data = base_curve + sudden_drops + noise
    data = np.clip(data, 0.0, 1.0)

    return data


def main():
    parser = argparse.ArgumentParser(description='RUL Forecaster for Vehicle Components')
    parser.add_argument('--model_path', type=str, default='../models/rul_model',
                        help='Path to save/load model')
    parser.add_argument('--data_path', type=str, help='Path to telemetry data')
    parser.add_argument('--train', action='store_true', help='Train new model')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test data')
    parser.add_argument('--vehicle_id', type=str, default='TEST001', help='Vehicle ID')
    parser.add_argument('--odometer', type=float, default=50000, help='Current odometer (km)')
    args = parser.parse_args()

    model_path = Path(args.model_path)

    if args.train:
        predictor = RULPredictor()

        # Train on synthetic or real data
        if args.synthetic:
            logger.info("Training on synthetic data")

            for component in [
                ComponentType.CATALYTIC_CONVERTER,
                ComponentType.O2_SENSOR,
                ComponentType.BATTERY,
            ]:
                data = generate_synthetic_degradation_data(1000)
                predictor.train(component, data, epochs=50)

        predictor.save(model_path)

    else:
        # Load or create predictor
        if model_path.exists():
            predictor = RULPredictor.load(model_path)
        else:
            predictor = RULPredictor()
            logger.info("No trained model found, using default estimations")

        # Generate predictions
        if args.synthetic:
            telemetry = {
                ComponentType.CATALYTIC_CONVERTER: generate_synthetic_degradation_data(100),
                ComponentType.O2_SENSOR: generate_synthetic_degradation_data(100),
                ComponentType.BATTERY: generate_synthetic_degradation_data(100),
            }
        else:
            telemetry = {}

        prediction = predictor.predict_all_components(
            vehicle_id=args.vehicle_id,
            current_odometer=args.odometer,
            telemetry_data=telemetry,
        )

        # Print results
        print("\n" + "=" * 60)
        print(f"MAINTENANCE PREDICTION - {prediction.vehicle_id}")
        print("=" * 60)
        print(f"Odometer: {prediction.odometer_km:,.0f} km")
        print(f"Overall Health: {prediction.overall_health:.1%}")
        print(f"Next Service: {prediction.next_recommended_service.strftime('%Y-%m-%d')}")
        print(f"Estimated Cost: ${prediction.estimated_maintenance_cost:,.0f}")

        if prediction.urgent_items:
            print("\n! URGENT ITEMS:")
            for item in prediction.urgent_items:
                print(f"  - {item}")

        print("\nCOMPONENT STATUS:")
        print("-" * 60)

        for comp in prediction.components:
            status_icon = {"normal": "[OK]", "warning": "[!!]", "critical": "[XX]"}[comp.warning_level]
            print(f"{status_icon} {comp.component.value:25} Health: {comp.health_score:.1%}")
            print(f"     RUL: {comp.estimated_rul_km:,.0f} km / {comp.estimated_rul_days} days")
            if comp.contributing_factors:
                for factor in comp.contributing_factors:
                    print(f"     - {factor}")
            print()

        print("=" * 60)


if __name__ == '__main__':
    main()
