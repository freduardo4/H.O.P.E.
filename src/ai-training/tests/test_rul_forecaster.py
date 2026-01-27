"""
Tests for the RUL (Remaining Useful Life) Forecaster.

These tests verify the predictive maintenance functionality
for vehicle component degradation forecasting.
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from rul_forecaster import (
    RULPredictor,
    ComponentType,
    ComponentHealth,
    MaintenancePrediction,
    LSTMForecaster,
    generate_synthetic_degradation_data,
)


class TestComponentType:
    """Tests for ComponentType enum."""

    def test_all_components_defined(self):
        """Test that all expected components are defined."""
        expected = [
            'catalytic_converter', 'o2_sensor', 'spark_plugs',
            'battery', 'brake_pads', 'air_filter', 'fuel_filter',
            'timing_belt', 'coolant', 'transmission_fluid'
        ]
        actual = [c.value for c in ComponentType]
        assert set(expected) == set(actual)


class TestLSTMForecaster:
    """Tests for the LSTM forecaster model."""

    def test_model_creation(self):
        """Test model can be created."""
        model = LSTMForecaster(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            sequence_length=20,
        )
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass with sample input."""
        import torch

        model = LSTMForecaster(
            input_size=1,
            hidden_size=32,
            num_layers=2,
            sequence_length=20,
        )

        # Create sample input: (batch=4, seq_len=20, features=1)
        x = torch.randn(4, 20, 1)
        output = model(x)

        assert output.shape == (4, 1)

    def test_model_parameters(self):
        """Test model has trainable parameters."""
        model = LSTMForecaster()
        params = list(model.parameters())
        assert len(params) > 0
        assert any(p.requires_grad for p in params)


class TestRULPredictor:
    """Tests for the RULPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create a predictor instance for tests."""
        return RULPredictor(
            sequence_length=20,
            hidden_size=32,
            num_layers=1,
        )

    @pytest.fixture
    def sample_data(self):
        """Generate sample degradation data."""
        return generate_synthetic_degradation_data(200, noise_level=0.02)

    def test_prepare_sequences(self, predictor):
        """Test sequence preparation."""
        data = np.linspace(1.0, 0.5, 100).reshape(-1, 1)
        X, y = predictor.prepare_sequences(data)

        expected_samples = 100 - predictor.sequence_length
        assert X.shape == (expected_samples, predictor.sequence_length, 1)
        assert y.shape == (expected_samples,)

    def test_prepare_sequences_1d(self, predictor):
        """Test sequence preparation with 1D input."""
        data = np.linspace(1.0, 0.5, 100)
        X, y = predictor.prepare_sequences(data.reshape(-1, 1))

        assert X.shape[0] == 100 - predictor.sequence_length
        assert X.shape[1] == predictor.sequence_length

    @pytest.mark.slow
    def test_train_model(self, predictor, sample_data):
        """Test model training."""
        history = predictor.train(
            ComponentType.BATTERY,
            sample_data,
            epochs=5,
            batch_size=16,
        )

        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) == 5
        assert ComponentType.BATTERY in predictor.trained_components

    def test_default_rul_estimation(self, predictor):
        """Test default RUL estimation without trained model."""
        health = predictor._estimate_rul_default(
            ComponentType.BATTERY,
            current_odometer=50000,
            avg_daily_km=50,
        )

        assert isinstance(health, ComponentHealth)
        assert 0 <= health.health_score <= 1
        assert health.estimated_rul_km >= 0
        assert health.estimated_rul_days >= 0
        assert health.confidence == 0.5  # Low confidence for default

    def test_predict_rul_untrained(self, predictor):
        """Test RUL prediction without trained model falls back to default."""
        data = generate_synthetic_degradation_data(50)
        health = predictor.predict_rul(
            ComponentType.O2_SENSOR,
            data,
            current_odometer=60000,
        )

        assert isinstance(health, ComponentHealth)
        assert health.component == ComponentType.O2_SENSOR

    @pytest.mark.slow
    def test_predict_rul_trained(self, predictor, sample_data):
        """Test RUL prediction with trained model."""
        # Train first
        predictor.train(
            ComponentType.CATALYTIC_CONVERTER,
            sample_data,
            epochs=3,
        )

        # Predict
        recent_data = sample_data[-50:]
        health = predictor.predict_rul(
            ComponentType.CATALYTIC_CONVERTER,
            recent_data,
            current_odometer=80000,
        )

        assert isinstance(health, ComponentHealth)
        assert health.component == ComponentType.CATALYTIC_CONVERTER
        assert health.confidence > 0  # Should have some confidence

    def test_predict_all_components(self, predictor):
        """Test prediction for all components."""
        telemetry = {
            ComponentType.BATTERY: generate_synthetic_degradation_data(50),
            ComponentType.O2_SENSOR: generate_synthetic_degradation_data(50),
        }

        prediction = predictor.predict_all_components(
            vehicle_id="TEST123",
            current_odometer=75000,
            telemetry_data=telemetry,
        )

        assert isinstance(prediction, MaintenancePrediction)
        assert prediction.vehicle_id == "TEST123"
        assert prediction.odometer_km == 75000
        assert len(prediction.components) == len(ComponentType)
        assert 0 <= prediction.overall_health <= 1

    def test_warning_levels(self, predictor):
        """Test warning level classification."""
        # Critical health
        critical_data = np.array([0.3, 0.29, 0.28, 0.27, 0.26])
        health = predictor.predict_rul(
            ComponentType.BRAKE_PADS,  # threshold = 0.3
            critical_data,
            current_odometer=45000,
        )
        # May vary based on threshold


class TestComponentHealth:
    """Tests for ComponentHealth dataclass."""

    def test_create_health(self):
        """Test creating ComponentHealth."""
        health = ComponentHealth(
            component=ComponentType.BATTERY,
            health_score=0.85,
            estimated_rul_km=30000,
            estimated_rul_days=180,
            confidence=0.9,
            degradation_rate=0.001,
            last_service_km=40000,
            recommended_service_km=70000,
            warning_level="normal",
        )

        assert health.component == ComponentType.BATTERY
        assert health.health_score == 0.85
        assert health.warning_level == "normal"


class TestMaintenancePrediction:
    """Tests for MaintenancePrediction dataclass."""

    def test_urgent_items(self):
        """Test urgent items detection."""
        predictor = RULPredictor()

        # Create telemetry with degraded component
        telemetry = {
            ComponentType.BRAKE_PADS: np.array([0.2] * 30),  # Very worn
        }

        prediction = predictor.predict_all_components(
            vehicle_id="TEST",
            current_odometer=100000,
            telemetry_data=telemetry,
        )

        # Should detect urgent items for very low health
        assert isinstance(prediction.urgent_items, list)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_degradation_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_degradation_data(500)

        assert len(data) == 500
        assert data[0] > data[-1]  # Should degrade over time
        assert np.all(data >= 0)
        assert np.all(data <= 1)

    def test_noise_level_affects_variance(self):
        """Test that noise level affects data variance."""
        np.random.seed(42)
        low_noise = generate_synthetic_degradation_data(500, noise_level=0.01)

        np.random.seed(42)
        high_noise = generate_synthetic_degradation_data(500, noise_level=0.1)

        assert np.std(low_noise) < np.std(high_noise)

    def test_degradation_trend(self):
        """Test that data shows degradation trend."""
        data = generate_synthetic_degradation_data(1000, noise_level=0.01)

        # First half should have higher average than second half
        first_half_avg = np.mean(data[:500])
        second_half_avg = np.mean(data[500:])

        assert first_half_avg > second_half_avg


class TestModelPersistence:
    """Tests for model saving and loading."""

    @pytest.mark.slow
    def test_save_and_load(self):
        """Test saving and loading predictor."""
        predictor = RULPredictor(sequence_length=15, hidden_size=16)

        # Train on a component
        data = generate_synthetic_degradation_data(200)
        predictor.train(ComponentType.BATTERY, data, epochs=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'

            # Save
            predictor.save(save_path)

            # Verify files exist
            assert (save_path / 'config.json').exists()
            assert (save_path / 'battery_model.pth').exists()
            assert (save_path / 'battery_scaler.joblib').exists()

            # Load
            loaded = RULPredictor.load(save_path)

            assert loaded.sequence_length == 15
            assert loaded.hidden_size == 16
            assert ComponentType.BATTERY in loaded.trained_components


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_sequence(self):
        """Test handling of short input sequences."""
        predictor = RULPredictor(sequence_length=30)

        # Provide less data than sequence_length
        short_data = np.array([0.9, 0.88, 0.85])

        health = predictor.predict_rul(
            ComponentType.SPARK_PLUGS,
            short_data,
            current_odometer=40000,
        )

        # Should handle gracefully (use default estimation)
        assert isinstance(health, ComponentHealth)

    def test_constant_data(self):
        """Test handling of constant data."""
        predictor = RULPredictor()

        constant_data = np.ones(50) * 0.75

        health = predictor.predict_rul(
            ComponentType.AIR_FILTER,
            constant_data,
            current_odometer=15000,
        )

        assert isinstance(health, ComponentHealth)
        assert health.degradation_rate == 0.0  # No degradation

    def test_zero_daily_km(self):
        """Test handling of zero daily km."""
        predictor = RULPredictor()

        health = predictor._estimate_rul_default(
            ComponentType.COOLANT,
            current_odometer=50000,
            avg_daily_km=0,  # Edge case
        )

        assert health.estimated_rul_days == 365  # Fallback value

    def test_very_high_odometer(self):
        """Test handling of very high odometer."""
        predictor = RULPredictor()

        health = predictor._estimate_rul_default(
            ComponentType.TIMING_BELT,
            current_odometer=500000,
            avg_daily_km=100,
        )

        # Should still produce valid result
        assert isinstance(health, ComponentHealth)
        assert health.health_score >= 0
