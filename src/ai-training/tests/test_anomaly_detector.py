"""
Tests for the LSTM Autoencoder Anomaly Detection model.

This test suite covers:
- Model architecture and initialization
- Sequence preparation
- Training pipeline
- Inference and anomaly detection
- Model serialization (save/load)
- ONNX export
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from train_anomaly_detector import (
    LSTMAutoencoder,
    AnomalyDetector,
    generate_synthetic_data,
    export_to_onnx,
    OBD2_FEATURES,
    SEQUENCE_LENGTH,
    LATENT_DIM,
    DEVICE,
)


class TestLSTMAutoencoder:
    """Tests for the LSTMAutoencoder model architecture."""

    def test_model_initialization(self):
        """Test that model initializes with correct architecture."""
        model = LSTMAutoencoder()

        assert model.sequence_length == SEQUENCE_LENGTH
        assert model.n_features == len(OBD2_FEATURES)
        assert model.latent_dim == LATENT_DIM

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = LSTMAutoencoder(
            sequence_length=30,
            n_features=5,
            latent_dim=8,
            encoder_units=32,
            decoder_units=32
        )

        assert model.sequence_length == 30
        assert model.n_features == 5
        assert model.latent_dim == 8
        assert model.encoder_units == 32
        assert model.decoder_units == 32

    def test_forward_pass_shape(self):
        """Test that forward pass produces correct output shape."""
        model = LSTMAutoencoder().to(DEVICE)
        batch_size = 8

        # Create dummy input
        x = torch.randn(batch_size, SEQUENCE_LENGTH, len(OBD2_FEATURES)).to(DEVICE)

        # Forward pass
        output = model(x)

        # Check output shape matches input shape (reconstruction)
        assert output.shape == x.shape

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass works with various batch sizes."""
        model = LSTMAutoencoder().to(DEVICE)

        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, SEQUENCE_LENGTH, len(OBD2_FEATURES)).to(DEVICE)
            output = model(x)
            assert output.shape == x.shape, f"Failed for batch_size={batch_size}"

    def test_get_latent_representation(self):
        """Test that latent representation has correct dimensions."""
        model = LSTMAutoencoder().to(DEVICE)
        batch_size = 8

        x = torch.randn(batch_size, SEQUENCE_LENGTH, len(OBD2_FEATURES)).to(DEVICE)
        latent = model.get_latent_representation(x)

        assert latent.shape == (batch_size, LATENT_DIM)

    def test_model_is_differentiable(self):
        """Test that model supports backpropagation."""
        model = LSTMAutoencoder().to(DEVICE)
        x = torch.randn(4, SEQUENCE_LENGTH, len(OBD2_FEATURES), requires_grad=True).to(DEVICE)

        output = model(x)
        loss = torch.mean((output - x) ** 2)
        loss.backward()

        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None


class TestAnomalyDetector:
    """Tests for the AnomalyDetector wrapper class."""

    @pytest.fixture
    def detector(self):
        """Create a fresh AnomalyDetector instance."""
        return AnomalyDetector()

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 500
        data = np.random.randn(n_samples, len(OBD2_FEATURES))
        return data

    def test_detector_initialization(self, detector):
        """Test AnomalyDetector initializes correctly."""
        assert detector.model is not None
        assert detector.scaler is not None
        assert detector.threshold is None  # Not set until training

    def test_prepare_sequences(self, detector, sample_data):
        """Test sequence preparation from raw data."""
        sequences = detector.prepare_sequences(sample_data, sequence_length=30)

        expected_n_sequences = len(sample_data) - 30 + 1
        assert sequences.shape[0] == expected_n_sequences
        assert sequences.shape[1] == 30
        assert sequences.shape[2] == len(OBD2_FEATURES)

    def test_prepare_sequences_edge_cases(self, detector):
        """Test sequence preparation with edge cases."""
        # Data exactly equal to sequence length
        data = np.random.randn(60, len(OBD2_FEATURES))
        sequences = detector.prepare_sequences(data, sequence_length=60)
        assert sequences.shape[0] == 1

        # Data shorter than sequence length should produce empty
        data = np.random.randn(30, len(OBD2_FEATURES))
        sequences = detector.prepare_sequences(data, sequence_length=60)
        assert sequences.shape[0] == 0

    def test_fit_basic(self, detector):
        """Test basic training loop completes."""
        # Generate minimal training data
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)

        # Train for just a few epochs
        history = detector.fit(
            sequences[:100],
            X_val=sequences[100:150],
            epochs=2,
            batch_size=16
        )

        assert 'loss' in history
        assert 'val_loss' in history
        assert len(history['loss']) >= 1
        assert detector.threshold is not None

    def test_threshold_is_set_after_training(self, detector):
        """Test that anomaly threshold is set after training."""
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)

        assert detector.threshold is None

        detector.fit(sequences[:100], epochs=1, batch_size=16)

        assert detector.threshold is not None
        assert isinstance(detector.threshold, float)
        assert detector.threshold > 0

    def test_detect_anomalies(self, detector):
        """Test anomaly detection on new data."""
        # Train first
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
        detector.fit(sequences[:100], epochs=2, batch_size=16)

        # Detect anomalies
        test_sequences = sequences[100:120]
        anomalies, scores = detector.detect_anomalies(test_sequences)

        assert len(anomalies) == len(test_sequences)
        assert len(scores) == len(test_sequences)
        assert anomalies.dtype == bool
        assert all(isinstance(s, (float, np.floating)) for s in scores)

    def test_scaler_is_fitted(self, detector):
        """Test that scaler is properly fitted during training."""
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)

        detector.fit(sequences[:100], epochs=1, batch_size=16)

        assert hasattr(detector.scaler, 'mean_')
        assert hasattr(detector.scaler, 'scale_')
        assert len(detector.scaler.mean_) == len(OBD2_FEATURES)


class TestModelSerialization:
    """Tests for saving and loading models."""

    @pytest.fixture
    def trained_detector(self):
        """Create and train a detector for serialization tests."""
        detector = AnomalyDetector()
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
        detector.fit(sequences[:100], epochs=2, batch_size=16)
        return detector

    def test_save_creates_files(self, trained_detector):
        """Test that save creates all required files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_detector.save(save_path)

            assert (save_path / 'model.pth').exists()
            assert (save_path / 'scaler.joblib').exists()
            assert (save_path / 'config.json').exists()

    def test_load_restores_model(self, trained_detector):
        """Test that load restores a working model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_detector.save(save_path)

            # Load the model
            loaded_detector = AnomalyDetector.load(save_path)

            assert loaded_detector.model is not None
            assert loaded_detector.threshold == trained_detector.threshold
            assert loaded_detector.model.sequence_length == trained_detector.model.sequence_length

    def test_loaded_model_produces_same_output(self, trained_detector):
        """Test that loaded model produces same predictions as original."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_detector.save(save_path)

            loaded_detector = AnomalyDetector.load(save_path)

            # Generate test data
            test_input = np.random.randn(5, SEQUENCE_LENGTH, len(OBD2_FEATURES))

            # Get predictions from both models
            _, original_scores = trained_detector.detect_anomalies(test_input)
            _, loaded_scores = loaded_detector.detect_anomalies(test_input)

            # Scores should be very close (may differ due to floating point)
            np.testing.assert_array_almost_equal(original_scores, loaded_scores, decimal=5)

    def test_config_json_contains_required_fields(self, trained_detector):
        """Test that config.json contains all required fields for C# interop."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'
            trained_detector.save(save_path)

            with open(save_path / 'config.json', 'r') as f:
                config = json.load(f)

            # Required fields for C# OnnxAnomalyService
            assert 'SequenceLength' in config
            assert 'NumFeatures' in config
            assert 'Threshold' in config
            assert 'Features' in config
            assert 'ScalerMean' in config
            assert 'ScalerStd' in config


class TestONNXExport:
    """Tests for ONNX model export."""

    @pytest.fixture
    def trained_detector(self):
        """Create and train a detector for ONNX tests."""
        detector = AnomalyDetector()
        data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
        sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
        detector.fit(sequences[:100], epochs=2, batch_size=16)
        return detector

    def test_onnx_export_creates_file(self, trained_detector):
        """Test that ONNX export creates a valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            export_to_onnx(trained_detector, onnx_path)

            assert onnx_path.exists()
            assert onnx_path.stat().st_size > 0

    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
        reason="onnxruntime not installed"
    )
    def test_onnx_model_runs_inference(self, trained_detector):
        """Test that exported ONNX model can run inference."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            export_to_onnx(trained_detector, onnx_path)

            # Load ONNX model
            session = ort.InferenceSession(str(onnx_path))

            # Create test input
            test_input = np.random.randn(
                1, SEQUENCE_LENGTH, len(OBD2_FEATURES)
            ).astype(np.float32)

            # Run inference
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: test_input})

            assert len(output) == 1
            assert output[0].shape == test_input.shape

    @pytest.mark.skipif(
        not pytest.importorskip("onnxruntime", reason="onnxruntime not installed"),
        reason="onnxruntime not installed"
    )
    def test_onnx_output_matches_pytorch(self, trained_detector):
        """Test that ONNX output matches PyTorch output."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / 'model.onnx'
            export_to_onnx(trained_detector, onnx_path)

            # Create test input
            test_input = np.random.randn(
                4, SEQUENCE_LENGTH, len(OBD2_FEATURES)
            ).astype(np.float32)

            # PyTorch inference
            trained_detector.model.eval()
            with torch.no_grad():
                pytorch_input = torch.FloatTensor(test_input).to(DEVICE)
                pytorch_output = trained_detector.model(pytorch_input).cpu().numpy()

            # ONNX inference
            session = ort.InferenceSession(str(onnx_path))
            input_name = session.get_inputs()[0].name
            onnx_output = session.run(None, {input_name: test_input})[0]

            # Outputs should be very close
            np.testing.assert_array_almost_equal(pytorch_output, onnx_output, decimal=4)


class TestSyntheticDataGeneration:
    """Tests for synthetic data generation."""

    def test_generate_correct_shape(self):
        """Test that generated data has correct shape."""
        n_vehicles = 5
        n_samples = 100

        data, labels = generate_synthetic_data(
            n_vehicles=n_vehicles,
            n_samples_per_vehicle=n_samples,
            anomaly_rate=0.1
        )

        expected_total = n_vehicles * n_samples
        assert data.shape == (expected_total, len(OBD2_FEATURES))
        assert labels.shape == (expected_total,)

    def test_generate_contains_anomalies(self):
        """Test that generated data contains anomalies."""
        data, labels = generate_synthetic_data(
            n_vehicles=10,
            n_samples_per_vehicle=1000,
            anomaly_rate=0.1
        )

        anomaly_rate = labels.mean()
        # Allow some variance but should be close to 10%
        assert 0.05 < anomaly_rate < 0.15

    def test_generate_values_in_realistic_range(self):
        """Test that generated values are within realistic OBD2 ranges."""
        data, _ = generate_synthetic_data(
            n_vehicles=5,
            n_samples_per_vehicle=500
        )

        # RPM: 0-8000
        assert data[:, 0].min() >= 0
        assert data[:, 0].max() <= 8000

        # Speed: 0-250 km/h
        assert data[:, 1].min() >= 0
        assert data[:, 1].max() <= 250

        # Load: 0-100%
        assert data[:, 2].min() >= 0
        assert data[:, 2].max() <= 100

        # Coolant temp: -40 to 150 C
        assert data[:, 3].min() >= -40
        assert data[:, 3].max() <= 150

    def test_generate_deterministic_with_seed(self):
        """Test that data generation is reproducible with seed."""
        np.random.seed(42)
        data1, labels1 = generate_synthetic_data(n_vehicles=3, n_samples_per_vehicle=100)

        np.random.seed(42)
        data2, labels2 = generate_synthetic_data(n_vehicles=3, n_samples_per_vehicle=100)

        np.testing.assert_array_equal(data1, data2)
        np.testing.assert_array_equal(labels1, labels2)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_input_handling(self):
        """Test handling of empty input."""
        detector = AnomalyDetector()

        # Empty array
        empty_data = np.array([]).reshape(0, len(OBD2_FEATURES))
        sequences = detector.prepare_sequences(empty_data, SEQUENCE_LENGTH)

        assert len(sequences) == 0

    def test_single_feature_sequence(self):
        """Test model with single feature."""
        model = LSTMAutoencoder(
            sequence_length=30,
            n_features=1,
            latent_dim=4
        ).to(DEVICE)

        x = torch.randn(4, 30, 1).to(DEVICE)
        output = model(x)

        assert output.shape == x.shape

    def test_very_short_sequence(self):
        """Test model with very short sequence length."""
        model = LSTMAutoencoder(
            sequence_length=5,
            n_features=10,
            latent_dim=4
        ).to(DEVICE)

        x = torch.randn(4, 5, 10).to(DEVICE)
        output = model(x)

        assert output.shape == x.shape

    def test_nan_handling_in_input(self):
        """Test that NaN values don't crash the model."""
        detector = AnomalyDetector()

        # Create data with some NaN values
        data = np.random.randn(100, len(OBD2_FEATURES))
        data[10, 0] = np.nan

        # This should not raise an error (but may produce NaN output)
        sequences = detector.prepare_sequences(data, 30)
        assert len(sequences) > 0
