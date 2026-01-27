"""
Tests for the inference module.

This test suite covers:
- Model loading (ONNX and PyTorch)
- Preprocessing
- Inference pipeline
- Anomaly analysis
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

# Check if onnxruntime is available
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

from train_anomaly_detector import (
    AnomalyDetector as TrainingDetector,
    generate_synthetic_data,
    export_to_onnx,
    SEQUENCE_LENGTH,
    OBD2_FEATURES,
)


class TestInferenceDetector:
    """Tests for the inference AnomalyDetector class."""

    @pytest.fixture
    def saved_model_path(self):
        """Create a trained model and save it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'

            # Train and save
            detector = TrainingDetector()
            data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
            sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
            detector.fit(sequences[:100], epochs=2, batch_size=16)
            detector.save(save_path)

            # Also export ONNX
            onnx_path = save_path / 'onnx' / 'anomaly_detector.onnx'
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_onnx(detector, onnx_path)

            yield save_path

    def test_load_model_pytorch(self, saved_model_path):
        """Test loading a PyTorch model."""
        from inference import AnomalyDetector

        # Remove ONNX to force PyTorch loading
        onnx_path = saved_model_path / 'onnx' / 'anomaly_detector.onnx'
        if onnx_path.exists():
            onnx_path.unlink()

        detector = AnomalyDetector(str(saved_model_path))

        assert detector.config is not None
        assert not detector.is_onnx

    @pytest.mark.skipif(
        not ONNXRUNTIME_AVAILABLE,
        reason="onnxruntime not installed"
    )
    def test_load_model_onnx(self, saved_model_path):
        """Test loading an ONNX model."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        assert detector.config is not None
        assert detector.is_onnx

    def test_preprocess_scales_data(self, saved_model_path):
        """Test that preprocessing scales data correctly."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        # Create test input
        test_data = np.random.randn(10, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        preprocessed = detector.preprocess(test_data)

        # Should have same shape
        assert preprocessed.shape == test_data.shape

        # Should be scaled (different values)
        assert not np.allclose(preprocessed, test_data)

    def test_predict_returns_reconstructions_and_scores(self, saved_model_path):
        """Test that predict returns correct outputs."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        # Create test input
        test_data = np.random.randn(5, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        reconstructions, scores = detector.predict(test_data)

        assert reconstructions.shape == test_data.shape
        assert len(scores) == len(test_data)
        assert all(s >= 0 for s in scores)  # MSE is non-negative

    def test_detect_anomalies_returns_boolean_array(self, saved_model_path):
        """Test that detect_anomalies returns boolean anomaly flags."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        test_data = np.random.randn(10, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        is_anomaly, scores, reconstructions = detector.detect_anomalies(test_data)

        assert is_anomaly.dtype == bool
        assert len(is_anomaly) == len(test_data)
        assert len(scores) == len(test_data)
        assert reconstructions.shape == test_data.shape

    def test_detect_anomalies_with_custom_threshold(self, saved_model_path):
        """Test anomaly detection with custom threshold."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        test_data = np.random.randn(10, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        # Very low threshold should flag all as anomalies
        is_anomaly_low, _, _ = detector.detect_anomalies(test_data, threshold=0.0001)

        # Very high threshold should flag none
        is_anomaly_high, _, _ = detector.detect_anomalies(test_data, threshold=1000.0)

        assert is_anomaly_low.sum() > is_anomaly_high.sum()

    def test_analyze_anomaly_returns_feature_contributions(self, saved_model_path):
        """Test that analyze_anomaly identifies contributing features."""
        from inference import AnomalyDetector

        detector = AnomalyDetector(str(saved_model_path))

        test_sequence = np.random.randn(SEQUENCE_LENGTH, len(OBD2_FEATURES))

        analysis = detector.analyze_anomaly(test_sequence, OBD2_FEATURES)

        assert 'anomaly_score' in analysis
        assert 'threshold' in analysis
        assert 'is_anomaly' in analysis
        assert 'feature_errors' in analysis
        assert 'top_contributing_features' in analysis

        # Should have correct number of features
        assert len(analysis['feature_errors']) == len(OBD2_FEATURES)

        # Top features should be ranked
        top_features = analysis['top_contributing_features']
        assert len(top_features) <= 5
        for i, feat in enumerate(top_features):
            assert feat['rank'] == i + 1
            assert feat['name'] in OBD2_FEATURES


class TestSequenceCreation:
    """Tests for the create_sequences helper function."""

    def test_create_sequences_basic(self):
        """Test basic sequence creation."""
        from inference import create_sequences

        data = np.arange(100).reshape(100, 1)
        sequences = create_sequences(data, sequence_length=10)

        assert sequences.shape == (91, 10, 1)

        # First sequence should be 0-9
        np.testing.assert_array_equal(sequences[0].flatten(), np.arange(10))

        # Last sequence should be 90-99
        np.testing.assert_array_equal(sequences[-1].flatten(), np.arange(90, 100))

    def test_create_sequences_multi_feature(self):
        """Test sequence creation with multiple features."""
        from inference import create_sequences

        data = np.random.randn(100, 5)
        sequences = create_sequences(data, sequence_length=20)

        assert sequences.shape == (81, 20, 5)

    def test_create_sequences_preserves_values(self):
        """Test that sequence creation preserves data values."""
        from inference import create_sequences

        data = np.random.randn(50, 3)
        sequences = create_sequences(data, sequence_length=10)

        # Check a specific sequence
        np.testing.assert_array_equal(sequences[5], data[5:15])


class TestDataLoading:
    """Tests for data loading utilities."""

    def test_load_test_data_npy(self):
        """Test loading data from .npy file."""
        from inference import load_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'test_data.npy'

            # Create test data
            data = np.random.randn(200, 10)
            np.save(file_path, data)

            # Load
            sequences = load_test_data(file_path, sequence_length=30)

            assert sequences.shape == (171, 30, 10)

    def test_load_test_data_csv(self):
        """Test loading data from .csv file."""
        from inference import load_test_data
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'test_data.csv'

            # Create test data with mixed types
            df = pd.DataFrame({
                'feature1': np.random.randn(200),
                'feature2': np.random.randn(200),
                'feature3': np.random.randn(200),
                'label': ['normal'] * 200,  # Non-numeric column
            })
            df.to_csv(file_path, index=False)

            # Load (should ignore non-numeric column)
            sequences = load_test_data(file_path, sequence_length=30)

            assert sequences.shape == (171, 30, 3)  # 3 numeric columns

    def test_load_test_data_invalid_format(self):
        """Test that invalid format raises error."""
        from inference import load_test_data

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / 'test_data.txt'
            file_path.write_text("invalid data")

            with pytest.raises(ValueError):
                load_test_data(file_path, sequence_length=30)


class TestBatchInference:
    """Tests for batch inference capabilities."""

    @pytest.fixture
    def inference_detector(self, saved_model_path):
        """Create detector for inference tests."""
        from inference import AnomalyDetector
        return AnomalyDetector(str(saved_model_path))

    @pytest.fixture
    def saved_model_path(self):
        """Create a trained model and save it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_model'

            detector = TrainingDetector()
            data, _ = generate_synthetic_data(n_vehicles=2, n_samples_per_vehicle=200)
            sequences = detector.prepare_sequences(data, SEQUENCE_LENGTH)
            detector.fit(sequences[:100], epochs=2, batch_size=16)
            detector.save(save_path)

            yield save_path

    def test_batch_inference_consistency(self, inference_detector):
        """Test that batch inference gives consistent results."""
        test_data = np.random.randn(20, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        # Single batch
        _, scores_batch = inference_detector.predict(test_data)

        # Individual predictions
        scores_individual = []
        for i in range(len(test_data)):
            _, score = inference_detector.predict(test_data[i:i+1])
            scores_individual.append(score[0])

        scores_individual = np.array(scores_individual)

        np.testing.assert_array_almost_equal(scores_batch, scores_individual, decimal=5)

    def test_large_batch_inference(self, inference_detector):
        """Test inference on large batches."""
        test_data = np.random.randn(100, SEQUENCE_LENGTH, len(OBD2_FEATURES))

        # Should not crash on large batches
        reconstructions, scores = inference_detector.predict(test_data)

        assert len(scores) == 100
        assert reconstructions.shape == test_data.shape
