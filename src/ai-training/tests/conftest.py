"""
Pytest configuration and shared fixtures for HOPE AI training tests.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Check if onnxruntime is available
try:
    import onnxruntime
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def sample_obd2_data():
    """Generate sample OBD2 data for testing."""
    np.random.seed(42)

    # 1 hour of data at 1 Hz
    n_samples = 3600
    n_features = 10

    # Generate realistic-ish OBD2 data
    data = np.zeros((n_samples, n_features))

    # RPM: idle around 800, cruising around 2500
    data[:, 0] = 800 + np.cumsum(np.random.randn(n_samples) * 50)
    data[:, 0] = np.clip(data[:, 0], 0, 6000)

    # Speed: 0-120 km/h
    data[:, 1] = np.clip(np.cumsum(np.random.randn(n_samples) * 2), 0, 120)

    # Engine load: 0-100%
    data[:, 2] = 20 + np.random.randn(n_samples) * 10
    data[:, 2] = np.clip(data[:, 2], 0, 100)

    # Coolant temp: warm up to 90C
    data[:, 3] = 90 + np.random.randn(n_samples) * 3

    # Intake air temp
    data[:, 4] = 30 + np.random.randn(n_samples) * 5

    # MAF flow
    data[:, 5] = data[:, 2] * 0.5 + np.random.randn(n_samples) * 2

    # Throttle position
    data[:, 6] = data[:, 2] * 0.8 + np.random.randn(n_samples) * 5

    # Fuel pressure
    data[:, 7] = 350 + np.random.randn(n_samples) * 10

    # Short-term fuel trim
    data[:, 8] = np.random.randn(n_samples) * 3

    # Long-term fuel trim
    data[:, 9] = np.random.randn(n_samples) * 2

    return data


@pytest.fixture
def small_trained_model(sample_obd2_data):
    """Create a small trained model for quick tests."""
    from train_anomaly_detector import AnomalyDetector, SEQUENCE_LENGTH

    detector = AnomalyDetector()
    sequences = detector.prepare_sequences(sample_obd2_data[:500], SEQUENCE_LENGTH)

    # Train for minimal epochs
    detector.fit(
        sequences[:50],
        X_val=sequences[50:70],
        epochs=2,
        batch_size=16
    )

    return detector


@pytest.fixture
def model_save_path(test_data_dir):
    """Provide a path for saving test models."""
    return test_data_dir / 'test_model'


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests."""
    for item in items:
        # Mark training tests as slow
        if 'train' in item.name.lower() or 'fit' in item.name.lower():
            item.add_marker(pytest.mark.slow)
