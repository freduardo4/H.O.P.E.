import unittest
import numpy as np
from hope_ai.dataset import generate_synthetic_data

class TestDataset(unittest.TestCase):
    def test_generate_synthetic_data_shape(self):
        n_vehicles = 2
        n_samples = 100
        data, labels = generate_synthetic_data(n_vehicles=n_vehicles, n_samples_per_vehicle=n_samples)
        
        expected_samples = n_vehicles * n_samples
        self.assertEqual(data.shape, (expected_samples, 12))
        self.assertEqual(labels.shape, (expected_samples,))

    def test_generate_synthetic_data_ranges(self):
        data, _ = generate_synthetic_data(n_vehicles=1, n_samples_per_vehicle=100)
        
        # Check RPM range (0-8000)
        self.assertTrue(np.all(data[:, 0] >= 0))
        self.assertTrue(np.all(data[:, 0] <= 8000))
        
        # Check Load range (0-100)
        self.assertTrue(np.all(data[:, 2] >= 0))
        self.assertTrue(np.all(data[:, 2] <= 100))

    def test_anomaly_labels_consistency(self):
        # High anomaly rate to guarantee some anomalies
        data, labels = generate_synthetic_data(n_vehicles=1, n_samples_per_vehicle=1000, anomaly_rate=0.5)
        
        # If label is 1, it should be an anomaly
        self.assertTrue(np.any(labels == 1))
        self.assertTrue(np.any(labels == 0))

if __name__ == '__main__':
    unittest.main()
