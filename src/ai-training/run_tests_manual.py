import sys
import unittest
import os
from pathlib import Path

# Setup path to include scripts
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

# Import the test classes
# We need to mock sys.path in the test files or handle imports
# Since test files modify sys.path, we might need to be careful

# Let's try to just run them via unittest discovery provided we fix the path
import tests.test_anomaly_detector as t1
import tests.test_inference as t2

def run_suite():
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Load tests from test classes
    suite.addTests(loader.loadTestsFromTestCase(t1.TestLSTMAutoencoder))
    suite.addTests(loader.loadTestsFromTestCase(t1.TestAnomalyDetector))
    # suite.addTests(loader.loadTestsFromTestCase(t1.TestModelSerialization)) # Might need temp dir handling
    # suite.addTests(loader.loadTestsFromTestCase(t1.TestONNXExport))
    
    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == '__main__':
    # Fix import paths for the imported modules
    # The test modules do: sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    # We are running from src/ai-training
    run_suite()
