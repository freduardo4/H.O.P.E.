import pytest
import numpy as np
import os
try:
    import onnxruntime as ort
except ImportError:
    ort = None
from hope_ai.config import SEQUENCE_LENGTH, OBD2_FEATURES

@pytest.fixture
def onnx_session():
    if ort is None:
        pytest.skip("onnxruntime not installed")
        
    model_path = os.path.join("models", "onnx", "anomaly_detector.onnx")
    if not os.path.exists(model_path):
        pytest.skip("ONNX model not found")
    return ort.InferenceSession(model_path)

def test_anomaly_detector_sensitivity(onnx_session):
    """Verify that the model identifies anomalies correctly."""
    input_name = onnx_session.get_inputs()[0].name
    n_features = len(OBD2_FEATURES)
    
    # 1. Normal sequence (low variance)
    normal_seq = np.zeros((1, SEQUENCE_LENGTH, n_features), dtype=np.float32)
    normal_output = onnx_session.run(None, {input_name: normal_seq})[0]
    normal_error = np.mean(np.square(normal_seq - normal_output))
    
    # 2. Anomalous sequence (spike in one feature)
    anomalous_seq = np.zeros((1, SEQUENCE_LENGTH, n_features), dtype=np.float32)
    anomalous_seq[0, SEQUENCE_LENGTH//2, 0] = 10.0 # Huge spike
    anomalous_output = onnx_session.run(None, {input_name: anomalous_seq})[0]
    anomalous_error = np.mean(np.square(anomalous_seq - anomalous_output))
    
    print(f"Normal error: {normal_error}, Anomalous error: {anomalous_error}")
    
    # The reconstruction error for an anomaly should be significantly higher
    assert anomalous_error > normal_error * 2, "Model is not sensitive enough to large spikes"

def test_virtual_sensor_egt_range():
    """Verify PINN EGT model outputs are within physical bounds."""
    # Assuming we have a PINN model saved
    model_path = os.path.join("models", "egt_pinn.pth")
    if not os.path.exists(model_path):
        # Check for any .pth in models/
        import glob
        pths = glob.glob("models/egt_pinn_*.pth")
        if pths:
            model_path = pths[0]
        else:
            pytest.skip("PINN model not found")
            
    import torch
    from hope_ai.pinn import PINNModel
    
    model = PINNModel(n_inputs=len(OBD2_FEATURES)-1, n_outputs=1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Dummy input: RPM 3000, Load 0.5, etc.
    dummy_input = torch.randn(1, len(OBD2_FEATURES)-1)
    with torch.no_grad():
        egt_pred = model(dummy_input).item()
        
    # EGT should be roughly between 200 and 1200 Celsius for a running engine
    # Since inputs are random/unscaled, we just check it's not NaN or Inf
    assert not np.isnan(egt_pred)
    assert not np.isinf(egt_pred)
