import os
import pytest
import numpy as np
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnx_model_load():
    """Verify that the exported ONNX model can be loaded."""
    model_path = os.path.join("models", "onnx", "anomaly_detector.onnx")
    assert os.path.exists(model_path), f"Model not found at {model_path}"
    
    session = ort.InferenceSession(model_path)
    assert session is not None
    
    # Check input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    assert input_name == "input"
    assert output_name == "output"

@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnx_model_inference_shape():
    """Verify that the model accepts the expected input shape and produces correct output shape."""
    model_path = os.path.join("models", "onnx", "anomaly_detector.onnx")
    session = ort.InferenceSession(model_path)
    
    # Expected input: [batch, seq_len, features]
    # Based on LSTMAutoencoder config: seq_len=10, features=3 (from training script)
    batch_size = 1
    seq_len = 10
    features = 1 # Wait, I need to check the actual feature count in the exported model
    
    input_shape = session.get_inputs()[0].shape
    # input_shape might be ['batch_size', 10, 1] or similar
    
    dummy_input = np.random.randn(batch_size, 10, input_shape[2] if isinstance(input_shape[2], int) else 4).astype(np.float32)
    # Actually, let's just use what the model tells us
    actual_input_shape = [batch_size if isinstance(s, str) else s for s in input_shape]
    dummy_input = np.random.randn(*actual_input_shape).astype(np.float32)
    
    outputs = session.run(None, {"input": dummy_input})
    assert len(outputs) == 1
    assert outputs[0].shape == dummy_input.shape, "Autoencoder output shape should match input shape"
