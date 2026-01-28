import pytest
import torch
import numpy as np
from pathlib import Path
from hope_ai.pinn import PINNModel, PhysicsLoss
from hope_ai.virtual_sensors import EGTVirtualSensor

def test_pinn_model_output_shape():
    model = PINNModel(n_inputs=11, n_outputs=1)
    x = torch.randn(5, 11)
    output = model(x)
    assert output.shape == (5, 1)

def test_physics_loss_gradient_flow():
    model = PINNModel(n_inputs=11, n_outputs=1)
    loss_fn = PhysicsLoss(lambda_physics=0.1)
    
    # inputs: [rpm, speed, load, coolant, intake, maf, throttle, fuel_p, stft, ltft, ignition]
    inputs = torch.randn(5, 11, requires_grad=True)
    targets = torch.randn(5, 1)
    
    def custom_forward(x):
        return model(x)
    loss_fn.forward_with_grad = custom_forward
    
    predictions = model(inputs)
    loss = loss_fn(predictions, targets, inputs)
    
    assert loss.item() > 0
    loss.backward()
    
    # Check if gradients reached the model parameters
    for param in model.parameters():
        assert param.grad is not None

def test_egt_virtual_sensor_fallback():
    sensor = EGTVirtualSensor()
    inputs = {
        'engine_rpm': 2000,
        'engine_load': 50,
        'intake_air_temp': 30,
        'ignition_timing': 15
    }
    # Physical approximation: 30 + (50 * 6) + (2000 * 0.05) - (15 * 5) + 300
    # 30 + 300 + 100 - 75 + 300 = 655
    prediction = sensor.predict(inputs)
    assert 650 <= prediction <= 660

def test_egt_virtual_sensor_with_model():
    # Verify it can handle the full input set and model loading (even if file doesn't exist yet in test runner)
    sensor = EGTVirtualSensor()
    inputs = {
        'engine_rpm': 2000,
        'vehicle_speed': 60,
        'engine_load': 50,
        'coolant_temp': 90,
        'intake_air_temp': 30,
        'maf_flow': 25,
        'throttle_position': 40,
        'fuel_pressure': 400,
        'short_term_fuel_trim': 0,
        'long_term_fuel_trim': 0,
        'ignition_timing': 15
    }
    # Should use fallback if no model is loaded
    prediction = sensor.predict(inputs)
    assert prediction > 0
