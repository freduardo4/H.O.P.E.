import torch
import torch.nn as nn
from typing import Optional

class PINNModel(nn.Module):
    """Neural Network for Physics-Informed learning."""
    
    def __init__(self, n_inputs: int, n_outputs: int, hidden_dim: int = 64):
        super(PINNModel, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(n_inputs, hidden_dim),
            nn.Tanh(), # Tanh is preferred for PINNs as it's twice differentiable
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_outputs)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class PhysicsLoss(nn.Module):
    """Custom loss function that penalizes deviations from physical laws."""
    
    def __init__(self, lambda_physics: float = 0.1):
        super(PhysicsLoss, self).__init__()
        self.lambda_physics = lambda_physics
        self.mse = nn.MSELoss()
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        # Data loss (Standard MSE)
        data_loss = self.mse(predictions, targets)
        
        # Physics Loss
        # For EGT, we want to enforce that EGT increases with Load and decreases with Ignition Advance
        # We can use gradients to enforce this monotonic behavior
        
        # inputs: [rpm, speed, load, coolant, intake, maf, throttle, fuel_p, stft, ltft, ignition]
        # index 2: load
        # index 10: ignition
        
        # This is a simplified PINN loss where we penalize non-physical gradients
        # or inconsistent behavior.
        
        physics_loss = torch.tensor(0.0).to(predictions.device)
        
        # Ensure gradients can be calculated
        inputs.requires_grad_(True)
        preds = self.forward_with_grad(inputs)
        
        # Gradient of EGT with respect to Load (should be positive)
        grad_load = torch.autograd.grad(preds.sum(), inputs, create_graph=True)[0][:, 2]
        physics_loss += torch.mean(torch.relu(-grad_load)) # Penalize if negative
        
        # Gradient of EGT with respect to Ignition (should be negative)
        grad_ign = torch.autograd.grad(preds.sum(), inputs, create_graph=True)[0][:, 10]
        physics_loss += torch.mean(torch.relu(grad_ign)) # Penalize if positive
        
        return data_loss + self.lambda_physics * physics_loss

    def forward_with_grad(self, inputs: torch.Tensor) -> torch.Tensor:
        # Helper to avoid issues with modules that might have side effects
        return inputs # Placeholder, needs to be the model's forward
