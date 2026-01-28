import argparse
import logging
import torch
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from hope_ai.config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    PHYSICS_WEIGHT,
    DEVICE
)
from hope_ai.dataset import generate_synthetic_data
from hope_ai.pinn import PINNModel, PhysicsLoss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_pinn():
    parser = argparse.ArgumentParser(description='Train PINN for EGT Virtual Sensor')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=50, # Fewer epochs for quick validation
                        help='Number of training epochs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Data
    # Dataset includes 12 features. EGT is at index 11.
    data, _ = generate_synthetic_data(n_vehicles=50)
    
    # 2. Preprocess
    # Inputs: indices 0-10 (rpm, speed, load, coolant, intake, maf, throttle, fuel_p, stft, ltft, ignition)
    # Target: index 11 (egt)
    X = data[:, :11]
    y = data[:, 11].reshape(-1, 1)
    
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=BATCH_SIZE
    )

    # 3. Model & Training Setup
    model = PINNModel(n_inputs=11, n_outputs=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = PhysicsLoss(lambda_physics=PHYSICS_WEIGHT)
    
    # Enable grad tracking for physics loss
    def custom_forward(inputs):
        return model(inputs)
    
    criterion.forward_with_grad = custom_forward

    # 4. Training Loop
    logger.info(f"Starting PINN training on {DEVICE}...")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y, batch_x)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                outputs = model(batch_x)
                # Note: Physics loss is only active during training as it needs gradients
                loss = torch.nn.functional.mse_loss(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1}/{args.epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f}")

    # 5. Save Model and Scalers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = output_dir / f'egt_pinn_{timestamp}.pth'
    torch.save(model.state_dict(), save_path)
    joblib.dump(scaler_x, output_dir / f'scaler_x_{timestamp}.joblib')
    joblib.dump(scaler_y, output_dir / f'scaler_y_{timestamp}.joblib')
    
    logger.info(f"EGT PINN model saved to {save_path}")

if __name__ == '__main__':
    train_pinn()
