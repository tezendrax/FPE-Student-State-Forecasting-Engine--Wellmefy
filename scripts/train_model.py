import os
import sys
import json
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from fpe.dataset import StudentSequenceDataset
from fpe.model import TemporalFusionTransformer
from fpe.pipeline import run_training_loop
from fpe.config import settings

def main():
    print("Initializing training script...")
    csv_path = "data/student_stress_dataset.csv"
    
    # 1. Generate data if not exists
    if not os.path.exists(csv_path):
        print("Dataset CSV not found, generating synthetic cohort first...")
        from scripts.generate_data import generate_synthetic_cohort
        generate_synthetic_cohort()
        
    # 2. Initialize Datasets
    print("Loading sequence datasets...")
    train_dataset = StudentSequenceDataset(csv_path, lookback_days=settings.LOOKBACK_DAYS, horizon_days=settings.FORECAST_HORIZON_DAYS, is_train=True)
    val_dataset = StudentSequenceDataset(csv_path, lookback_days=settings.LOOKBACK_DAYS, horizon_days=settings.FORECAST_HORIZON_DAYS, is_train=False)
    
    # Save the scaler bounds so they can be re-used in inference
    scaler_params = {
        "scaler_min": train_dataset.scaler_min,
        "scaler_max": train_dataset.scaler_max
    }
    
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    scaler_path = os.path.join(settings.MODEL_DIR, "scaler_params.json")
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=4)
    print(f"Saved dataset scaling bounds to {scaler_path}")
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)} (Batches: {len(train_loader)})")
    print(f"Val samples: {len(val_dataset)} (Batches: {len(val_loader)})")
    
    # 3. Instantiate Model
    # TFT takes 17 historical features, 3 future known features, 10 static covariates, and outputs 10 targets
    model = TemporalFusionTransformer(
        num_hist_features=17,
        num_future_features=3,
        num_static_features=10,
        hidden_dim=16,
        num_heads=2,
        num_targets=10,
        dropout=0.1
    )
    
    # Count model parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TFT Model parameters: {num_params:,}")
    
    # 4. Train Model
    # Epochs = 30, early stopping at 15 epochs, learning rate = 1e-3
    train_losses, val_losses = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,
        lr=1e-3,
        early_stopping_epochs=15,
        save_dir=settings.MODEL_DIR
    )
    
    print("Training process finished successfully!")

if __name__ == "__main__":
    main()
