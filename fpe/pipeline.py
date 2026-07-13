import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple, List
from fpe.model import TemporalFusionTransformer
from fpe.config import settings

class QuantileLoss(nn.Module):
    """
    Quantile Loss function (also known as pinball loss) for estimating confidence bounds.
    Evaluates loss across 10th (p10), 50th (p50), and 90th (p90) percentiles.
    """
    def __init__(self, quantiles: List[float] = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        # preds: dict with keys "p10", "p50", "p90" containing tensors of shape (batch, horizon, targets)
        # target: tensor of shape (batch, horizon, targets)
        loss = 0.0
        q_map = {"p10": 0.1, "p50": 0.5, "p90": 0.9}
        
        for key, q in q_map.items():
            pred = preds[key]
            error = target - pred
            # Pinball loss: max(q * error, (q - 1) * error)
            loss_q = torch.max(q * error, (q - 1) * error)
            loss += loss_q.mean()
            
        return loss / len(q_map)

def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    early_stopping_epochs: int = 15,
    save_dir: str = "data/models"
) -> Tuple[List[float], List[float]]:
    """Runs PyTorch training and validation with early stopping."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = QuantileLoss()
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, settings.MODEL_FILENAME)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float("inf")
    epochs_no_improve = 0
    
    print(f"Training started on device: {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        
        for x_hist, x_future, static_cov, y_target in train_loader:
            x_hist = x_hist.to(device)
            x_future = x_future.to(device)
            static_cov = static_cov.to(device)
            y_target = y_target.to(device)
            
            optimizer.zero_grad()
            preds = model(x_hist, x_future, static_cov)
            loss = criterion(preds, y_target)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item() * x_hist.size(0)
            
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation pass
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_hist, x_future, static_cov, y_target in val_loader:
                x_hist = x_hist.to(device)
                x_future = x_future.to(device)
                static_cov = static_cov.to(device)
                y_target = y_target.to(device)
                
                preds = model(x_hist, x_future, static_cov)
                loss = criterion(preds, y_target)
                epoch_val_loss += loss.item() * x_hist.size(0)
                
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {epoch_train_loss:.6f} - Val Loss: {epoch_val_loss:.6f}")
        
        # Early stopping logic
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
            print(f"  --> Saved new best checkpoint to {model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_epochs:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}")
                break
                
    # Load best model weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        
    return train_losses, val_losses

def calculate_mase(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    y_history: np.ndarray
) -> float:
    """
    Calculates Mean Absolute Scaled Error (MASE).
    y_true: (horizon, D) - actual target dimension states
    y_pred: (horizon, D) - p50 forecast states
    y_history: (lookback, D) - historical input states
    """
    mae_model = np.mean(np.abs(y_true - y_pred))
    
    # One-step naive baseline MAE on lookback history
    diffs = np.abs(np.diff(y_history, axis=0))
    mae_naive = np.mean(diffs)
    
    # Avoid divide by zero
    if mae_naive == 0.0:
        mae_naive = 1e-5
        
    return float(mae_model / mae_naive)

def run_evaluation(model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
    """
    Evaluates the model on test loader and returns evaluation metrics:
    - Quantile Loss (q-Loss)
    - Mean Absolute Scaled Error (MASE)
    - Prediction Drift (Wasserstein Distance)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    criterion = QuantileLoss()
    total_q_loss = 0.0
    
    # For MASE and Wasserstein distance
    all_trues = []
    all_preds_p50 = []
    all_histories = []
    
    with torch.no_grad():
        for x_hist, x_future, static_cov, y_target in test_loader:
            x_hist_dev = x_hist.to(device)
            x_future_dev = x_future.to(device)
            static_cov_dev = static_cov.to(device)
            y_target_dev = y_target.to(device)
            
            preds = model(x_hist_dev, x_future_dev, static_cov_dev)
            loss = criterion(preds, y_target_dev)
            total_q_loss += loss.item() * x_hist.size(0)
            
            # Store numpy representations for MASE and Drift calculations
            all_trues.append(y_target.numpy())
            all_preds_p50.append(preds["p50"].cpu().numpy())
            all_histories.append(x_hist.numpy())
            
    mean_q_loss = total_q_loss / len(test_loader.dataset)
    
    # Concatenate all batches
    all_trues = np.concatenate(all_trues, axis=0)       # (N, horizon, D)
    all_preds_p50 = np.concatenate(all_preds_p50, axis=0) # (N, horizon, D)
    all_histories = np.concatenate(all_histories, axis=0) # (N, lookback, D)
    
    # Compute MASE averaged across all samples
    mase_list = []
    for i in range(len(all_trues)):
        # Slices corresponding to the 10 target dimensions (indices 0 to 9 in x_hist)
        y_hist_dims = all_histories[i][:, :10]
        mase_val = calculate_mase(all_trues[i], all_preds_p50[i], y_hist_dims)
        mase_list.append(mase_val)
    mean_mase = np.mean(mase_list)
    
    # Compute Wasserstein prediction drift
    # Flatten actual vs predicted arrays along batch and horizon dimensions to see overall distribution
    # shape of flat arrays: (N * horizon, D)
    flat_trues = all_trues.reshape(-1, all_trues.shape[-1])
    flat_preds = all_preds_p50.reshape(-1, all_preds_p50.shape[-1])
    
    wasserstein_list = []
    for d in range(flat_trues.shape[-1]):
        wd = wasserstein_distance(flat_trues[:, d], flat_preds[:, d])
        wasserstein_list.append(wd)
    mean_drift = np.mean(wasserstein_list)
    
    return {
        "quantile_loss": float(mean_q_loss),
        "mase": float(mean_mase),
        "prediction_drift": float(mean_drift)
    }
