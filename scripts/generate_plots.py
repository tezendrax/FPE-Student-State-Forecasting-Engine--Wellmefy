import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from fpe.dataset import StudentSequenceDataset, DIMENSIONS, HISTORICAL_COVARIATES
from fpe.model import TemporalFusionTransformer
from fpe.config import settings

def main():
    print("Generating evaluation plots...")
    csv_path = "data/student_stress_dataset.csv"
    model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)
    
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print("Error: Model or dataset not found. Run training first.")
        return
        
    # 1. Load validation dataset
    val_dataset = StudentSequenceDataset(csv_path, lookback_days=settings.LOOKBACK_DAYS, horizon_days=settings.FORECAST_HORIZON_DAYS, is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    
    # 2. Load model
    model = TemporalFusionTransformer(
        num_hist_features=17,
        num_future_features=3,
        num_static_features=10,
        hidden_dim=16,
        num_heads=2,
        num_targets=10,
        dropout=0.1
    )
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    
    # Get all validation data in one batch
    x_hist, x_future, static_cov, y_target = next(iter(val_loader))
    
    with torch.no_grad():
        preds = model(x_hist, x_future, static_cov)
        p10 = preds["p10"].numpy()
        p50 = preds["p50"].numpy()
        p90 = preds["p90"].numpy()
        y_true = y_target.numpy()
        
    os.makedirs("data/plots", exist_ok=True)
    
    # Set style
    sns.set_theme(style="darkgrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16
    })
    
    # -------------------------------------------------------------
    # Plot 1: Actual vs Predicted Scatter Plot (for Stress p50)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    
    stress_idx = DIMENSIONS.index("stress")
    y_true_flat = y_true[:, :, stress_idx].flatten()
    p50_flat = np.clip(p50[:, :, stress_idx].flatten(), 0.0, 1.0)
    
    # Scatter plot
    sns.scatterplot(x=y_true_flat, y=p50_flat, color="#10b981", alpha=0.4, edgecolors="none")
    
    # Perfect prediction line
    plt.plot([0, 1], [0, 1], color="#f43f5e", linestyle="--", linewidth=2, label="Perfect Prediction")
    
    plt.title("Actual vs. Predicted - Stress Level")
    plt.xlabel("Ground Truth Stress Index [0.0 - 1.0]")
    plt.ylabel("Model Predicted Stress (p50) [0.0 - 1.0]")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("data/plots/actual_vs_predicted.png", dpi=150)
    plt.close()
    print("Generated actual_vs_predicted.png")
    
    # -------------------------------------------------------------
    # Plot 2: Permutation Feature Importances (Top 10)
    # -------------------------------------------------------------
    print("Calculating permutation feature importances...")
    base_mse = np.mean((y_true - p50)**2)
    importances = {}
    
    for idx, feature in enumerate(HISTORICAL_COVARIATES):
        x_hist_perm = x_hist.clone()
        perm = torch.randperm(x_hist.shape[0])
        x_hist_perm[:, :, idx] = x_hist[perm, :, idx]
        
        with torch.no_grad():
            preds_perm = model(x_hist_perm, x_future, static_cov)
            p50_perm = preds_perm["p50"].numpy()
            
        perm_mse = np.mean((y_true - p50_perm)**2)
        importances[feature] = max(0.0, perm_mse - base_mse)
        
    # Sort and take top 10
    sorted_importances = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
    features_top = [x[0] for x in sorted_importances]
    scores_top = [x[1] for x in sorted_importances]
    
    scores_top = np.array(scores_top)
    if scores_top.max() > 0:
        scores_top = scores_top / scores_top.max()
        
    plt.figure(figsize=(9, 6))
    sns.barplot(x=scores_top, y=features_top, palette="viridis", hue=features_top, legend=False)
    plt.title("Top 10 Feature Importances - TFT Predictor")
    plt.xlabel("Relative Importance Score")
    plt.ylabel("Engineered Feature Name")
    plt.tight_layout()
    plt.savefig("data/plots/feature_importances.png", dpi=150)
    plt.close()
    print("Generated feature_importances.png")
    
    # -------------------------------------------------------------
    # Plot 3: Quantile Forecast Trajectory Shading (Sample Student)
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    
    # Select sample index 0 from validation set
    student_sample_idx = 0
    hist_vals = x_hist[student_sample_idx, :, stress_idx].numpy()
    
    future_true = y_true[student_sample_idx, :, stress_idx]
    future_p10 = np.clip(p10[student_sample_idx, :, stress_idx], 0.0, 1.0)
    future_p50 = np.clip(p50[student_sample_idx, :, stress_idx], 0.0, 1.0)
    future_p90 = np.clip(p90[student_sample_idx, :, stress_idx], 0.0, 1.0)
    
    x_hist_range = np.arange(-13, 1)
    x_future_range = np.arange(1, 8)
    
    # Plot history
    plt.plot(x_hist_range, hist_vals, color="#94a3b8", marker="o", linewidth=2, label="Historical Lookback (14 days)")
    
    # Plot future true
    plt.plot(x_future_range, future_true, color="#10b981", marker="s", linewidth=2, label="Actual Trajectory")
    
    # Plot future predicted p50
    plt.plot(x_future_range, future_p50, color="#3b82f6", marker="^", linestyle="-", linewidth=2.5, label="Predicted Median (p50)")
    
    # Shaded confidence band p10 - p90
    plt.fill_between(x_future_range, future_p10, future_p90, color="#3b82f6", alpha=0.15, label="90% Confidence Band (p10 - p90)")
    plt.plot(x_future_range, future_p10, color="#3b82f6", linestyle="--", alpha=0.6)
    plt.plot(x_future_range, future_p90, color="#3b82f6", linestyle="--", alpha=0.6)
    
    plt.title("Sample Multi-Quantile Forecast - Stress Trajectory")
    plt.xlabel("Timeline (Days relative to forecast epoch)")
    plt.ylabel("Stress Score Value")
    plt.xticks(np.arange(-13, 8))
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig("data/plots/quantile_forecast_sample.png", dpi=150)
    plt.close()
    print("Generated quantile_forecast_sample.png")
    print("All evaluation plots generated successfully!")

if __name__ == "__main__":
    main()
