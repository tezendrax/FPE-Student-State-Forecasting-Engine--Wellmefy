import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fpe.dataset import StudentSequenceDataset, preprocess_and_interpolate, prepare_inference_sequence
from fpe.pipeline import calculate_mase

def test_preprocess_and_interpolate():
    # Create synthetic dataframe with missing days for two students
    data = {
        "student_id": ["std-1", "std-1", "std-2", "std-2"],
        "day": [1, 3, 2, 4],
        "stress": [0.2, 0.4, 0.3, 0.5]
    }
    df = pd.DataFrame(data)
    
    interpolated = preprocess_and_interpolate(df)
    
    # Check that day 2 was interpolated for std-1, and day 3 was interpolated for std-2
    assert len(interpolated) == 6 # std-1 has days 1,2,3; std-2 has days 2,3,4
    
    std1_group = interpolated[interpolated["student_id"] == "std-1"].sort_values("day")
    assert list(std1_group["day"]) == [1, 2, 3]
    # Check linear interpolation (average of 0.2 and 0.4 = 0.3)
    assert np.allclose(std1_group[std1_group["day"] == 2]["stress"].values[0], 0.3)

def test_dataset_sequence_loading():
    csv_path = "data/student_stress_dataset.csv"
    if not os.path.exists(csv_path):
        pytest.skip("Dataset CSV not generated yet, skipping dataset tests.")
        
    dataset = StudentSequenceDataset(csv_path, lookback_days=14, horizon_days=7, is_train=True)
    
    assert len(dataset) > 0
    
    # Inspect shape of a single item
    x_hist, x_future, static_cov, y_target = dataset[0]
    
    assert x_hist.shape == (14, 17) # 14 lookback days, 17 historical features
    assert x_future.shape == (7, 3) # 7 horizon days, 3 future features
    assert static_cov.shape == (10,) # 10 static dimensions
    assert y_target.shape == (7, 10)  # 7 horizon days, 10 target dimensions
    
    # Check that scaled values are indeed normalized within [0, 1] range
    assert torch.all(x_hist >= 0.0)
    assert torch.all(x_hist <= 1.0)

def test_mase_calculation():
    # Simple check on MASE formula
    # Let history be stationary (no change, so naive baseline diff = 0, but we floor it at 1e-5)
    # Let y_true and y_pred match perfectly, MASE should be 0.0
    y_true = np.ones((7, 10)) * 0.5
    y_pred = np.ones((7, 10)) * 0.5
    y_history = np.ones((14, 10)) * 0.5
    
    mase = calculate_mase(y_true, y_pred, y_history)
    assert np.isclose(mase, 0.0)
    
    # Let's verify that poor forecasts give high MASE
    y_pred_bad = np.ones((7, 10)) * 0.9 # wrong forecasts
    y_history_moving = np.tile(np.linspace(0.4, 0.6, 14).reshape(-1, 1), (1, 10)) # moving history
    
    mase_bad = calculate_mase(y_true, y_pred_bad, y_history_moving)
    assert mase_bad > 0.0
    
def test_prepare_inference_sequence():
    # Create mock history dataframe for inference
    lookback = 14
    horizon = 7
    
    data = {
        "day": list(range(1, 20)),
        "stress": [0.3] * 19,
        "anxiety": [0.25] * 19,
        "fatigue": [0.3] * 19,
        "social": [0.7] * 19,
        "academic": [0.6] * 19,
        "burnout": [0.15] * 19,
        "sleep": [0.75] * 19,
        "mood": [0.7] * 19,
        "resilience": [0.65] * 19,
        "focus": [0.7] * 19
    }
    history_df = pd.DataFrame(data)
    
    scaler_min = {col: 0.0 for col in data.keys() if col != "day"}
    scaler_min.update({"sin_day": -1.0, "cos_day": -1.0, "sleep_volatility_7d": 0.0, "stress_volatility_7d": 0.0, "delta_stress_7d": -1.0, "sleep_stress_ratio": 0.0, "academic_pressure": 0.0})
    
    scaler_max = {col: 1.0 for col in data.keys() if col != "day"}
    scaler_max.update({"sin_day": 1.0, "cos_day": 1.0, "sleep_volatility_7d": 1.0, "stress_volatility_7d": 1.0, "delta_stress_7d": 1.0, "sleep_stress_ratio": 10.0, "academic_pressure": 1.0})
    
    x_hist, x_future, static_cov = prepare_inference_sequence(
        history_df, 
        lookback_days=lookback, 
        horizon_days=horizon,
        scaler_min=scaler_min,
        scaler_max=scaler_max
    )
    
    assert x_hist.shape == (1, lookback, 17)
    assert x_future.shape == (1, horizon, 3)
    assert static_cov.shape == (1, 10)
    assert not torch.isnan(x_hist).any()
    assert not torch.isnan(x_future).any()
