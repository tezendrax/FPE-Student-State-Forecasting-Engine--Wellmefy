import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Tuple

DIMENSIONS = ["stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus"]

HISTORICAL_COVARIATES = [
    "stress", "anxiety", "fatigue", "social", "academic", "burnout", "sleep", "mood", "resilience", "focus",
    "academic_pressure", "sin_day", "cos_day", "sleep_volatility_7d", "stress_volatility_7d", "delta_stress_7d", "sleep_stress_ratio"
]

FUTURE_COVARIATES = [
    "academic_pressure", "sin_day", "cos_day"
]

STATIC_COVARIATES = [
    "mean_stress", "mean_anxiety", "mean_fatigue", "mean_social", "mean_academic",
    "mean_burnout", "mean_sleep", "mean_mood", "mean_resilience", "mean_focus"
]

def preprocess_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks daily frequency for each student, sorts by day, and fills missing days
    using linear interpolation.
    """
    df = df.copy()
    processed_groups = []
    
    for sid, group in df.groupby("student_id"):
        group = group.sort_values("day")
        
        # Ensure complete day sequence
        min_day = group["day"].min()
        max_day = group["day"].max()
        full_days = pd.Series(range(min_day, max_day + 1), name="day")
        
        group = pd.merge(full_days, group, on="day", how="left")
        group["student_id"] = sid
        
        # Fill missing values using linear interpolation
        numeric_cols = group.select_dtypes(include=[np.number]).columns
        group[numeric_cols] = group[numeric_cols].interpolate(method="linear")
        
        # Forward/Backward fill for any edge NaNs (e.g. if start/end values are missing)
        group = group.ffill().bfill()
        
        processed_groups.append(group)
        
    return pd.concat(processed_groups, ignore_index=True)

class StudentSequenceDataset(Dataset):
    def __init__(self, csv_path: str, lookback_days: int = 14, horizon_days: int = 7, is_train: bool = True, train_split: float = 0.8):
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.seq_len = lookback_days + horizon_days
        
        # Load and preprocess
        df = pd.read_csv(csv_path)
        df = preprocess_and_interpolate(df)
        
        # Split students to prevent data leakage across lookback windows
        students = sorted(df["student_id"].unique())
        split_idx = int(len(students) * train_split)
        
        if is_train:
            selected_students = students[:split_idx]
        else:
            selected_students = students[split_idx:]
            
        self.df = df[df["student_id"].isin(selected_students)].copy()
        
        # Normalize numeric columns to [0.0, 1.0] using min-max scaling
        self.scaler_min = {}
        self.scaler_max = {}
        for col in HISTORICAL_COVARIATES:
            # Avoid divide-by-zero
            val_min = self.df[col].min()
            val_max = self.df[col].max()
            if val_max == val_min:
                val_max += 1e-5
            self.scaler_min[col] = val_min
            self.scaler_max[col] = val_max
            self.df[col] = (self.df[col] - val_min) / (val_max - val_min)
            
        # Extract sequences
        self.samples = []
        for sid, group in self.df.groupby("student_id"):
            group = group.sort_values("day")
            n_days = len(group)
            
            # Static student attributes (mean baseline states)
            static_vals = [group[dim].mean() for dim in DIMENSIONS]
            
            if n_days < self.seq_len:
                continue
                
            for start_t in range(n_days - self.seq_len + 1):
                hist_chunk = group.iloc[start_t : start_t + self.lookback_days]
                future_chunk = group.iloc[start_t + self.lookback_days : start_t + self.seq_len]
                
                x_hist = hist_chunk[HISTORICAL_COVARIATES].values.astype(np.float32)
                x_future = future_chunk[FUTURE_COVARIATES].values.astype(np.float32)
                y_target = future_chunk[DIMENSIONS].values.astype(np.float32)
                
                self.samples.append({
                    "student_id": sid,
                    "x_hist": torch.tensor(x_hist),
                    "x_future": torch.tensor(x_future),
                    "static_cov": torch.tensor(static_vals, dtype=torch.float32),
                    "y_target": torch.tensor(y_target)
                })
                
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample["x_hist"], sample["x_future"], sample["static_cov"], sample["y_target"]

def prepare_inference_sequence(
    history_df: pd.DataFrame, 
    lookback_days: int = 14, 
    horizon_days: int = 7,
    scaler_min: Dict[str, float] = None,
    scaler_max: Dict[str, float] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Processes a single student's recent DataFrame history for real-time inference.
    Returns (x_hist, x_future, static_cov).
    """
    # 1. Fill gaps and interpolate
    history_df = history_df.sort_values("day").copy()
    
    # 2. Add rolling features
    midterm_day = 45
    final_day = 88
    dist_to_exam = np.minimum(
        np.abs(history_df["day"] - midterm_day),
        np.abs(history_df["day"] - final_day)
    )
    history_df["academic_pressure"] = np.exp(-dist_to_exam / 7.0)
    
    history_df["sin_day"] = np.sin(2 * np.pi * (history_df["day"] % 7) / 7.0)
    history_df["cos_day"] = np.cos(2 * np.pi * (history_df["day"] % 7) / 7.0)
    history_df["sleep_volatility_7d"] = history_df["sleep"].rolling(window=7, min_periods=1).std().fillna(0.0)
    history_df["stress_volatility_7d"] = history_df["stress"].rolling(window=7, min_periods=1).std().fillna(0.0)
    history_df["delta_stress_7d"] = history_df["stress"].diff(periods=7).fillna(0.0)
    history_df["sleep_stress_ratio"] = history_df["sleep"] / (history_df["stress"] + 1e-5)
    
    # 3. Handle interpolation/imputation
    for col in HISTORICAL_COVARIATES:
        if col in history_df.columns:
            history_df[col] = history_df[col].interpolate(method="linear").ffill().bfill()
            
    # Take last 14 days for history
    hist_chunk = history_df.iloc[-lookback_days:]
    
    # Static covariates (student average state baseline)
    static_vals = [hist_chunk[dim].mean() for dim in DIMENSIONS]
    
    # Generate future covariates for upcoming 7 days (forecast exam calendar and days of week)
    last_day = hist_chunk["day"].iloc[-1]
    future_days = np.arange(last_day + 1, last_day + 1 + horizon_days)
    
    # Assume exam pressures decay similarly
    midterm_day = 45
    final_day = 88
    
    future_academic_pressure = []
    future_sin_day = []
    future_cos_day = []
    
    for day in future_days:
        dist_to_exam = min(abs(day - midterm_day), abs(day - final_day))
        academic_pressure = np.exp(-dist_to_exam / 7.0)
        day_of_week = day % 7
        
        future_academic_pressure.append(academic_pressure)
        future_sin_day.append(np.sin(2 * np.pi * day_of_week / 7.0))
        future_cos_day.append(np.cos(2 * np.pi * day_of_week / 7.0))
        
    x_future_data = np.stack([future_academic_pressure, future_sin_day, future_cos_day], axis=1).astype(np.float32)
    
    # Normalize features using saved scales if provided
    x_hist_data = hist_chunk[HISTORICAL_COVARIATES].values.copy().astype(np.float32)
    
    if scaler_min and scaler_max:
        for idx, col in enumerate(HISTORICAL_COVARIATES):
            min_val = scaler_min.get(col, 0.0)
            max_val = scaler_max.get(col, 1.0)
            diff = max_val - min_val if max_val != min_val else 1e-5
            x_hist_data[:, idx] = (x_hist_data[:, idx] - min_val) / diff
            
    x_hist_tensor = torch.tensor(x_hist_data).unsqueeze(0)  # Shape: (1, Lookback, Num_Hist_Features)
    x_future_tensor = torch.tensor(x_future_data).unsqueeze(0) # Shape: (1, Horizon, Num_Future_Features)
    static_tensor = torch.tensor(static_vals, dtype=torch.float32).unsqueeze(0) # Shape: (1, Num_Static_Features)
    
    return x_hist_tensor, x_future_tensor, static_tensor
