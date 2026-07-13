import os
import json
import time
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from cryptography.fernet import Fernet
from sqlalchemy.orm import Session
import redis

from fpe.config import settings
from fpe.db import FutureForecast, SdtTwinStateHistory, SdtEncryptionKey
from fpe.model import TemporalFusionTransformer, LinearBaselineFallback
from fpe.dataset import prepare_inference_sequence, HISTORICAL_COVARIATES, DIMENSIONS

logger = logging.getLogger("fpe_inference")
logging.basicConfig(level=logging.INFO)

# ==========================================
# Caching Layer (Redis with local fallback)
# ==========================================

class ForecastCache:
    def __init__(self, redis_url: str):
        self.use_redis = False
        try:
            self.r = redis.from_url(redis_url, socket_timeout=1.0)
            # Test connection
            self.r.ping()
            self.use_redis = True
            logger.info("Connected to Redis cache successfully.")
        except Exception as e:
            logger.warning(f"Redis connection failed ({e}). Falling back to local in-memory caching.")
            self.local_cache = {}

    def get(self, student_id: str) -> Dict[str, Any]:
        if self.use_redis:
            try:
                data = self.r.get(f"fpe_forecast:{student_id}")
                if data:
                    return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        else:
            return self.local_cache.get(student_id)
        return None

    def set(self, student_id: str, data: Dict[str, Any], expire_seconds: int = 86400):
        if self.use_redis:
            try:
                self.r.setex(f"fpe_forecast:{student_id}", expire_seconds, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        else:
            self.local_cache[student_id] = data

cache = ForecastCache(settings.REDIS_URL)


# ==========================================
# SDT History Decryption & Resampling
# ==========================================

def decrypt_history_records(db_sdt: Session, records: List[SdtTwinStateHistory]) -> List[Dict[str, Any]]:
    """Decrypts historical telemetry payloads using keys in the database."""
    decrypted_records = []
    key_cache = {}
    
    for rec in records:
        key_id = rec.key_id
        if key_id not in key_cache:
            key_rec = db_sdt.query(SdtEncryptionKey).filter(SdtEncryptionKey.id == key_id).first()
            if key_rec:
                key_cache[key_id] = Fernet(key_rec.key_bytes)
            else:
                continue
                
        f = key_cache[key_id]
        try:
            decrypted_bytes = f.decrypt(rec.encrypted_payload.encode('utf-8'))
            state_dict = pd.read_json(decrypted_bytes.decode('utf-8'), typ='series').to_dict()
            state_dict['timestamp'] = rec.timestamp
            decrypted_records.append(state_dict)
        except Exception as e:
            logger.warning(f"Decryption failed for record {rec.id}: {e}")
            continue
            
    return decrypted_records

def preprocess_and_impute_history(
    decrypted_records: List[Dict[str, Any]], 
    num_days: int = 14, 
    baseline_state: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Imputes and resamples student twin state history.
    1. Extracts state vectors and timestamps.
    2. Takes the last observation of each calendar day.
    3. Fills gaps using linear interpolation.
    4. Backfills with baseline values if history is short.
    """
    if not baseline_state:
        baseline_state = {
            "stress": 0.3, "anxiety": 0.25, "fatigue": 0.3, "social": 0.7,
            "academic": 0.6, "burnout": 0.15, "sleep": 0.75, "mood": 0.7,
            "resilience": 0.65, "focus": 0.7
        }
        
    if not decrypted_records:
        dates = [datetime.utcnow().date() - timedelta(days=i) for i in range(num_days)]
        dates.reverse()
        df = pd.DataFrame([baseline_state] * num_days)
        df['date'] = dates
        df['day'] = range(1, num_days + 1)
        return df.set_index('date')

    df = pd.DataFrame(decrypted_records)
    df['date'] = df['timestamp'].dt.date
    
    # Sort and group by date, taking the latest state vector of the day
    df = df.sort_values('timestamp').groupby('date').last()
    
    # Ensure all dimensions exist
    for dim in DIMENSIONS:
        if dim not in df.columns:
            df[dim] = baseline_state[dim]
            
    df = df[DIMENSIONS]
    
    # Reindex to a complete daily range ending at the latest date
    latest_date = max(df.index)
    start_date = latest_date - timedelta(days=num_days - 1)
    full_date_range = pd.date_range(start=start_date, end=latest_date).date
    
    df = df.reindex(full_date_range)
    
    # Impute missing values using linear interpolation
    df = df.interpolate(method="linear")
    df = df.ffill().bfill()
    
    # Fill remaining NaNs with baseline
    for col in DIMENSIONS:
        df[col] = df[col].fillna(baseline_state[col])
        
    # Map index date to a numeric day index starting at 1
    df['day'] = range(1, num_days + 1)
    return df


# ==========================================
# Real-Time Inference Runner
# ==========================================

class ForecastRunner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler_min = None
        self.scaler_max = None
        
        # Load scaler parameters
        scaler_path = os.path.join(settings.MODEL_DIR, "scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                params = json.load(f)
                self.scaler_min = params["scaler_min"]
                self.scaler_max = params["scaler_max"]
                
        # Load TFT model
        model_path = os.path.join(settings.MODEL_DIR, settings.MODEL_FILENAME)
        if os.path.exists(model_path):
            try:
                self.model = TemporalFusionTransformer(
                    num_hist_features=17,
                    num_future_features=3,
                    num_static_features=10,
                    hidden_dim=16,
                    num_heads=2,
                    num_targets=10,
                    dropout=0.1
                )
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.to(self.device).eval()
                logger.info("Forecasting TFT model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load forecasting TFT model: {e}")
                self.model = None
        else:
            logger.warning(f"No forecasting model checkpoint found at {model_path}. Fallback models will be used.")

    def run_forecast(self, student_id: str, db_sdt: Session, db_fpe: Session) -> Dict[str, Any]:
        """
        Executes forecasting for a student.
        1. Fetch state history from sdt.db.
        2. Decrypt & resample history to 14 days.
        3. Preprocess and format for TFT forward pass.
        4. If TFT fails or diverges, fall back to linear baseline.
        5. Save predictions to fpe.db and update cache.
        """
        t0 = time.time()
        
        # Fetch encrypted state history from SDT
        records = db_sdt.query(SdtTwinStateHistory).filter(SdtTwinStateHistory.student_id == student_id).all()
        decrypted = decrypt_history_records(db_sdt, records)
        
        # Resample to 14-day lookback
        history_df = preprocess_and_impute_history(decrypted, num_days=settings.LOOKBACK_DAYS)
        
        # Prepare input tensors
        x_hist, x_future, static_cov = prepare_inference_sequence(
            history_df, 
            lookback_days=settings.LOOKBACK_DAYS,
            horizon_days=settings.FORECAST_HORIZON_DAYS,
            scaler_min=self.scaler_min,
            scaler_max=self.scaler_max
        )
        
        # Initialize output dicts
        forecasts = {} # keys: p10, p50, p90, each is (horizon, D)
        use_fallback = False
        anomaly_warning = False
        
        # Run TFT Model inference
        if self.model is not None:
            try:
                with torch.no_grad():
                    x_hist_dev = x_hist.to(self.device)
                    x_future_dev = x_future.to(self.device)
                    static_cov_dev = static_cov.to(self.device)
                    
                    preds = self.model(x_hist_dev, x_future_dev, static_cov_dev)
                    
                    p10_np = preds["p10"].cpu().squeeze(0).numpy()
                    p50_np = preds["p50"].cpu().squeeze(0).numpy()
                    p90_np = preds["p90"].cpu().squeeze(0).numpy()
                    
                    # Verify boundary limits to check for severe model divergence (values exceeding [-0.5, 1.5] range)
                    if (np.any(p50_np > 1.5) or np.any(p50_np < -0.5) or np.isnan(p50_np).any()):
                        logger.warning(f"TFT forecasts for {student_id} diverged severely. Falling back to linear baseline model.")
                        use_fallback = True
                        anomaly_warning = True
                    else:
                        forecasts["p10"] = np.clip(p10_np, 0.0, 1.0)
                        forecasts["p50"] = np.clip(p50_np, 0.0, 1.0)
                        forecasts["p90"] = np.clip(p90_np, 0.0, 1.0)
            except Exception as e:
                logger.error(f"Error during TFT inference: {e}. Falling back to linear baseline model.")
                use_fallback = True
                anomaly_warning = True
        else:
            use_fallback = True
            
        # Linear Regression Fallback Execution
        if use_fallback:
            history_matrix = history_df[DIMENSIONS].values
            fallback_model = LinearBaselineFallback(
                lookback_days=settings.LOOKBACK_DAYS, 
                horizon_days=settings.FORECAST_HORIZON_DAYS
            )
            fallback_forecast = fallback_model.forecast(history_matrix)
            forecasts = fallback_forecast
            
        # Write forecasts to database fpe.db
        forecast_epoch = int(datetime.utcnow().timestamp())
        
        # Delete old forecasts for this student to keep database compact
        db_fpe.query(FutureForecast).filter(FutureForecast.student_id == student_id).delete()
        
        for d_idx, dim in enumerate(DIMENSIONS):
            for day in range(1, settings.FORECAST_HORIZON_DAYS + 1):
                p10_val = float(forecasts["p10"][day - 1, d_idx])
                p50_val = float(forecasts["p50"][day - 1, d_idx])
                p90_val = float(forecasts["p90"][day - 1, d_idx])
                
                db_record = FutureForecast(
                    student_id=student_id,
                    forecast_epoch=forecast_epoch,
                    horizon_days=settings.FORECAST_HORIZON_DAYS,
                    target_dimension=dim,
                    day=day,
                    p10_value=p10_val,
                    p50_value=p50_val,
                    p90_value=p90_val
                )
                db_fpe.add(db_record)
                
        db_fpe.commit()
        
        # Format REST response dictionary
        forecast_list = []
        for day in range(1, settings.FORECAST_HORIZON_DAYS + 1):
            day_rec = {"day": day}
            for d_idx, dim in enumerate(DIMENSIONS):
                day_rec[f"{dim}_p10"] = float(forecasts["p10"][day - 1, d_idx])
                day_rec[f"{dim}_p50"] = float(forecasts["p50"][day - 1, d_idx])
                day_rec[f"{dim}_p90"] = float(forecasts["p90"][day - 1, d_idx])
            forecast_list.append(day_rec)
            
        result = {
            "student_id": student_id,
            "horizon_days": settings.FORECAST_HORIZON_DAYS,
            "forecast_epoch": forecast_epoch,
            "fallback_used": use_fallback,
            "anomaly_warning": anomaly_warning,
            "latency_ms": float((time.time() - t0) * 1000.0),
            "forecast": forecast_list
        }
        
        # Store in cache
        cache.set(student_id, result)
        
        return result

forecast_runner = ForecastRunner()
