import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Future Prediction Engine (FPE)"
    # Base database for forecasts
    DATABASE_URL: str = "sqlite:///fpe.db"
    # Reference database to load student digital twin states
    SDT_DATABASE_URL: str = "sqlite:///../Digital Twin/sdt.db"
    
    # Server configuration
    PORT: int = 8003
    HOST: str = "0.0.0.0"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Model configuration
    MODEL_DIR: str = "data/models"
    MODEL_FILENAME: str = "tft_model_7d.pt"
    
    # Lookback & Forecasting Configurations
    LOOKBACK_DAYS: int = 14
    FORECAST_HORIZON_DAYS: int = 7
    
    # Fallback/Divergence threshold (predictions must stay within 0.0 - 1.0)
    MAX_STATE_VAL: float = 1.0
    MIN_STATE_VAL: float = 0.0
    
    class Config:
        env_file = ".env"

settings = Settings()

# Resolve absolute paths for SQLite to avoid relative directory issues
def get_absolute_db_url(url: str) -> str:
    if url.startswith("sqlite:///"):
        relative_path = url.replace("sqlite:///", "")
        # If relative, make it absolute from this file's root
        if not os.path.isabs(relative_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_path = os.path.normpath(os.path.join(base_dir, relative_path))
            return f"sqlite:///{abs_path}"
    return url

settings.DATABASE_URL = get_absolute_db_url(settings.DATABASE_URL)
settings.SDT_DATABASE_URL = get_absolute_db_url(settings.SDT_DATABASE_URL)
