import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, LargeBinary, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fpe.config import settings

Base = declarative_base()

# ==========================================
# Local FPE Database Models (fpe.db)
# ==========================================

class FutureForecast(Base):
    __tablename__ = "future_forecasts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String(64), nullable=False, index=True)
    forecast_epoch = Column(Integer, nullable=False)  # UTC timestamp of when forecast was generated
    horizon_days = Column(Integer, nullable=False)    # Number of days forecasted (always 7 here)
    target_dimension = Column(String(32), nullable=False) # e.g. "burnout", "stress"
    day = Column(Integer, nullable=False)             # Forecast horizon day index (1 to 7)
    p10_value = Column(Float, nullable=False)         # 10th percentile bound
    p50_value = Column(Float, nullable=False)         # 50th percentile prediction (median)
    p90_value = Column(Float, nullable=False)         # 90th percentile bound
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ==========================================
# External SDT Database Models (sdt.db)
# ==========================================

SdtBase = declarative_base()

class SdtDigitalTwinState(SdtBase):
    __tablename__ = "digital_twin_states"
    
    student_id = Column(String(64), primary_key=True)
    s_stress = Column(Float, nullable=False)
    s_anxiety = Column(Float, nullable=False)
    s_fatigue = Column(Float, nullable=False)
    s_social = Column(Float, nullable=False)
    s_academic = Column(Float, nullable=False)
    s_burnout = Column(Float, nullable=False)
    s_sleep = Column(Float, nullable=False)
    s_mood = Column(Float, nullable=False)
    s_resilience = Column(Float, nullable=False)
    s_focus = Column(Float, nullable=False)
    last_update_epoch = Column(Integer, nullable=False)
    updated_at = Column(DateTime, nullable=False)


class SdtTwinStateHistory(SdtBase):
    __tablename__ = "twin_state_history"
    
    id = Column(Integer, primary_key=True)
    student_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    encrypted_payload = Column(Text, nullable=False)
    key_id = Column(String(64), nullable=False)
    trigger_source = Column(String(128), nullable=False)


class SdtEncryptionKey(SdtBase):
    __tablename__ = "encryption_keys"
    
    id = Column(String(64), primary_key=True)
    key_bytes = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, nullable=False)
    active = Column(Boolean, nullable=False)


# ==========================================
# Database Connection Setup
# ==========================================

# 1. Local Database (fpe.db)
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 2. SDT Database (sdt.db)
sdt_engine = create_engine(
    settings.SDT_DATABASE_URL, 
    connect_args={"check_same_thread": False} if "sqlite" in settings.SDT_DATABASE_URL else {}
)
SdtSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sdt_engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_sdt_db():
    db = SdtSessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    # Ensure directories exist
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(settings.DATABASE_URL.replace("sqlite:///", "")), exist_ok=True)
    Base.metadata.create_all(bind=engine)
