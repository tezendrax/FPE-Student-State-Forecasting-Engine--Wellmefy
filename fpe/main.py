import asyncio
import logging
from datetime import datetime
import os
from fastapi import FastAPI, Depends, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from fpe.config import settings
from fpe.db import init_db, get_db, get_sdt_db, SdtDigitalTwinState, FutureForecast
from fpe.inference import cache, forecast_runner

logger = logging.getLogger("fpe_main")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Forecasts 7-day student wellness trajectories and confidence intervals.",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Security Verification (Access Tokens)
# ==========================================

def verify_access_token(x_access_token: str = Header(default="wellmate-secure-token")):
    # A simple verification token check consistent with security protocols
    if x_access_token != "wellmate-secure-token":
        raise HTTPException(status_code=403, detail="Invalid secure access token.")
    return x_access_token


# ==========================================
# API Endpoints
# ==========================================

@app.get("/api/v1/predictions/forecast")
def get_student_forecast(
    student_id: str = Query(..., description="The unique student identifier, e.g. std-9874"),
    force_refresh: bool = Query(False, description="Bypass cache and force model inference"),
    db_fpe: Session = Depends(get_db),
    db_sdt: Session = Depends(get_sdt_db),
    token: str = Depends(verify_access_token)
):
    """
    Returns the 7-day forecasted wellness trajectory and 10%/90% quantiles for a student.
    Uses cached forecasts from Redis/memory by default, running inference if cache is missed
    or force_refresh is requested.
    """
    # 1. Try fetching from cache
    if not force_refresh:
        cached_data = cache.get(student_id)
        if cached_data:
            logger.info(f"Cache hit for student {student_id}")
            return cached_data
            
    # 2. Check if student digital twin exists in database
    student_twin = db_sdt.query(SdtDigitalTwinState).filter(SdtDigitalTwinState.student_id == student_id).first()
    if not student_twin:
        raise HTTPException(
            status_code=404, 
            detail=f"Student ID '{student_id}' not found in the Student Digital Twin registry."
        )
        
    # 3. Execute forecast pipeline
    try:
        forecast_result = forecast_runner.run_forecast(student_id, db_sdt, db_fpe)
        return forecast_result
    except Exception as e:
        logger.error(f"Inference pipeline execution error for {student_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate trajectory forecast: {str(e)}"
        )


@app.get("/api/v1/predictions/health")
def get_health_status(db_fpe: Session = Depends(get_db)):
    """Returns database health metrics and model loading state."""
    model_loaded = (forecast_runner.model is not None)
    try:
        forecast_count = db_fpe.query(FutureForecast).count()
        db_healthy = True
    except Exception:
        forecast_count = 0
        db_healthy = False
        
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database_connected": db_healthy,
        "forecasting_model_loaded": model_loaded,
        "total_cached_forecasts": forecast_count,
        "cache_type": "Redis" if cache.use_redis else "LocalMemory",
        "current_utc_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    }


# ==========================================
# Nightly Batch Scheduler Loop
# ==========================================

async def nightly_batch_scheduler():
    """Runs asynchronously to perform batch forecasting updates for all students every night at 2:00 AM UTC."""
    logger.info("Nightly batch forecast scheduler initiated.")
    while True:
        try:
            now = datetime.utcnow()
            # Check if it is 2:00 AM UTC
            if now.hour == 2 and now.minute == 0:
                logger.info("Nightly scheduler triggered at 2:00 AM UTC. Initiating batch updates...")
                
                # Fetch new session links
                from fpe.db import SessionLocal, SdtSessionLocal
                db_sdt = SdtSessionLocal()
                db_fpe = SessionLocal()
                
                try:
                    students = db_sdt.query(SdtDigitalTwinState.student_id).all()
                    student_ids = [s.student_id for s in students]
                    
                    logger.info(f"Running nightly batch forecasts for {len(student_ids)} students...")
                    for idx, sid in enumerate(student_ids):
                        try:
                            forecast_runner.run_forecast(sid, db_sdt, db_fpe)
                        except Exception as e:
                            logger.error(f"Nightly batch forecast failed for student {sid}: {e}")
                    logger.info("Nightly batch forecast updates successfully finalized.")
                finally:
                    db_sdt.close()
                    db_fpe.close()
                    
                # Sleep 65 seconds to prevent double execution in the same minute
                await asyncio.sleep(65)
            else:
                # Check every 30 seconds
                await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Error in nightly scheduler task loop: {e}")
            await asyncio.sleep(30)


# ==========================================
# Startup Lifecycle Hook
# ==========================================

@app.on_event("startup")
def on_startup():
    # Initialize SQLite database
    logger.info("Initializing FPE local databases...")
    init_db()
    
    # Start the nightly batch scheduler as a background task
    asyncio.create_task(nightly_batch_scheduler())
    logger.info("Startup complete.")

# Serve frontend dashboard static files
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
frontend_path = os.path.join(base_dir, "frontend")
if os.path.exists(frontend_path):
    app.mount("/dashboard", StaticFiles(directory=frontend_path, html=True), name="dashboard")

@app.get("/")
def redirect_to_dashboard():
    return RedirectResponse(url="/dashboard/")
