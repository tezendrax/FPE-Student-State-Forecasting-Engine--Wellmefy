import uvicorn
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fpe.config import settings

if __name__ == "__main__":
    print(f"Starting Future Prediction Engine Server on {settings.HOST}:{settings.PORT}...")
    uvicorn.run("fpe.main:app", host=settings.HOST, port=settings.PORT, reload=True)
