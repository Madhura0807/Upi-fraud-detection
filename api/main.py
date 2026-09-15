"""
FastAPI app entrypoint. Run with:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.database import init_db  # noqa: E402
from api.routes import router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time UPI transaction fraud risk scoring backend.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "UPI Fraud Detection API is running.",
        "docs": "/docs",
        "endpoints": ["/health", "/predict", "/alerts", "/transactions"],
    }
