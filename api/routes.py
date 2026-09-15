"""
API routes. Kept separate from main.py (app setup) and database.py
(connection/session/ORM), so each file has one job.
"""

import os
import sys
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.database import FraudAlert, Transaction, engine, get_db  # noqa: E402
from api.schemas import (  # noqa: E402
    AlertOut,
    HealthResponse,
    PredictionResponse,
    TransactionInput,
    TransactionOut,
)
from src.predict_service import predict_transaction  # noqa: E402

router = APIRouter()


def _generate_transaction_id() -> str:
    return f"TXNAPI{uuid.uuid4().hex[:8].upper()}"


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    model_ok = os.path.exists(os.path.join(
        os.path.dirname(__file__), "..", "models", "isolation_forest.pkl"))

    return HealthResponse(status="ok" if db_ok and model_ok else "degraded",
                           database_connected=db_ok, model_loaded=model_ok)


@router.post("/predict", response_model=PredictionResponse)
def predict(txn_input: TransactionInput, db: Session = Depends(get_db)):
    transaction_id = _generate_transaction_id()
    ts = txn_input.timestamp or datetime.now()

    txn_dict = txn_input.model_dump()
    txn_dict["timestamp"] = ts

    try:
        result = predict_transaction(db, txn_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # Persist the transaction
    db_txn = Transaction(
        transaction_id=transaction_id,
        timestamp=result["timestamp"],
        sender_id=txn_input.sender_id,
        receiver_id=txn_input.receiver_id,
        amount=txn_input.amount,
        merchant_category=txn_input.merchant_category,
        transaction_type=txn_input.transaction_type,
        location=txn_input.location,
        device_type=txn_input.device_type,
        upi_channel=txn_input.upi_channel,
    )
    db.add(db_txn)

    # Persist the alert
    db_alert = FraudAlert(
        transaction_id=transaction_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        iqr_flag=result["iqr_flag"],
        isolation_forest_flag=result["isolation_forest_flag"],
        time_anomaly_flag=result["time_anomaly_flag"],
        device_change=result["device_change"],
        location_change=result["location_change"],
        reasons="; ".join(result["reasons"]) if result["reasons"] else "No anomaly signals triggered",
    )
    db.add(db_alert)
    db.commit()

    return PredictionResponse(
        transaction_id=transaction_id,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        iqr_flag=result["iqr_flag"],
        isolation_forest_flag=result["isolation_forest_flag"],
        time_anomaly_flag=result["time_anomaly_flag"],
        device_change=result["device_change"],
        location_change=result["location_change"],
        reasons=result["reasons"],
    )


@router.get("/alerts", response_model=list[AlertOut])
def get_alerts(
    risk_level: Optional[str] = Query(default=None, description="Filter by LOW, MEDIUM, or HIGH"),
    limit: int = Query(default=50, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(FraudAlert)
    if risk_level:
        if risk_level.upper() not in ("LOW", "MEDIUM", "HIGH"):
            raise HTTPException(status_code=400, detail="risk_level must be LOW, MEDIUM, or HIGH")
        query = query.filter(FraudAlert.risk_level == risk_level.upper())
    alerts = query.order_by(FraudAlert.created_at.desc()).limit(limit).all()
    return alerts


@router.get("/transactions", response_model=list[TransactionOut])
def get_transactions(
    sender_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=500),
    db: Session = Depends(get_db),
):
    query = db.query(Transaction)
    if sender_id:
        query = query.filter(Transaction.sender_id == sender_id)
    transactions = query.order_by(Transaction.timestamp.desc()).limit(limit).all()
    return transactions
