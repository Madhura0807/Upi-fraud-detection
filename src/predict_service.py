"""
Real-Time Prediction Service
============================

This is the ONLINE counterpart to the batch pipeline
(feature_engineering.py -> anomaly_detection.py -> time_anomaly.py ->
fraud_scoring.py). Instead of processing a whole CSV at once, it
computes the SAME behavioral features for ONE incoming transaction by
querying the sender's transaction history from MySQL, then applies the
already-trained Isolation Forest model and the same IQR bounds /
scoring rules learned from the training data.

WHY BOUNDS/MODEL ARE LOADED ONCE, NOT RECOMPUTED PER REQUEST:

IQR bounds and the Isolation Forest model represent "what normal looks
like" across the whole population, learned once during the batch
training run. Recomputing them per request would be slow and would
also mean each prediction is judged against a slightly different
baseline - undermining consistency between predictions.
"""

import os
import statistics
from datetime import datetime

import joblib
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.anomaly_detection import (
    ISOLATION_FOREST_FEATURES,
    IQR_COLUMNS,
    compute_iqr_bounds,
)
from src.fraud_scoring import (
    AMOUNT_DEVIATION_THRESHOLD,
    REASON_LABELS,
    WEIGHTS,
    _risk_level_for_score,
)
from src.time_anomaly import ROLLING_WINDOW, Z_THRESHOLD


_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_MODEL_PATH = os.path.join(_BASE_DIR, "models", "isolation_forest.pkl")
_TRAINING_DATA_PATH = os.path.join(
    _BASE_DIR, "data", "processed", "upi_transactions_features.csv"
)

_model = None
_iqr_bounds: dict[str, tuple[float, float]] = {}


def _load_artifacts() -> None:
    """Loads the trained model and IQR bounds once, cached at module level."""
    global _model, _iqr_bounds

    if _model is None:
        _model = joblib.load(_MODEL_PATH)

    if not _iqr_bounds:
        training_df = pd.read_csv(_TRAINING_DATA_PATH)
        for col in IQR_COLUMNS:
            _, _, _, lower, upper = compute_iqr_bounds(training_df[col])
            _iqr_bounds[col] = (lower, upper)


def _fetch_sender_history(
    db: Session, sender_id: str, before: datetime
) -> pd.DataFrame:
    query = text(
        """
        SELECT timestamp, amount, receiver_id, device_type, location
        FROM transactions
        WHERE sender_id = :sender_id AND timestamp < :before
        ORDER BY timestamp
        """
    )

    rows = db.execute(
        query,
        {"sender_id": sender_id, "before": before},
    ).fetchall()

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "amount",
                "receiver_id",
                "device_type",
                "location",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "amount",
            "receiver_id",
            "device_type",
            "location",
        ],
    )

    # Normalize database timestamps to timezone-naive pandas datetimes.
    # This keeps them compatible with the incoming API timestamp.
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

    # MySQL DECIMAL columns come back as Python Decimal via PyMySQL,
    # which cannot be mixed with float in arithmetic.
    df["amount"] = df["amount"].astype(float)

    return df


def _compute_behavioral_features(txn: dict, history: pd.DataFrame) -> dict:
    """Mirrors feature_engineering.add_behavioral_features for a single new row."""
    ts = txn["timestamp"]

    if history.empty:
        return {
            "historical_avg_amount": txn["amount"],
            "amount_deviation": 0.0,
            "transactions_last_hour": 0,
            "time_since_last_transaction": -1.0,
            "recipient_frequency": 0,
            "device_changed": 0,
            "location_changed": 0,
        }

    amounts = history["amount"].tolist()
    avg = statistics.mean(amounts)
    std = statistics.pstdev(amounts) if len(amounts) > 1 else 0.0

    amount_deviation = (
        (txn["amount"] - avg) / std if std > 0 else 0.0
    )

    window_start = pd.Timestamp(ts) - pd.Timedelta(hours=1)
    txns_last_hour = int(
        (history["timestamp"] >= window_start).sum()
    )

    last_ts = pd.Timestamp(history["timestamp"].iloc[-1])
    time_since_last = (
        (pd.Timestamp(ts) - last_ts).total_seconds() / 60.0
    )

    recipient_frequency = int(
        (history["receiver_id"] == txn["receiver_id"]).sum()
    )

    most_common_device = history["device_type"].mode().iloc[0]
    device_changed = int(txn["device_type"] != most_common_device)

    most_common_location = history["location"].mode().iloc[0]
    location_changed = int(txn["location"] != most_common_location)

    return {
        "historical_avg_amount": avg,
        "amount_deviation": amount_deviation,
        "transactions_last_hour": txns_last_hour,
        "time_since_last_transaction": time_since_last,
        "recipient_frequency": recipient_frequency,
        "device_changed": device_changed,
        "location_changed": location_changed,
    }


def _compute_time_anomaly_flag(
    history: pd.DataFrame, ts: datetime
) -> int:
    """
    Mirrors time_anomaly.py's rolling z-score logic, applied to this
    sender's own hourly transaction-count history, for the hour bucket
    the new transaction falls into.
    """
    if history.empty:
        return 0

    hist = history.copy()
    hist["hour_bucket"] = pd.to_datetime(
        hist["timestamp"]
    ).dt.floor("h")

    bucket_counts = hist.groupby("hour_bucket").size()
    current_bucket = pd.Timestamp(ts).floor("h")

    # Count of this sender's transactions already in the CURRENT bucket.
    current_bucket_count = int(bucket_counts.get(current_bucket, 0)) + 1

    recent_counts = bucket_counts.tail(ROLLING_WINDOW).to_numpy()

    if len(recent_counts) < 2:
        return 0

    mean = float(recent_counts.mean())
    std = float(recent_counts.std())

    if std > 0 and (current_bucket_count - mean) / std > Z_THRESHOLD:
        return 1

    if std == 0 and current_bucket_count > mean and mean > 0:
        return 1

    return 0


def predict_transaction(db: Session, txn: dict) -> dict:
    """
    txn: dict with sender_id, receiver_id, amount, merchant_category,
    transaction_type, location, device_type, upi_channel, timestamp.

    Returns a dict ready to build PredictionResponse and to persist
    into fraud_alerts.
    """
    _load_artifacts()

    ts = txn["timestamp"] or datetime.now()

    # Swagger/Pydantic may provide a timezone-aware timestamp (e.g. the
    # trailing Z). MySQL/pandas history is timezone-naive, so normalize
    # the incoming timestamp before comparisons and feature calculations.
    if ts.tzinfo is not None:
        ts = ts.replace(tzinfo=None)

    txn["timestamp"] = ts

    history = _fetch_sender_history(
        db,
        txn["sender_id"],
        before=ts,
    )

    behavioral = _compute_behavioral_features(txn, history)

    hour = ts.hour
    day_of_week = ts.weekday()

    feature_row = {
        "amount": txn["amount"],
        "hour": hour,
        "day_of_week": day_of_week,
        **behavioral,
    }

    # --- IQR flag ---
    iqr_flag = False

    for col in IQR_COLUMNS:
        lower, upper = _iqr_bounds[col]
        value = feature_row[col]

        if value < lower or value > upper:
            iqr_flag = True
            break

    # --- Isolation Forest flag ---
    feature_vector = pd.DataFrame(
        [
            {
                k: feature_row[k]
                for k in ISOLATION_FOREST_FEATURES
            }
        ]
    )

    prediction = _model.predict(feature_vector)[0]
    isolation_forest_flag = bool(prediction == -1)

    # --- Time anomaly flag ---
    time_anomaly_flag = bool(
        _compute_time_anomaly_flag(history, ts)
    )

    # --- Rule-based flags (matches fraud_scoring.py exactly) ---
    device_changed = bool(behavioral["device_changed"])
    location_changed = bool(behavioral["location_changed"])
    new_recipient = behavioral["recipient_frequency"] == 0
    high_amount_deviation = (
        behavioral["amount_deviation"] > AMOUNT_DEVIATION_THRESHOLD
    )

    signals = {
        "iqr_flag": iqr_flag,
        "isolation_forest_flag": isolation_forest_flag,
        "time_anomaly_flag": time_anomaly_flag,
        "device_changed": device_changed,
        "location_changed": location_changed,
        "new_recipient": new_recipient,
        "high_amount_deviation": high_amount_deviation,
    }

    risk_score = sum(
        WEIGHTS[k]
        for k, v in signals.items()
        if v
    )

    risk_level = _risk_level_for_score(risk_score)

    reasons = [
        REASON_LABELS[k]
        for k, v in signals.items()
        if v
    ]

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "iqr_flag": iqr_flag,
        "isolation_forest_flag": isolation_forest_flag,
        "time_anomaly_flag": time_anomaly_flag,
        "device_change": device_changed,
        "location_change": location_changed,
        "reasons": reasons,
        "timestamp": ts,
    }
