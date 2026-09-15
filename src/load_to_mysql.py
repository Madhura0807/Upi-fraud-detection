"""
Loads the fully processed + scored dataset into MySQL: one row per
transaction into `transactions`, and one row per transaction into
`fraud_alerts` (since every transaction gets scored, even LOW risk
ones - this gives the dashboard a complete history to query).

Uses bulk inserts via pandas.to_sql for speed on ~10k rows, going
through the same SQLAlchemy engine defined in api/database.py so
there is exactly ONE place that owns the DB connection string.
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.database import engine, init_db  # noqa: E402


def load_transactions(df: pd.DataFrame) -> None:
    txn_cols = [
        "transaction_id", "timestamp", "sender_id", "receiver_id", "amount",
        "merchant_category", "transaction_type", "location", "device_type",
        "upi_channel",
    ]
    txn_df = df[txn_cols].copy()
    txn_df.to_sql("transactions", con=engine, if_exists="append", index=False, chunksize=1000)
    print(f"Loaded {len(txn_df)} rows into `transactions`.")


def load_fraud_alerts(df: pd.DataFrame) -> None:
    alerts_df = pd.DataFrame({
        "transaction_id": df["transaction_id"],
        "risk_score": df["risk_score"],
        "risk_level": df["risk_level"],
        "iqr_flag": df["iqr_flag"].astype(bool),
        "isolation_forest_flag": df["isolation_forest_flag"].astype(bool),
        "time_anomaly_flag": df["time_anomaly_flag"].astype(bool),
        "device_change": df["device_changed"].astype(bool),
        "location_change": df["location_changed"].astype(bool),
        "reasons": df["reasons"],
    })
    alerts_df.to_sql("fraud_alerts", con=engine, if_exists="append", index=False, chunksize=1000)
    print(f"Loaded {len(alerts_df)} rows into `fraud_alerts`.")


def run_load(input_path: str) -> None:
    init_db()
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    with engine.begin() as conn:
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")
        conn.exec_driver_sql("TRUNCATE TABLE fraud_alerts")
        conn.exec_driver_sql("TRUNCATE TABLE transactions")
        conn.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")
    print("Cleared existing rows (fresh load).")

    load_transactions(df)
    load_fraud_alerts(df)


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_scored.csv")
    run_load(input_path)
