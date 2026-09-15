"""
Preprocessing Pipeline
=======================

Each step below states WHY it's needed - this is the part you should be
able to explain in an interview, not just "I called dropna()".
"""

import os

import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: `fraud_pattern` is blank for every normal transaction BY DESIGN
    (no rule was violated, so there's nothing to record). Pandas reads
    that blank CSV field as NaN. This is NOT missing/dirty data - it's
    a legitimate "not applicable" value, so we fill it with an explicit
    string rather than dropping rows or leaving NaN (which would break
    later groupby/value_counts operations).

    Any OTHER column with nulls would be genuinely missing data - we
    check for that separately and would need a real decision (drop vs
    impute) if it occurred.
    """
    df = df.copy()
    df["fraud_pattern"] = df["fraud_pattern"].fillna("none")

    other_cols = [c for c in df.columns if c != "fraud_pattern"]
    remaining_nulls = df[other_cols].isnull().sum()
    if remaining_nulls.sum() > 0:
        # Genuine missing data in core fields would compromise a
        # transaction record, so we drop those rows rather than
        # guessing values for something like `amount` or `sender_id`.
        before = len(df)
        df = df.dropna(subset=other_cols)
        print(f"Dropped {before - len(df)} rows with missing core fields.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: A duplicate transaction_id would mean the same transaction was
    recorded twice (e.g. a retry/resubmission bug upstream), which would
    double-count it in every aggregate and bias the anomaly detectors.
    We keep the first occurrence and drop the rest.
    """
    before = len(df)
    df = df.drop_duplicates(subset="transaction_id", keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"Removed {dropped} duplicate transaction(s).")
    return df


def fix_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: CSV round-tripping loses type information - `timestamp` comes
    back as plain text, and categorical fields come back as generic
    object dtype. Converting `timestamp` to datetime is required before
    ANY time-based feature (hour, day_of_week, rolling windows) can be
    computed. Categorical columns are cast to pandas 'category' dtype,
    which is more memory-efficient and communicates intent.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = df["amount"].astype(float)

    categorical_cols = [
        "merchant_category", "transaction_type", "location",
        "device_type", "upi_channel",
    ]
    for col in categorical_cols:
        df[col] = df[col].astype("category")

    df["is_fraud_simulated"] = df["is_fraud_simulated"].astype(int)
    return df


def validate_amounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: A transaction amount of zero or negative value is not a valid
    UPI transaction and would either be a data-entry error or a
    generation bug - either way it shouldn't reach the modeling stage.
    """
    before = len(df)
    df = df[df["amount"] > 0]
    dropped = before - len(df)
    if dropped:
        print(f"Removed {dropped} transaction(s) with invalid amount.")
    return df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Several later steps (time_since_last_transaction, rolling
    windows, transactions_last_hour) depend on processing transactions
    in time order, per user. Sorting once here avoids re-sorting in
    every downstream function.
    """
    return df.sort_values("timestamp").reset_index(drop=True)


def run_preprocessing_pipeline(input_path: str, output_path: str) -> pd.DataFrame:
    df = load_raw_data(input_path)
    print(f"Loaded {len(df)} raw transactions.")

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = validate_amounts(df)
    df = sort_chronologically(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} cleaned transactions -> {output_path}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "raw", "upi_transactions.csv")
    output_path = os.path.join(base, "..", "data", "processed", "upi_transactions_clean.csv")
    df = run_preprocessing_pipeline(input_path, output_path)

    print()
    print("Final dtypes:")
    print(df.dtypes)
    print()
    print("Nulls remaining:", df.isnull().sum().sum())
