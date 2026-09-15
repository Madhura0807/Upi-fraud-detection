"""
Time-Based Anomaly Detection
==============================

WHY THIS IS SEPARATE FROM IQR/ISOLATION FOREST:

IQR and Isolation Forest both look at a transaction's features in
isolation from the TIME AXIS - they don't care whether transaction
volume across the whole system suddenly spiked in the last 15 minutes.
Time-based detection looks at behavior over a SLIDING TIME WINDOW,
which catches a different kind of anomaly: sudden system-wide or
per-user bursts that individual feature values might not flag on
their own.

METHOD: Bucket transactions into fixed time windows (per hour, per
user-hour), then compute a ROLLING MEAN and ROLLING STANDARD DEVIATION
of transaction count/amount over recent windows. A window is flagged
as a time anomaly if its count or total amount deviates more than
`z_threshold` standard deviations from the rolling mean - i.e. a
z-score check applied over time, not over the whole dataset at once.

This is intentionally simple and explainable: no black-box model,
just "is this hour unusual compared to the recent recent past for
this user."
"""

import os

import numpy as np
import pandas as pd

ROLLING_WINDOW = 6  # look back over the last 6 hourly buckets
Z_THRESHOLD = 2.5


def build_hourly_user_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions into (sender_id, hour_bucket) buckets:
    count of transactions and total amount per bucket. This is the
    unit that rolling statistics are computed over.
    """
    df = df.copy()
    df["hour_bucket"] = df["timestamp"].dt.floor("h")

    buckets = (
        df.groupby(["sender_id", "hour_bucket"], observed=True)
        .agg(txn_count=("transaction_id", "count"), total_amount=("amount", "sum"))
        .reset_index()
        .sort_values(["sender_id", "hour_bucket"])
    )
    return buckets


def flag_time_anomalies(buckets: pd.DataFrame,
                         window: int = ROLLING_WINDOW,
                         z_threshold: float = Z_THRESHOLD) -> pd.DataFrame:
    """
    For each user, computes a rolling mean/std of txn_count over the
    previous `window` buckets (excluding the current one, so we don't
    use the value to judge itself) and flags the bucket if its count
    is more than `z_threshold` standard deviations above that rolling
    mean. Only spikes ABOVE normal are flagged - a quiet hour isn't
    suspicious, a sudden burst is.
    """
    buckets = buckets.copy()
    buckets["rolling_mean"] = np.nan
    buckets["rolling_std"] = np.nan
    buckets["time_anomaly_flag"] = 0

    for sender, group in buckets.groupby("sender_id", observed=True):
        counts = group["txn_count"].to_numpy()
        idx = group.index.to_numpy()

        for i in range(len(counts)):
            history = counts[max(0, i - window):i]
            if len(history) >= 2:
                mean = float(np.mean(history))
                std = float(np.std(history))
                buckets.loc[idx[i], "rolling_mean"] = mean
                buckets.loc[idx[i], "rolling_std"] = std
                if std > 0 and (counts[i] - mean) / std > z_threshold:
                    buckets.loc[idx[i], "time_anomaly_flag"] = 1
                elif std == 0 and counts[i] > mean and mean > 0:
                    # No variance in recent history but this bucket
                    # still jumped above it - still worth flagging.
                    buckets.loc[idx[i], "time_anomaly_flag"] = 1

    return buckets


def merge_time_flags_into_transactions(df: pd.DataFrame, buckets: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_bucket"] = df["timestamp"].dt.floor("h")
    merged = df.merge(
        buckets[["sender_id", "hour_bucket", "time_anomaly_flag"]],
        on=["sender_id", "hour_bucket"],
        how="left",
    )
    merged["time_anomaly_flag"] = merged["time_anomaly_flag"].fillna(0).astype(int)
    merged = merged.drop(columns=["hour_bucket"])
    return merged


def run_time_anomaly_detection(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    buckets = build_hourly_user_buckets(df)
    buckets = flag_time_anomalies(buckets)
    df = merge_time_flags_into_transactions(df, buckets)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows with time anomaly flags -> {output_path}")
    return df, buckets


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_anomalies.csv")
    output_path = os.path.join(base, "..", "data", "processed", "upi_transactions_time.csv")

    df, buckets = run_time_anomaly_detection(input_path, output_path)

    print()
    print("Time anomaly flag rate:", df["time_anomaly_flag"].mean())
    print()
    print("Cross-tab: time anomaly flag vs known simulated fraud label")
    print(pd.crosstab(df["time_anomaly_flag"], df["is_fraud_simulated"]))
    print()
    print("Sample flagged buckets:")
    print(buckets[buckets["time_anomaly_flag"] == 1].head(8).to_string())
