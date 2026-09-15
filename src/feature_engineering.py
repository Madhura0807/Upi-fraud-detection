"""
Feature Engineering
=====================

CORE IDEA: A transaction is not suspicious in isolation - amount=50000
is normal for one user and wildly abnormal for another. So every feature
here is built RELATIVE TO THE SENDER'S OWN HISTORY, not just the raw
transaction. This is what "behavioral" means in fraud detection.

All rolling/history features are computed causally (only using
transactions that happened BEFORE the current one for that user), which
matters both for realism (you can't use future data to judge a
transaction happening now) and for correctness when this same logic
is reused for a single incoming transaction in the FastAPI /predict
endpoint.
"""

import os

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    hour, day_of_week: Fraudulent activity often clusters at unusual
    hours (late night) or specific days. Raw timestamp is too granular
    for a model to learn patterns from directly, so we extract cyclical
    components a model/rule can actually use.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday
    return df


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes, per sender, in chronological order:

    - historical_avg_amount: running mean of amount BEFORE this txn.
      This is the user's personal baseline.
    - amount_deviation: how many standard deviations this transaction
      is from the user's own historical average. This single feature
      does more work than raw amount, because it's user-relative.
    - transactions_last_hour: count of this user's transactions in the
      60 minutes before this one - a rapid burst is a classic account-
      takeover signal.
    - average_transaction_amount: overall running average transaction
      size for the user (kept alongside historical_avg_amount for
      interpretability in the dashboard/API output).
    - time_since_last_transaction: minutes since this user's previous
      transaction - unusually short gaps support the burst signal;
      unusually long gaps can indicate a dormant/compromised account
      suddenly reactivating.
    - recipient_frequency: how many times this user has paid this
      receiver before. A brand-new receiver getting a large first
      payment is a well-known fraud pattern.
    - device_changed / location_changed: whether this transaction's
      device/location differs from the user's most common
      (mode) device/location historically.
    """
    df = df.copy()
    df = df.sort_values(["sender_id", "timestamp"]).reset_index(drop=True)

    hist_avg = np.zeros(len(df))
    amount_dev = np.zeros(len(df))
    txns_last_hour = np.zeros(len(df))
    running_avg = np.zeros(len(df))
    time_since_last = np.zeros(len(df))
    recipient_freq = np.zeros(len(df))
    device_changed = np.zeros(len(df), dtype=int)
    location_changed = np.zeros(len(df), dtype=int)

    for sender, group in df.groupby("sender_id", observed=True):
        idx = group.index.to_numpy()
        amounts = group["amount"].to_numpy()
        timestamps = group["timestamp"].to_numpy()
        receivers = group["receiver_id"].to_numpy()
        devices = group["device_type"].to_numpy()
        locations = group["location"].to_numpy()

        seen_amounts = []
        seen_receivers: dict[str, int] = {}
        seen_devices: dict[str, int] = {}
        seen_locations: dict[str, int] = {}

        for i, row_idx in enumerate(idx):
            # --- historical average & deviation (uses only PAST txns) ---
            if seen_amounts:
                avg = float(np.mean(seen_amounts))
                std = float(np.std(seen_amounts)) if len(seen_amounts) > 1 else 0.0
            else:
                avg = amounts[i]  # first txn: no history yet, baseline = itself
                std = 0.0

            hist_avg[row_idx] = avg
            running_avg[row_idx] = avg
            amount_dev[row_idx] = (amounts[i] - avg) / std if std > 0 else 0.0

            # --- transactions in the last 60 minutes (past txns only) ---
            if i > 0:
                past_ts = timestamps[:i]
                window_start = timestamps[i] - np.timedelta64(1, "h")
                txns_last_hour[row_idx] = int(np.sum(past_ts >= window_start))

                gap_minutes = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, "m")
                time_since_last[row_idx] = float(gap_minutes)
            else:
                txns_last_hour[row_idx] = 0
                time_since_last[row_idx] = -1  # sentinel: first-ever transaction

            # --- recipient frequency (past occurrences of this receiver) ---
            recipient_freq[row_idx] = seen_receivers.get(receivers[i], 0)

            # --- device / location changed vs user's historical MODE ---
            if seen_devices:
                most_common_device = max(seen_devices, key=seen_devices.get)
                device_changed[row_idx] = int(devices[i] != most_common_device)
            if seen_locations:
                most_common_location = max(seen_locations, key=seen_locations.get)
                location_changed[row_idx] = int(locations[i] != most_common_location)

            # update running history AFTER computing this row's features
            seen_amounts.append(amounts[i])
            seen_receivers[receivers[i]] = seen_receivers.get(receivers[i], 0) + 1
            seen_devices[devices[i]] = seen_devices.get(devices[i], 0) + 1
            seen_locations[locations[i]] = seen_locations.get(locations[i], 0) + 1

    df["historical_avg_amount"] = hist_avg
    df["amount_deviation"] = amount_dev
    df["transactions_last_hour"] = txns_last_hour.astype(int)
    df["average_transaction_amount"] = running_avg
    df["time_since_last_transaction"] = time_since_last
    df["recipient_frequency"] = recipient_freq.astype(int)
    df["device_changed"] = device_changed
    df["location_changed"] = location_changed

    return df.sort_values("timestamp").reset_index(drop=True)


def run_feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = add_time_features(df)
    df = add_behavioral_features(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} feature-engineered rows -> {output_path}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_clean.csv")
    output_path = os.path.join(base, "..", "data", "processed", "upi_transactions_features.csv")
    df = run_feature_engineering(input_path, output_path)

    print()
    print("New feature columns preview:")
    cols = ["amount", "historical_avg_amount", "amount_deviation",
            "transactions_last_hour", "time_since_last_transaction",
            "recipient_frequency", "device_changed", "location_changed",
            "is_fraud_simulated"]
    print(df[cols].head(10).to_string())
    print()
    print("Sanity check - mean feature values by fraud label:")
    print(df.groupby("is_fraud_simulated")[
        ["amount_deviation", "transactions_last_hour", "device_changed",
         "location_changed", "recipient_frequency"]
    ].mean())
