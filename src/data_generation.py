"""
Synthetic UPI Transaction Data Generator
=========================================

WHY THIS APPROACH (read this before touching the code):

A dataset of pure random noise is useless for a fraud-detection portfolio
project, because there is no realistic "normal" behavior for the anomaly
detectors to learn against, and no honest way to evaluate them afterwards.

So generation happens in two layers:

1. USER PROFILES - each synthetic user (sender) gets a fixed behavioral
   baseline: a typical spending range, a preferred set of active hours,
   a home device, a home location, and a small pool of regular receivers
   they transact with repeatedly. Most of a user's transactions are drawn
   from THEIR OWN profile, which is what makes the "normal" class look
   like a real distribution instead of noise.

2. INJECTED ANOMALIES - a minority of transactions are deliberately built
   to violate that SAME user's profile in specific, named ways (unusually
   high amount, odd hour, new device, new location, new/rare receiver,
   or a rapid burst of transactions). Each injected transaction carries a
   ground-truth label (`is_fraud_simulated`) and a machine-readable list
   of which rule(s) were violated (`fraud_pattern`).

IMPORTANT: `is_fraud_simulated` is ONLY used later for evaluation
(comparing detector output against known injected patterns). The
unsupervised detectors (IQR, Isolation Forest) never see this column
during detection - that would defeat the purpose of unsupervised anomaly
detection and make the whole exercise dishonest.
"""

import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Reproducibility: fixed seeds everywhere so the dataset (and every
# downstream model trained on it) is reproducible run-to-run.
# ------------------------------------------------------------------
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_USERS = 500
N_TRANSACTIONS = 10_000
FRAUD_RATE = 0.06  # ~6% of transactions are injected suspicious patterns

MERCHANT_CATEGORIES = [
    "Grocery", "Food Delivery", "Utilities", "Shopping", "Entertainment",
    "Fuel", "Healthcare", "Education", "Travel", "P2P Transfer", "Rent",
]
TRANSACTION_TYPES = ["P2P", "P2M"]
LOCATIONS = [
    "Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad", "Chennai",
    "Kolkata", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Nagpur",
]
DEVICE_TYPES = ["Android_Phone", "iPhone", "Web", "Tablet"]
UPI_CHANNELS = ["GPay", "PhonePe", "Paytm", "BHIM", "AmazonPay"]

FRAUD_PATTERNS = [
    "high_amount",
    "odd_hour",
    "new_device",
    "new_location",
    "new_receiver",
    "rapid_burst",
]


def _random_id(prefix: str, n: int) -> str:
    return f"{prefix}{n:05d}"


def _generate_user_profiles(n_users: int) -> list[dict]:
    """
    Build one behavioral baseline per user. This is what "normal" is
    measured against, both during generation and later during feature
    engineering (e.g. historical_avg_amount).
    """
    profiles = []
    for i in range(n_users):
        avg_amount = float(np.random.choice(
            [np.random.uniform(50, 500), np.random.uniform(500, 3000),
             np.random.uniform(3000, 15000)],
            p=None,
        ))
        active_hours = sorted(random.sample(range(6, 23), k=random.randint(4, 8)))
        home_device = random.choice(DEVICE_TYPES)
        home_location = random.choice(LOCATIONS)
        # a small pool of receivers this user regularly pays
        regular_receivers = [_random_id("USR", random.randint(0, n_users - 1))
                              for _ in range(random.randint(2, 6))]
        preferred_channel = random.choice(UPI_CHANNELS)

        profiles.append({
            "user_id": _random_id("USR", i),
            "avg_amount": round(avg_amount, 2),
            "active_hours": active_hours,
            "home_device": home_device,
            "home_location": home_location,
            "regular_receivers": regular_receivers,
            "preferred_channel": preferred_channel,
        })
    return profiles


def _random_timestamp(base_date: datetime, hour_pool: list[int]) -> datetime:
    day_offset = random.randint(0, 89)  # spread across ~3 months
    hour = random.choice(hour_pool)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return base_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second) \
        - base_date.replace(hour=0, minute=0, second=0, microsecond=0) + \
        base_date.replace(hour=0, minute=0, second=0, microsecond=0)


def _normal_transaction(profile: dict, txn_counter: int, base_date: datetime) -> dict:
    """A transaction consistent with the user's own behavioral profile."""
    amount = max(10.0, np.random.normal(profile["avg_amount"], profile["avg_amount"] * 0.25))
    hour = random.choice(profile["active_hours"])
    day_offset = random.randint(0, 89)
    ts = base_date + timedelta(days=day_offset, hours=hour,
                                minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    receiver = random.choice(profile["regular_receivers"])

    return {
        "transaction_id": _random_id("TXN", txn_counter),
        "timestamp": ts,
        "sender_id": profile["user_id"],
        "receiver_id": receiver,
        "amount": round(amount, 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "location": profile["home_location"],
        "device_type": profile["home_device"],
        "upi_channel": profile["preferred_channel"],
        "is_fraud_simulated": 0,
        "fraud_pattern": "",
    }


def _suspicious_transaction(profile: dict, txn_counter: int, base_date: datetime,
                             all_user_ids: list[str]) -> dict:
    """
    A transaction that deliberately violates ONE OR MORE aspects of the
    user's own profile. The specific violated rule(s) are recorded in
    `fraud_pattern` purely for later evaluation/debugging - detectors
    never see this field.
    """
    n_patterns = np.random.choice([1, 2], p=[0.7, 0.3])
    patterns = random.sample(FRAUD_PATTERNS, k=n_patterns)

    amount = np.random.normal(profile["avg_amount"], profile["avg_amount"] * 0.25)
    hour = random.choice(profile["active_hours"])
    location = profile["home_location"]
    device = profile["home_device"]
    receiver = random.choice(profile["regular_receivers"])
    day_offset = random.randint(0, 89)
    minute = random.randint(0, 59)

    if "high_amount" in patterns:
        amount = profile["avg_amount"] * np.random.uniform(6, 15)
    if "odd_hour" in patterns:
        hour = random.choice([0, 1, 2, 3, 4])
    if "new_device" in patterns:
        device = random.choice([d for d in DEVICE_TYPES if d != profile["home_device"]])
    if "new_location" in patterns:
        location = random.choice([l for l in LOCATIONS if l != profile["home_location"]])
    if "new_receiver" in patterns:
        receiver = random.choice(all_user_ids)

    amount = max(10.0, amount)
    ts = base_date + timedelta(days=day_offset, hours=hour, minutes=minute,
                                seconds=random.randint(0, 59))

    return {
        "transaction_id": _random_id("TXN", txn_counter),
        "timestamp": ts,
        "sender_id": profile["user_id"],
        "receiver_id": receiver,
        "amount": round(amount, 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "transaction_type": random.choice(TRANSACTION_TYPES),
        "location": location,
        "device_type": device,
        "upi_channel": profile["preferred_channel"],
        "is_fraud_simulated": 1,
        "fraud_pattern": ",".join(patterns),
    }


def _inject_rapid_bursts(df: pd.DataFrame, profiles: list[dict], base_date: datetime,
                          txn_counter_start: int, n_bursts: int = 60) -> pd.DataFrame:
    """
    Adds short bursts of 3-6 transactions within a few minutes of each
    other for a random user - simulating a compromised account being
    drained quickly. This pattern is intentionally handled separately
    because it's a MULTI-ROW pattern (relationship between transactions),
    unlike the other single-row anomalies above.
    """
    rows = []
    counter = txn_counter_start
    for _ in range(n_bursts):
        profile = random.choice(profiles)
        burst_size = random.randint(3, 6)
        day_offset = random.randint(0, 89)
        start_hour = random.randint(0, 23)
        start_minute = random.randint(0, 50)
        start_ts = base_date + timedelta(days=day_offset, hours=start_hour, minutes=start_minute)

        for j in range(burst_size):
            ts = start_ts + timedelta(minutes=j * random.randint(1, 3))
            amount = max(10.0, np.random.normal(profile["avg_amount"] * 1.5,
                                                  profile["avg_amount"] * 0.3))
            rows.append({
                "transaction_id": _random_id("TXN", counter),
                "timestamp": ts,
                "sender_id": profile["user_id"],
                "receiver_id": random.choice(profile["regular_receivers"]),
                "amount": round(amount, 2),
                "merchant_category": random.choice(MERCHANT_CATEGORIES),
                "transaction_type": random.choice(TRANSACTION_TYPES),
                "location": profile["home_location"],
                "device_type": profile["home_device"],
                "upi_channel": profile["preferred_channel"],
                "is_fraud_simulated": 1,
                "fraud_pattern": "rapid_burst",
            })
            counter += 1
    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


def generate_dataset(n_transactions: int = N_TRANSACTIONS,
                      n_users: int = N_USERS,
                      fraud_rate: float = FRAUD_RATE) -> pd.DataFrame:
    base_date = datetime(2026, 1, 1)
    profiles = _generate_user_profiles(n_users)
    all_user_ids = [p["user_id"] for p in profiles]

    n_fraud = int(n_transactions * fraud_rate)
    n_normal = n_transactions - n_fraud

    rows = []
    counter = 0
    for _ in range(n_normal):
        profile = random.choice(profiles)
        rows.append(_normal_transaction(profile, counter, base_date))
        counter += 1

    for _ in range(n_fraud):
        profile = random.choice(profiles)
        rows.append(_suspicious_transaction(profile, counter, base_date, all_user_ids))
        counter += 1

    df = pd.DataFrame(rows)
    df = _inject_rapid_bursts(df, profiles, base_date, txn_counter_start=counter, n_bursts=60)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import os

    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "upi_transactions.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"Simulated fraud rate: {df['is_fraud_simulated'].mean():.2%}")
    print(df["fraud_pattern"].value_counts())
