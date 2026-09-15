"""
Fraud Risk Scoring
====================

WHY A RULE-BASED SCORE INSTEAD OF A SINGLE ML PROBABILITY:

We have THREE independent, unsupervised signals (IQR, Isolation Forest,
time anomaly) plus several rule-based behavioral flags (device change,
location change, new/rare recipient, high amount deviation). None of
them individually should be trusted as "the" fraud decision - each one
catches a different pattern and each one produces false positives on
its own (see the cross-tabs from earlier stages).

So instead of picking one model's output as ground truth, we SUM
independent binary signals into a score. This is:
  - TRANSPARENT: every point in the score traces back to a specific,
    nameable reason - critical for an interview and for a real fraud
    analyst reviewing an alert.
  - CONFIGURABLE: weights and thresholds live in one place (`WEIGHTS`,
    `RISK_THRESHOLDS`) and can be tuned without touching detection
    logic.
  - HONEST ABOUT WHAT IT IS: this is a **Fraud Risk Score**, not a
    calibrated fraud probability. A calibrated probability would
    require a properly labeled, non-synthetic dataset and a supervised
    model trained/validated against it - out of scope for a synthetic
    portfolio project, and we don't pretend otherwise.

SCORING LOGIC (default weights, all = 1 point unless noted):
    iqr_flag                     -> "Statistical outlier (IQR)"
    isolation_forest_flag        -> "Flagged by Isolation Forest model"
    time_anomaly_flag            -> "Unusual transaction volume for this hour"
    device_changed                -> "New/unusual device detected"
    location_changed              -> "New/unusual location detected"
    new_recipient (freq == 0)     -> "First-time recipient"
    high_amount_deviation (>3σ)   -> "Amount far above user's historical average"

RISK LEVELS (configurable):
    0-1 points -> LOW
    2-3 points -> MEDIUM
    4+  points -> HIGH
"""

import os

import pandas as pd

WEIGHTS = {
    "iqr_flag": 1,
    "isolation_forest_flag": 1,
    "time_anomaly_flag": 1,
    "device_changed": 1,
    "location_changed": 1,
    "new_recipient": 1,
    "high_amount_deviation": 1,
}

REASON_LABELS = {
    "iqr_flag": "Statistical outlier (IQR)",
    "isolation_forest_flag": "Flagged by Isolation Forest model",
    "time_anomaly_flag": "Unusual transaction volume for this hour",
    "device_changed": "New/unusual device detected",
    "location_changed": "New/unusual location detected",
    "new_recipient": "First-time recipient",
    "high_amount_deviation": "Amount far above user's historical average",
}

RISK_THRESHOLDS = {
    "LOW": (0, 1),
    "MEDIUM": (2, 3),
    "HIGH": (4, float("inf")),
}

AMOUNT_DEVIATION_THRESHOLD = 3.0


def _risk_level_for_score(score: int) -> str:
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score <= high:
            return level
    return "HIGH"


def compute_fraud_risk(df: pd.DataFrame,
                        weights: dict = WEIGHTS,
                        amount_dev_threshold: float = AMOUNT_DEVIATION_THRESHOLD) -> pd.DataFrame:
    df = df.copy()

    df["new_recipient"] = (df["recipient_frequency"] == 0).astype(int)
    df["high_amount_deviation"] = (df["amount_deviation"] > amount_dev_threshold).astype(int)

    signal_cols = list(weights.keys())
    for col in signal_cols:
        if col not in df.columns:
            raise ValueError(f"Expected signal column '{col}' not found in dataframe.")

    df["risk_score"] = sum(df[col] * weights[col] for col in signal_cols)
    df["risk_level"] = df["risk_score"].apply(_risk_level_for_score)

    def build_reasons(row) -> str:
        reasons = [REASON_LABELS[col] for col in signal_cols if row[col] == 1]
        return "; ".join(reasons) if reasons else "No anomaly signals triggered"

    df["reasons"] = df.apply(build_reasons, axis=1)
    return df


def run_fraud_scoring(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = compute_fraud_risk(df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows with fraud risk scores -> {output_path}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_time.csv")
    output_path = os.path.join(base, "..", "data", "processed", "upi_transactions_scored.csv")

    df = run_fraud_scoring(input_path, output_path)

    print()
    print("Risk level distribution:")
    print(df["risk_level"].value_counts())
    print()
    print("Cross-tab: risk level vs known simulated fraud label")
    print(pd.crosstab(df["risk_level"], df["is_fraud_simulated"]))
    print()
    print("Sample HIGH risk rows:")
    print(df[df["risk_level"] == "HIGH"][
        ["transaction_id", "sender_id", "amount", "risk_score", "reasons"]
    ].head(5).to_string())
