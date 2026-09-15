"""
IQR-Based Anomaly Detection + Isolation Forest
================================================

PART 1: IQR (Interquartile Range) - a simple, explainable STATISTICAL
baseline. For a numeric column:

    Q1 = 25th percentile, Q3 = 75th percentile
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

Any value outside [lower_bound, upper_bound] is flagged as a statistical
outlier. The 1.5x multiplier is a standard convention (Tukey's fences) -
it's not a guarantee of fraud, just "this value is unusually far from
where most values in this column sit." We apply it per relevant numeric
column (not just raw amount) and combine the flags with OR logic.

IMPORTANT: an IQR outlier is a SIGNAL, not a verdict. A legitimately
huge rent payment is a real outlier that isn't fraud - that's exactly
why IQR is only ONE input into the final risk score, not the decision
by itself.

PART 2: Isolation Forest - an UNSUPERVISED machine learning model
(scikit-learn). It works by randomly partitioning the feature space
with random splits; anomalous points are "isolated" (separated from
the rest of the data) in fewer splits than normal points, because they
sit apart from the dense regions where most data lives. The model
outputs an anomaly score for each row; more negative = more anomalous.

WHY Isolation Forest specifically:
- It doesn't require labeled fraud data to train (unsupervised), which
  matches real-world fraud detection where confirmed fraud labels are
  rare/delayed.
- It scales well and handles multi-dimensional feature interactions
  that a single-column rule like IQR cannot see (e.g. "high amount AND
  new device AND odd hour" together, even if no single feature is
  extreme on its own).
- `contamination` is a hyperparameter that tells the model the
  EXPECTED proportion of anomalies in the data - it's used to set the
  decision threshold on the anomaly score, not to bias training.

WHAT IT IS NOT: Isolation Forest is NOT a supervised fraud classifier.
It has no concept of "fraud" - it only knows "this point looks
structurally different from the bulk of the data." Whether that
difference means fraud, a legitimate rare event, or noise is exactly
what the downstream risk-scoring layer has to reason about.
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

IQR_COLUMNS = ["amount", "amount_deviation", "transactions_last_hour"]

ISOLATION_FOREST_FEATURES = [
    "amount", "hour", "day_of_week", "amount_deviation",
    "transactions_last_hour", "time_since_last_transaction",
    "recipient_frequency", "device_changed", "location_changed",
]

CONTAMINATION = 0.06  # matches our known injected fraud rate (~6%)
RANDOM_SEED = 42


def compute_iqr_bounds(series: pd.Series) -> tuple[float, float, float, float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return q1, q3, iqr, lower_bound, upper_bound


def apply_iqr_detection(df: pd.DataFrame, columns: list[str] = IQR_COLUMNS) -> pd.DataFrame:
    """
    Flags a row as an IQR anomaly if ANY of the specified numeric
    columns falls outside that column's own IQR bounds. Bounds are
    printed so they can be inspected/explained.
    """
    df = df.copy()
    flags = np.zeros(len(df), dtype=bool)

    print("IQR bounds per feature:")
    for col in columns:
        q1, q3, iqr, lower, upper = compute_iqr_bounds(df[col])
        print(f"  {col}: Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, "
              f"bounds=({lower:.2f}, {upper:.2f})")
        col_flag = (df[col] < lower) | (df[col] > upper)
        flags |= col_flag.to_numpy()

    df["iqr_flag"] = flags.astype(int)
    return df


def train_isolation_forest(df: pd.DataFrame,
                            features: list[str] = ISOLATION_FOREST_FEATURES,
                            contamination: float = CONTAMINATION) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(df[features])
    return model


def apply_isolation_forest(df: pd.DataFrame, model: IsolationForest,
                            features: list[str] = ISOLATION_FOREST_FEATURES) -> pd.DataFrame:
    """
    model.predict() returns -1 for anomalies, 1 for normal points.
    We convert that to a 0/1 flag matching our other flags, and also
    keep the raw anomaly score (more negative = more anomalous) for
    transparency in the dashboard.
    """
    df = df.copy()
    predictions = model.predict(df[features])
    scores = model.decision_function(df[features])

    df["isolation_forest_flag"] = (predictions == -1).astype(int)
    df["isolation_forest_score"] = scores
    return df


def run_anomaly_detection(input_path: str, output_path: str, model_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    df = apply_iqr_detection(df)

    model = train_isolation_forest(df)
    df = apply_isolation_forest(df, model)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved trained Isolation Forest -> {model_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows with anomaly flags -> {output_path}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_features.csv")
    output_path = os.path.join(base, "..", "data", "processed", "upi_transactions_anomalies.csv")
    model_path = os.path.join(base, "..", "models", "isolation_forest.pkl")

    df = run_anomaly_detection(input_path, output_path, model_path)

    print()
    print("IQR flag rate:", df["iqr_flag"].mean())
    print("Isolation Forest flag rate:", df["isolation_forest_flag"].mean())
    print()
    print("Cross-tab: IQR flag vs known simulated fraud label")
    print(pd.crosstab(df["iqr_flag"], df["is_fraud_simulated"]))
    print()
    print("Cross-tab: Isolation Forest flag vs known simulated fraud label")
    print(pd.crosstab(df["isolation_forest_flag"], df["is_fraud_simulated"]))
