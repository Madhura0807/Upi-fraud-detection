"""
Model Evaluation
===================

WHY NOT JUST "ACCURACY":

Fraud is a RARE-EVENT / IMBALANCED problem - in our synthetic data,
roughly 8% of transactions are labeled fraud, so a detector that
predicts "not fraud" for EVERYTHING would already score ~92% accuracy
while catching zero fraud. Accuracy is therefore a misleading metric
here. We report:

  - PRECISION: of the transactions we flagged, how many were actually
    fraud? (cost of false positives - annoying/blocking real users)
  - RECALL: of the actual fraud transactions, how many did we catch?
    (cost of false negatives - fraud that slips through)
  - F1-SCORE: harmonic mean of precision and recall, useful as a
    single number when both matter.
  - CONFUSION MATRIX: the raw counts behind those metrics.
  - ROC-AUC: how well a continuous score (here, risk_score) ranks
    fraud above non-fraud across all thresholds, independent of any
    single cutoff.

EVALUATED AGAINST: `is_fraud_simulated`, the ground-truth label we
injected during data generation. This is a HONEST LIMITATION worth
stating clearly (see bottom of this file / README): these are KNOWN,
RULE-INJECTED patterns, not real confirmed fraud cases. Real fraud
data is messier, evolves over time (adversarial), and is expensive/
slow to label. Performance here should be read as "does the pipeline
recover the specific patterns we deliberately built in," not as a
claim about real-world fraud-catching performance.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_binary_flag(y_true: pd.Series, y_pred: pd.Series, name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"--- {name} ---")
    print(f"Confusion matrix:\n{cm}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print()

    return {"name": name, "confusion_matrix": cm, "precision": precision,
            "recall": recall, "f1": f1}


def evaluate_risk_score(y_true: pd.Series, risk_score: pd.Series, out_dir: str) -> float:
    """
    ROC-AUC treats `risk_score` (0-7) as a continuous ranking signal:
    does a higher score correlate with actual fraud, across ALL
    possible thresholds? This is threshold-independent, unlike
    precision/recall which depend on where we draw the LOW/MEDIUM/HIGH
    cutoffs.
    """
    auc = roc_auc_score(y_true, risk_score)
    fpr, tpr, _ = roc_curve(y_true, risk_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Risk Score (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Fraud Risk Score vs Simulated Fraud Label")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=120)
    plt.close(fig)

    print(f"ROC-AUC (risk_score as ranking signal): {auc:.3f}")
    return auc


def save_confusion_matrix_plot(y_true: pd.Series, y_pred: pd.Series,
                                title: str, filename: str, out_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Fraud"]).plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, filename), dpi=120)
    plt.close(fig)


def run_evaluation(input_path: str, out_dir: str) -> None:
    df = pd.read_csv(input_path)
    y_true = df["is_fraud_simulated"]
    os.makedirs(out_dir, exist_ok=True)

    results = []
    results.append(evaluate_binary_flag(y_true, df["iqr_flag"], "IQR Detector"))
    results.append(evaluate_binary_flag(y_true, df["isolation_forest_flag"], "Isolation Forest"))
    results.append(evaluate_binary_flag(y_true, df["time_anomaly_flag"], "Time Anomaly Detector"))

    # Final combined detector: MEDIUM or HIGH risk level counts as "flagged"
    combined_flag = df["risk_level"].isin(["MEDIUM", "HIGH"]).astype(int)
    results.append(evaluate_binary_flag(y_true, combined_flag, "Combined (risk_level >= MEDIUM)"))

    high_only_flag = (df["risk_level"] == "HIGH").astype(int)
    results.append(evaluate_binary_flag(y_true, high_only_flag, "Combined (risk_level == HIGH only)"))

    save_confusion_matrix_plot(y_true, combined_flag,
                                "Combined Detector (>= MEDIUM) vs Simulated Fraud",
                                "confusion_matrix_combined.png", out_dir)

    auc = evaluate_risk_score(y_true, df["risk_score"], out_dir)

    print("=" * 60)
    print("LIMITATIONS OF THIS EVALUATION (read before quoting these numbers):")
    print("1. Ground truth is SYNTHETIC/INJECTED, not real confirmed fraud.")
    print("   The system is being scored on how well it recovers patterns")
    print("   we deliberately built in - not on real-world generalization.")
    print("2. Real fraud is adversarial and evolves; these patterns are static.")
    print("3. Class imbalance (~8% fraud) means precision/recall trade-offs")
    print("   matter more than accuracy - accuracy is not reported for this")
    print("   reason.")
    print("=" * 60)

    return results, auc


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_scored.csv")
    out_dir = os.path.join(base, "..", "reports", "figures")
    run_evaluation(input_path, out_dir)
