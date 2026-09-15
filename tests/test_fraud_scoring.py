import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fraud_scoring import compute_fraud_risk


def _base_row(**overrides):
    row = {
        "iqr_flag": 0,
        "isolation_forest_flag": 0,
        "time_anomaly_flag": 0,
        "device_changed": 0,
        "location_changed": 0,
        "recipient_frequency": 5,
        "amount_deviation": 0.5,
    }
    row.update(overrides)
    return row


def test_no_signals_means_low_risk():
    df = pd.DataFrame([_base_row()])
    result = compute_fraud_risk(df)
    assert result.loc[0, "risk_score"] == 0
    assert result.loc[0, "risk_level"] == "LOW"


def test_all_signals_mean_high_risk():
    df = pd.DataFrame([_base_row(
        iqr_flag=1, isolation_forest_flag=1, time_anomaly_flag=1,
        device_changed=1, location_changed=1, recipient_frequency=0,
        amount_deviation=5.0,
    )])
    result = compute_fraud_risk(df)
    assert result.loc[0, "risk_score"] == 7
    assert result.loc[0, "risk_level"] == "HIGH"


def test_medium_risk_boundary():
    """Exactly 2 signals should land in MEDIUM (per RISK_THRESHOLDS)."""
    df = pd.DataFrame([_base_row(iqr_flag=1, isolation_forest_flag=1)])
    result = compute_fraud_risk(df)
    assert result.loc[0, "risk_score"] == 2
    assert result.loc[0, "risk_level"] == "MEDIUM"


def test_new_recipient_detected_from_zero_frequency():
    df = pd.DataFrame([_base_row(recipient_frequency=0)])
    result = compute_fraud_risk(df)
    assert result.loc[0, "new_recipient"] == 1


def test_reasons_list_matches_triggered_flags():
    df = pd.DataFrame([_base_row(iqr_flag=1, device_changed=1)])
    result = compute_fraud_risk(df)
    reasons = result.loc[0, "reasons"]
    assert "Statistical outlier (IQR)" in reasons
    assert "New/unusual device detected" in reasons
    assert "Flagged by Isolation Forest model" not in reasons


def test_risk_score_is_never_negative():
    df = pd.DataFrame([_base_row()])
    result = compute_fraud_risk(df)
    assert (result["risk_score"] >= 0).all()
