import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import add_behavioral_features, add_time_features


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "transaction_id": ["T1", "T2", "T3", "T4"],
        "timestamp": pd.to_datetime([
            "2026-01-01 10:00:00", "2026-01-01 10:05:00",
            "2026-01-02 09:00:00", "2026-01-02 09:30:00",
        ]),
        "sender_id": ["USR001", "USR001", "USR001", "USR002"],
        "receiver_id": ["USR010", "USR010", "USR011", "USR010"],
        "amount": [100.0, 5000.0, 150.0, 300.0],
        "device_type": ["Android_Phone", "Android_Phone", "iPhone", "Web"],
        "location": ["Mumbai", "Mumbai", "Delhi", "Pune"],
    })


def test_add_time_features(sample_df):
    result = add_time_features(sample_df)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns
    assert result.loc[0, "hour"] == 10


def test_first_transaction_has_no_history_deviation(sample_df):
    """A user's very first transaction should have zero deviation and
    zero recipient frequency, since there's no history to compare to."""
    result = add_behavioral_features(sample_df)
    first_txn = result[result["transaction_id"] == "T1"].iloc[0]
    assert first_txn["amount_deviation"] == 0.0
    assert first_txn["recipient_frequency"] == 0
    assert first_txn["time_since_last_transaction"] == -1.0


def test_repeat_receiver_increments_frequency(sample_df):
    """T1 and T2 both go to USR010 from USR001 - by T2, recipient
    frequency should reflect one prior payment to that receiver."""
    result = add_behavioral_features(sample_df)
    second_txn = result[result["transaction_id"] == "T2"].iloc[0]
    assert second_txn["recipient_frequency"] == 1


def test_device_change_detected(sample_df):
    """T3 uses a different device than USR001's established pattern
    (Android_Phone seen twice before) - should be flagged as changed."""
    result = add_behavioral_features(sample_df)
    third_txn = result[result["transaction_id"] == "T3"].iloc[0]
    assert third_txn["device_changed"] == 1


def test_no_null_values_in_output(sample_df):
    result = add_behavioral_features(sample_df)
    behavioral_cols = [
        "historical_avg_amount", "amount_deviation", "transactions_last_hour",
        "average_transaction_amount", "time_since_last_transaction",
        "recipient_frequency", "device_changed", "location_changed",
    ]
    assert result[behavioral_cols].isnull().sum().sum() == 0
