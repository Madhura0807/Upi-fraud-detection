import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    fix_data_types,
    handle_missing_values,
    remove_duplicates,
    validate_amounts,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "transaction_id": ["TXN001", "TXN002", "TXN002", "TXN003"],
        "timestamp": ["2026-01-01 10:00:00", "2026-01-01 11:00:00",
                       "2026-01-01 11:00:00", "2026-01-01 12:00:00"],
        "sender_id": ["USR001", "USR002", "USR002", "USR003"],
        "receiver_id": ["USR010", "USR011", "USR011", "USR012"],
        "amount": [100.0, -50.0, -50.0, 200.0],
        "merchant_category": ["Grocery", "Shopping", "Shopping", "Fuel"],
        "transaction_type": ["P2P", "P2M", "P2M", "P2P"],
        "location": ["Mumbai", "Delhi", "Delhi", "Pune"],
        "device_type": ["Android_Phone", "iPhone", "iPhone", "Web"],
        "upi_channel": ["GPay", "PhonePe", "PhonePe", "Paytm"],
        "is_fraud_simulated": [0, 1, 1, 0],
        "fraud_pattern": [np.nan, "high_amount", "high_amount", np.nan],
    })


def test_handle_missing_values_fills_fraud_pattern(sample_df):
    result = handle_missing_values(sample_df)
    assert result["fraud_pattern"].isnull().sum() == 0
    assert (result["fraud_pattern"] == "none").sum() == 2


def test_remove_duplicates(sample_df):
    result = remove_duplicates(sample_df)
    assert result["transaction_id"].is_unique
    assert len(result) == 3


def test_fix_data_types_converts_timestamp(sample_df):
    result = fix_data_types(sample_df)
    assert pd.api.types.is_datetime64_any_dtype(result["timestamp"])


def test_validate_amounts_removes_negative(sample_df):
    result = validate_amounts(sample_df)
    assert (result["amount"] > 0).all()
    assert len(result) == 2  # two rows had amount=-50.0
