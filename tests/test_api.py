"""
API tests using FastAPI's TestClient, which runs the app in-process
(no need for a separate uvicorn server). These tests DO require a
running, reachable MySQL instance configured via .env, since /predict
and the GET endpoints hit the real database - this mirrors how the
API actually behaves rather than mocking the database away.
"""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.main import app  # noqa: E402

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in ("ok", "degraded")
    assert "database_connected" in body
    assert "model_loaded" in body


def test_predict_rejects_negative_amount():
    payload = {
        "sender_id": "USR00001", "receiver_id": "USR00002", "amount": -10,
        "merchant_category": "Grocery", "transaction_type": "P2P",
        "location": "Mumbai", "device_type": "Android_Phone", "upi_channel": "GPay",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_invalid_transaction_type():
    payload = {
        "sender_id": "USR00001", "receiver_id": "USR00002", "amount": 100,
        "merchant_category": "Grocery", "transaction_type": "NOT_A_REAL_TYPE",
        "location": "Mumbai", "device_type": "Android_Phone", "upi_channel": "GPay",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = {
        "sender_id": "USR00001", "amount": 100,
        "merchant_category": "Grocery", "transaction_type": "P2P",
        "location": "Mumbai", "device_type": "Android_Phone", "upi_channel": "GPay",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.integration
def test_predict_success_returns_expected_fields():
    payload = {
        "sender_id": "USR00001", "receiver_id": "USR00002", "amount": 250.0,
        "merchant_category": "Grocery", "transaction_type": "P2P",
        "location": "Mumbai", "device_type": "Android_Phone", "upi_channel": "GPay",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"].startswith("TXNAPI")
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert isinstance(body["risk_score"], int)
    assert isinstance(body["reasons"], list)


@pytest.mark.integration
def test_get_transactions_returns_list():
    response = client.get("/transactions?limit=5")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body, list)
    assert len(body) <= 5


@pytest.mark.integration
def test_get_alerts_filters_by_risk_level():
    response = client.get("/alerts?risk_level=HIGH&limit=5")
    assert response.status_code == 200
    body = response.json()
    for alert in body:
        assert alert["risk_level"] == "HIGH"


def test_get_alerts_rejects_invalid_risk_level():
    response = client.get("/alerts?risk_level=NOT_REAL")
    assert response.status_code == 400
