"""
Pydantic schemas - request/response validation for the FastAPI layer.

Using Pydantic here means malformed requests (wrong types, missing
fields, negative amounts) are rejected automatically with a clear
422 error before any of our detection logic runs.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

MERCHANT_CATEGORIES = [
    "Grocery", "Food Delivery", "Utilities", "Shopping", "Entertainment",
    "Fuel", "Healthcare", "Education", "Travel", "P2P Transfer", "Rent",
]
TRANSACTION_TYPES = ["P2P", "P2M"]
DEVICE_TYPES = ["Android_Phone", "iPhone", "Web", "Tablet"]
UPI_CHANNELS = ["GPay", "PhonePe", "Paytm", "BHIM", "AmazonPay"]


class TransactionInput(BaseModel):
    sender_id: str = Field(..., examples=["USR00042"])
    receiver_id: str = Field(..., examples=["USR00099"])
    amount: float = Field(..., gt=0, description="Transaction amount, must be positive")
    merchant_category: str = Field(..., examples=["Shopping"])
    transaction_type: str = Field(..., examples=["P2P"])
    location: str = Field(..., examples=["Mumbai"])
    device_type: str = Field(..., examples=["Android_Phone"])
    upi_channel: str = Field(..., examples=["GPay"])
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Defaults to current server time if not provided",
    )

    @field_validator("transaction_type")
    @classmethod
    def validate_transaction_type(cls, v: str) -> str:
        if v not in TRANSACTION_TYPES:
            raise ValueError(f"transaction_type must be one of {TRANSACTION_TYPES}")
        return v

    @field_validator("device_type")
    @classmethod
    def validate_device_type(cls, v: str) -> str:
        if v not in DEVICE_TYPES:
            raise ValueError(f"device_type must be one of {DEVICE_TYPES}")
        return v


class PredictionResponse(BaseModel):
    transaction_id: str
    risk_score: int
    risk_level: str
    iqr_flag: bool
    isolation_forest_flag: bool
    time_anomaly_flag: bool
    device_change: bool
    location_change: bool
    reasons: list[str]


class TransactionOut(BaseModel):
    model_config = {"from_attributes": True}

    transaction_id: str
    timestamp: datetime
    sender_id: str
    receiver_id: str
    amount: float
    merchant_category: str
    transaction_type: str
    location: str
    device_type: str
    upi_channel: str


class AlertOut(BaseModel):
    model_config = {"from_attributes": True}

    alert_id: int
    transaction_id: str
    risk_score: int
    risk_level: str
    reasons: Optional[str]
    created_at: datetime


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    database_connected: bool
    model_loaded: bool
