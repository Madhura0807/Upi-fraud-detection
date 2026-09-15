"""
Database layer - SQLAlchemy engine, session, and ORM models.

WHY THIS IS SEPARATE FROM api/routes.py:
Keeping DB connection/session setup and table definitions out of the
route handlers means (1) routes stay focused on request/response
logic, (2) the same DB layer can be reused by scripts outside the API
(e.g. a batch loader), and (3) it's easier to swap out the database
later without touching route code.

Credentials are read from environment variables (via python-dotenv),
NEVER hard-coded - see .env.example for the expected variables.
"""

import os

from dotenv import load_dotenv
from sqlalchemy import (
    DECIMAL,
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker
from sqlalchemy.sql import func

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "upi_fraud_db")
DB_USER = os.getenv("DB_USER", "upi_app")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Transaction(Base):
    __tablename__ = "transactions"

    transaction_id = Column(String(20), primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    sender_id = Column(String(20), nullable=False, index=True)
    receiver_id = Column(String(20), nullable=False, index=True)
    amount = Column(DECIMAL(12, 2), nullable=False)
    merchant_category = Column(String(50), nullable=False)
    transaction_type = Column(String(20), nullable=False)
    location = Column(String(50), nullable=False)
    device_type = Column(String(30), nullable=False)
    upi_channel = Column(String(30), nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    alerts = relationship("FraudAlert", back_populates="transaction", cascade="all, delete-orphan")


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"

    alert_id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(20), ForeignKey("transactions.transaction_id", ondelete="CASCADE"),
                             nullable=False, index=True)
    risk_score = Column(Integer, nullable=False)
    risk_level = Column(Enum("LOW", "MEDIUM", "HIGH", name="risk_level_enum"),
                         nullable=False, index=True)
    iqr_flag = Column(Boolean, default=False)
    isolation_forest_flag = Column(Boolean, default=False)
    time_anomaly_flag = Column(Boolean, default=False)
    device_change = Column(Boolean, default=False)
    location_change = Column(Boolean, default=False)
    reasons = Column(Text)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    transaction = relationship("Transaction", back_populates="alerts")


def get_db():
    """FastAPI dependency: yields a session and always closes it."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Creates all tables if they don't already exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
