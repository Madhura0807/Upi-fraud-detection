"""
UPI Fraud Detection - Streamlit Dashboard

Reads aggregated data directly from MySQL (fast, no need to round-trip
through the API for read-only views), and calls the FastAPI /predict
endpoint for the real-time prediction tool (so the dashboard exercises
the exact same code path a real client integration would use).
"""

import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from api.schemas import DEVICE_TYPES, MERCHANT_CATEGORIES, TRANSACTION_TYPES, UPI_CHANNELS  # noqa: E402

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "upi_fraud_db")
DB_USER = os.getenv("DB_USER", "upi_app")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

st.set_page_config(page_title="UPI Fraud Detection Dashboard", layout="wide")


@st.cache_resource
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=30)
def load_overview_data() -> pd.DataFrame:
    engine = get_engine()
    query = """
        SELECT t.transaction_id, t.timestamp, t.sender_id, t.receiver_id,
               t.amount, t.merchant_category, t.location, t.device_type,
               f.risk_score, f.risk_level, f.reasons
        FROM transactions t
        JOIN fraud_alerts f ON t.transaction_id = f.transaction_id
    """
    return pd.read_sql(query, engine, parse_dates=["timestamp"])


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    suspicious = df[df["risk_level"].isin(["MEDIUM", "HIGH"])]
    col2.metric("Suspicious Transactions", f"{len(suspicious):,}")
    high_risk = df[df["risk_level"] == "HIGH"]
    col3.metric("High-Risk Transactions", f"{len(high_risk):,}")
    col4.metric("Avg Transaction Amount", f"₹{df['amount'].mean():,.2f}")


def render_visualizations(df: pd.DataFrame) -> None:
    st.subheader("Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        daily = df.set_index("timestamp").resample("D").size().reset_index(name="count")
        fig = px.line(daily, x="timestamp", y="count", title="Transaction Volume Over Time")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(df, x="amount", nbins=50, title="Transaction Amount Distribution",
                            log_y=True)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        risk_counts = df["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        fig = px.bar(x=risk_counts.index, y=risk_counts.values, title="Risk Level Distribution",
                     labels={"x": "Risk Level", "y": "Count"},
                     color=risk_counts.index,
                     color_discrete_map={"LOW": "#4C72B0", "MEDIUM": "#DD8452", "HIGH": "#C44E52"})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        df["hour"] = df["timestamp"].dt.hour
        suspicious_by_hour = (
            df[df["risk_level"].isin(["MEDIUM", "HIGH"])]
            .groupby("hour").size().reindex(range(24), fill_value=0)
        )
        fig = px.bar(x=suspicious_by_hour.index, y=suspicious_by_hour.values,
                     title="Suspicious Transactions by Hour",
                     labels={"x": "Hour of Day", "y": "Count"})
        st.plotly_chart(fig, use_container_width=True)

    merchant_avg = df.groupby("merchant_category")["amount"].mean().sort_values(ascending=False)
    fig = px.bar(x=merchant_avg.values, y=merchant_avg.index, orientation="h",
                 title="Average Transaction Amount by Merchant Category",
                 labels={"x": "Average Amount (₹)", "y": ""})
    st.plotly_chart(fig, use_container_width=True)


def render_alert_table(df: pd.DataFrame) -> None:
    st.subheader("Alert Table")

    risk_filter = st.multiselect("Filter by risk level", ["LOW", "MEDIUM", "HIGH"],
                                  default=["MEDIUM", "HIGH"])
    filtered = df[df["risk_level"].isin(risk_filter)].sort_values("timestamp", ascending=False)

    st.dataframe(
        filtered[["transaction_id", "timestamp", "amount", "sender_id", "receiver_id",
                  "risk_score", "risk_level", "reasons"]].head(200),
        use_container_width=True,
        hide_index=True,
    )


def render_real_time_prediction() -> None:
    st.subheader("Real-Time Prediction")
    st.caption(f"Sends the transaction to the FastAPI backend at {API_BASE_URL}/predict")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            sender_id = st.text_input("Sender ID", value="USR00042")
            receiver_id = st.text_input("Receiver ID", value="USR00099")
            amount = st.number_input("Amount (₹)", min_value=0.01, value=500.0, step=10.0)
            merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)
        with col2:
            transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
            location = st.text_input("Location", value="Mumbai")
            device_type = st.selectbox("Device Type", DEVICE_TYPES)
            upi_channel = st.selectbox("UPI Channel", UPI_CHANNELS)

        submitted = st.form_submit_button("Check Transaction")

    if submitted:
        payload = {
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "transaction_type": transaction_type,
            "location": location,
            "device_type": device_type,
            "upi_channel": upi_channel,
        }
        try:
            response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the FastAPI backend: {e}")
            return

        risk_level = result["risk_level"]
        color = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red"}[risk_level]

        st.markdown(f"### Risk Level: :{color}[{risk_level}]  (score: {result['risk_score']})")

        flag_cols = st.columns(5)
        flags = [
            ("IQR", result["iqr_flag"]),
            ("Isolation Forest", result["isolation_forest_flag"]),
            ("Time Anomaly", result["time_anomaly_flag"]),
            ("Device Change", result["device_change"]),
            ("Location Change", result["location_change"]),
        ]
        for col, (label, value) in zip(flag_cols, flags):
            col.metric(label, "Yes" if value else "No")

        st.markdown("**Reasons:**")
        if result["reasons"]:
            for reason in result["reasons"]:
                st.markdown(f"- {reason}")
        else:
            st.markdown("- No anomaly signals triggered")


def main() -> None:
    st.title("UPI Fraud Detection Dashboard")

    try:
        df = load_overview_data()
    except Exception as e:
        st.error(f"Could not load data from MySQL: {e}")
        st.stop()

    render_overview(df)
    st.divider()
    render_visualizations(df)
    st.divider()
    render_alert_table(df)
    st.divider()
    render_real_time_prediction()


if __name__ == "__main__":
    main()
