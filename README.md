# UPI Fraud Detection using AI & Anomaly Detection

An end-to-end fraud monitoring system for UPI (Unified Payments Interface) transactions, combining statistical anomaly detection, unsupervised machine learning, behavioral feature engineering, a MySQL-backed FastAPI service, and a live Streamlit dashboard.

This is a **portfolio / learning project built on synthetic data** — see [Limitations](#limitations) for what that does and doesn't mean about the results below.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Tech Stack](#tech-stack)
6. [Dataset Generation](#dataset-generation)
7. [Feature Engineering](#feature-engineering)
8. [IQR Methodology](#iqr-methodology)
9. [Isolation Forest Methodology](#isolation-forest-methodology)
10. [Time-Based Anomaly Detection](#time-based-anomaly-detection)
11. [Fraud Risk Scoring](#fraud-risk-scoring)
12. [Model Evaluation](#model-evaluation)
13. [MySQL Database Design](#mysql-database-design)
14. [SQL Analysis Examples](#sql-analysis-examples)
15. [FastAPI Documentation](#fastapi-documentation)
16. [Streamlit Dashboard](#streamlit-dashboard)
17. [Project Structure](#project-structure)
18. [Installation](#installation)
19. [Environment Variables](#environment-variables)
20. [How to Run](#how-to-run)
21. [Example API Request/Response](#example-api-requestresponse)
22. [Screenshots](#screenshots)
23. [Limitations](#limitations)
24. [Future Improvements](#future-improvements)

---

## Project Overview

UPI processes billions of transactions a month in India, and fraud detection at that scale can't rely on a single technique — no one signal (a fixed amount threshold, one ML model, etc.) catches every kind of suspicious behavior without also generating a flood of false positives. This project builds a small but realistic **layered detection system**: multiple independent, explainable signals are combined into one transparent risk score, backed by a real relational database and served through a production-shaped API and dashboard.

## Problem Statement

Given a stream of UPI transactions, flag the ones that look suspicious **relative to each user's own behavior** — not against a single global threshold — and explain *why* each one was flagged, in language a fraud analyst (or an interviewer) can actually evaluate.

## Features

- Synthetic but behaviorally realistic UPI transaction generator (~10,000 transactions, per-user behavioral profiles, six named fraud patterns)
- Statistical (IQR) and unsupervised ML (Isolation Forest) anomaly detection
- Time-window—based anomaly detection (rolling z-score on transaction volume)
- 8 behavioral features computed relative to each sender's own history
- Transparent, configurable, rule-based Fraud Risk Score (not a black-box probability)
- MySQL database with a proper schema (foreign keys, indexes, constraints)
- FastAPI backend with real-time `/predict`, plus `/health`, `/alerts`, `/transactions`
- Streamlit dashboard: KPIs, charts, alert table, and a live prediction form wired to the API
- Automated pytest suite (29 tests) covering data generation, preprocessing, features, scoring, and the API

## Architecture

```
Synthetic Data Generator
        │
        ▼
  Preprocessing  ──►  EDA (saved figures)
        │
        ▼
Feature Engineering (behavioral features, per-sender history)
        │
        ▼
   ┌────────────┬─────────────────────┬───────────────────────┐
   │            │                     │
IQR Detector   Isolation Forest    Time-Based Anomaly Detector
   │            │                     │
   └────────────┴─────────────────────┘
                 │
                 ▼
        Fraud Risk Scoring
       (transparent, weighted, rule-based)
                 │
                 ▼
          MySQL Database
     (transactions, fraud_alerts)
           │           │
           ▼           ▼
     FastAPI      Streamlit Dashboard
   (/predict,      (KPIs, charts, alert
  /alerts, etc.)    table, live prediction)
```

Each detector is independent and disagrees with the others sometimes — that's intentional and realistic. The risk score is a transparent aggregation, not a single model's opaque output.

## Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn (Isolation Forest), Joblib |
| Visualization | Matplotlib, Seaborn, Plotly |
| Backend | FastAPI, Pydantic, Uvicorn |
| Database | MySQL, SQLAlchemy, PyMySQL / mysql-connector-python |
| Dashboard | Streamlit |
| Testing | Pytest |

## Dataset Generation

`src/data_generation.py` generates ~10,000 synthetic transactions using **per-user behavioral profiles** (typical spending range, active hours, home device/location, a regular pool of receivers), so that "normal" behavior looks like a real distribution rather than noise.

A minority (~6-8%) of transactions are then generated to deliberately **violate that same user's profile** in one of six named ways:

| Pattern | What it simulates |
|---|---|
| `high_amount` | Amount 6-15x the user's historical average |
| `odd_hour` | Transaction during 12am-4am when the user is never normally active |
| `new_device` | A device the user has never used before |
| `new_location` | A location the user has never transacted from before |
| `new_receiver` | Payment to someone outside the user's regular circle |
| `rapid_burst` | 3-6 transactions within a few minutes (simulated account takeover) |

Every row carries a ground-truth `is_fraud_simulated` label and a `fraud_pattern` tag, used **only for evaluation** — never fed into the unsupervised detectors, which would defeat the purpose of unsupervised detection.

Reproducible via a fixed random seed (`RANDOM_SEED = 42`).

## Feature Engineering

`src/feature_engineering.py` builds features **relative to each sender's own transaction history**, computed causally (only using transactions strictly before the current one):

| Feature | Why it matters |
|---|---|
| `hour`, `day_of_week` | Fraud often clusters at unusual times |
| `historical_avg_amount` | The user's personal baseline |
| `amount_deviation` | How many std devs this transaction is from that baseline — does more work than raw amount |
| `transactions_last_hour` | Classic account-takeover signal (rapid burst) |
| `time_since_last_transaction` | Very short gaps support the burst signal; very long gaps flag a dormant account reactivating |
| `recipient_frequency` | A brand-new receiver getting a large first payment is a known fraud pattern |
| `device_changed` / `location_changed` | Deviation from the user's historically most common device/location |

## IQR Methodology

For a numeric column: `Q1` = 25th percentile, `Q3` = 75th percentile, `IQR = Q3 - Q1`, and Tukey's fences define the normal range as `[Q1 - 1.5×IQR, Q3 + 1.5×IQR]`. Anything outside that range is flagged. This is applied to `amount`, `amount_deviation`, and `transactions_last_hour`, combined with OR logic.

**Important:** an IQR outlier is a *signal*, not a verdict — a legitimately large rent payment is a real statistical outlier that isn't fraud. That's why it's only one input into the final score.

## Isolation Forest Methodology

Isolation Forest (scikit-learn) is an **unsupervised** model: it randomly partitions the feature space, and anomalous points get isolated in fewer splits than normal points because they sit apart from dense regions. `contamination` (set to 0.06, matching the known injected fraud rate) tunes the decision threshold on the anomaly score — it does not bias training with labels.

It's used here because it needs no labeled fraud data (matching real-world fraud detection, where confirmed labels are rare and delayed) and it can catch multi-feature interactions (e.g. "high amount + new device + odd hour together") that a single-column rule like IQR cannot see.

**It is not a supervised fraud classifier** — it has no concept of "fraud," only "structurally different from the bulk of the data."

## Time-Based Anomaly Detection

`src/time_anomaly.py` buckets each user's transactions into hourly windows and computes a rolling mean/std of transaction count over the previous 6 buckets. A bucket is flagged if its count is more than 2.5 standard deviations above that rolling mean — catching sudden volume spikes that individual-transaction features might miss.

## Fraud Risk Scoring

`src/fraud_scoring.py` sums independent binary signals into one score:

| Signal | Points |
|---|---|
| IQR flag | +1 |
| Isolation Forest flag | +1 |
| Time anomaly flag | +1 |
| Device changed | +1 |
| Location changed | +1 |
| First-time recipient | +1 |
| Amount > 3σ above user average | +1 |

| Score | Risk Level |
|---|---|
| 0-1 | LOW |
| 2-3 | MEDIUM |
| 4+ | HIGH |

Weights and thresholds are configurable in one place. This is called a **Fraud Risk Score**, deliberately not a "fraud probability" — a calibrated probability would require a real, confirmed-fraud-labeled dataset, which this synthetic project does not have.

## Model Evaluation

Evaluated against the known `is_fraud_simulated` label (see [Limitations](#limitations) for what this can and can't tell you):

| Detector | Precision | Recall | F1 |
|---|---|---|---|
| IQR only | 0.264 | 0.483 | 0.342 |
| Isolation Forest only | 0.739 | 0.528 | 0.616 |
| Time anomaly only | 0.511 | 0.281 | 0.363 |
| Combined (MEDIUM or HIGH) | 0.493 | 0.677 | 0.571 |
| Combined (HIGH only) | 0.865 | 0.096 | 0.173 |

**ROC-AUC (risk_score as a ranking signal): 0.873**

Accuracy is deliberately not reported: fraud is a rare-event problem (~8% of this dataset), so a detector predicting "not fraud" for everything would already score ~92% accuracy while catching nothing.

The HIGH-only threshold trades recall for very high precision (86.5%) — useful if the goal is "only escalate what we're quite sure about." The combined MEDIUM+ threshold is more balanced. This precision/recall trade-off, and how you'd tune it for a real deployment, is a good thing to discuss in an interview.

## MySQL Database Design

Two tables, `transactions` and `fraud_alerts`, in a 1-to-many relationship (a transaction can in principle be re-scored, producing more than one alert record over time):

- `transactions.transaction_id` is the primary key (UPI transaction IDs are already globally unique).
- `fraud_alerts.transaction_id` is a foreign key to `transactions`, with `ON DELETE CASCADE`.
- Indexes on `sender_id`, `receiver_id`, `timestamp`, `merchant_category` (transactions) and `transaction_id`, `risk_level`, `created_at` (fraud_alerts) — chosen because those are the columns actually filtered/joined on in the API and analysis queries, not indexed indiscriminately.
- A `CHECK` constraint enforces `amount > 0`.

Full DDL: [`sql/schema.sql`](sql/schema.sql).

## SQL Analysis Examples

See [`sql/analysis_queries.sql`](sql/analysis_queries.sql) for all 8 queries (tested against the live database). Highlights:

- High-risk transactions with their reasons (a join across both tables)
- Transaction count and volume per user
- Average transaction amount by merchant category
- Suspicious transactions by hour of day
- Users with unusually high transaction frequency (mean + 2×stddev, a SQL-native version of the z-score idea used in Python)
- Daily transaction volume
- Most common individual fraud reasons (via a recursive CTE splitting the semicolon-delimited `reasons` field)

## FastAPI Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Reports API/DB/model status |
| `/predict` | POST | Scores a new transaction in real time and persists it |
| `/alerts` | GET | Recent fraud alerts, optional `risk_level` filter |
| `/transactions` | GET | Recent transactions, optional `sender_id` filter |

Interactive docs available at `/docs` once the server is running (auto-generated by FastAPI from the Pydantic schemas).

**How `/predict` works:** it queries the sender's transaction history from MySQL, computes the same behavioral features as the batch pipeline (historical average, deviation, time-since-last, recipient frequency, device/location change), applies the pre-trained Isolation Forest model and the IQR bounds learned during training, and combines everything with the same scoring logic — so a prediction made through the API is scored identically to how the batch pipeline would have scored it.

## Streamlit Dashboard

`dashboard/app.py` connects directly to MySQL for fast read-only views (Overview KPIs, transaction volume over time, amount distribution, risk-level distribution, suspicious-transactions-by-hour, merchant category breakdown, and a filterable alert table), and calls the FastAPI `/predict` endpoint for the **Real-Time Prediction** form — so the dashboard exercises the same code path any real client integration would use.

## Project Structure

```
upi-fraud-detection/
├── data/
│   ├── raw/                    # upi_transactions.csv
│   └── processed/              # cleaned + feature-engineered + scored data
├── notebooks/
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py     # IQR + Isolation Forest
│   ├── time_anomaly.py
│   ├── fraud_scoring.py
│   ├── evaluation.py
│   ├── load_to_mysql.py
│   └── predict_service.py       # real-time scoring, shared with the API
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── routes.py
│   └── database.py
├── dashboard/
│   └── app.py
├── models/
│   └── isolation_forest.pkl
├── reports/figures/
├── sql/
│   ├── schema.sql
│   └── analysis_queries.sql
├── tests/
├── requirements.txt
├── .env.example
├── .gitignore
├── pytest.ini
├── README.md
└── run_project.py
```

## Installation

```bash
git clone <your-repo-url>
cd upi-fraud-detection

python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

You'll also need a running MySQL 8 server. Create the app database and a dedicated user:

```sql
CREATE DATABASE upi_fraud_db CHARACTER SET utf8mb4;
CREATE USER 'upi_app'@'localhost' IDENTIFIED BY 'your_password_here';
GRANT ALL PRIVILEGES ON upi_fraud_db.* TO 'upi_app'@'localhost';
FLUSH PRIVILEGES;
```

## Environment Variables

Copy `.env.example` to `.env` and fill in real values — **never commit `.env`**:

```
DB_HOST=localhost
DB_PORT=3306
DB_NAME=upi_fraud_db
DB_USER=upi_app
DB_PASSWORD=your_password_here

API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=http://localhost:8000
```

## How to Run

**1. Run the full pipeline** (data generation → preprocessing → EDA → features → detection → scoring → evaluation → MySQL load):

```bash
python run_project.py
```

**2. Start the API:**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**3. Start the dashboard** (in a separate terminal):

```bash
streamlit run dashboard/app.py
```

**4. Run the tests:**

```bash
pytest tests/ -v
```

## Example API Request/Response

**Request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sender_id": "USR00042",
    "receiver_id": "USR00499",
    "amount": 95000.00,
    "merchant_category": "Shopping",
    "transaction_type": "P2P",
    "location": "Delhi",
    "device_type": "Web",
    "upi_channel": "GPay",
    "timestamp": "2026-09-11T03:15:00"
  }'
```

**Response:**

```json
{
  "transaction_id": "TXNAPID28C4278",
  "risk_score": 6,
  "risk_level": "HIGH",
  "iqr_flag": true,
  "isolation_forest_flag": true,
  "time_anomaly_flag": false,
  "device_change": true,
  "location_change": true,
  "reasons": [
    "Statistical outlier (IQR)",
    "Flagged by Isolation Forest model",
    "New/unusual device detected",
    "New/unusual location detected",
    "First-time recipient",
    "Amount far above user's historical average"
  ]
}
```

## Screenshots

_Add screenshots of the Streamlit dashboard (Overview, Visualizations, Alert Table, Real-Time Prediction) and the FastAPI `/docs` page here before publishing._

## Limitations

- **Synthetic ground truth.** `is_fraud_simulated` marks patterns *we deliberately built in* — the evaluation numbers show how well the pipeline recovers those specific, static patterns, not how well it would generalize to real, evolving fraud.
- **No adversarial adaptation.** Real fraudsters adapt to detection; this dataset doesn't simulate that arms race.
- **Cold-start users.** A brand-new user (no transaction history) can only be judged against population-level detectors (Isolation Forest, IQR) — behavioral features default to zero deviation, since there's nothing yet to deviate from.
- **Risk Score is not a calibrated probability.** It's an interpretable, weighted count of independent signals — useful for triage and explanation, not for statements like "this transaction has an 80% chance of being fraud."
- **Single-node MySQL setup.** No replication, backups, or connection pooling tuning — fine for a portfolio project, not for production transaction volumes.

## Future Improvements

- Retrain/recalibrate on real (properly anonymized and labeled) transaction data if it ever became available
- Add a supervised model layered on top of the unsupervised signals, once real fraud labels exist
- Move the FastAPI service to async DB access (`asyncpg`/async SQLAlchemy) for higher throughput
- Add authentication/rate-limiting to the API before any real deployment
- Track model/data drift over time and add automated retraining triggers
- Expand the Streamlit dashboard with per-user drill-down views and a "mark as reviewed" workflow for alerts
