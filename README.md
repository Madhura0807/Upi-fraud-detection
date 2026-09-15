UPI Fraud Detection using Anomaly Detection

An end-to-end machine learning system for detecting potentially fraudulent UPI transactions using behavioral analysis, statistical anomaly detection, Isolation Forest, and transparent risk scoring.

This project uses a synthetic dataset and is intended for learning and portfolio purposes.

Features

Behavioral feature engineering

IQR-based anomaly detection

Isolation Forest

Time-based anomaly detection

Rapid transaction burst detection

New device, location, and recipient detection

Transparent risk scoring

FastAPI REST API

MySQL database

Streamlit dashboard

Model evaluation and testing

How It Works
Transaction
     |
     v
Feature Engineering
     |
     +-------- IQR Anomaly
     |
     +-------- Isolation Forest
     |
     +-------- Time Anomaly
     |
     +-------- Behavioral Signals
     |
     v
Risk Scoring
     |
     v
LOW / MEDIUM / HIGH


The system compares each transaction with the sender's historical behavior to identify unusual activity.

Detection Methods
IQR Anomaly Detection

Uses the Interquartile Range to detect statistical outliers in:

Transaction amount

Amount deviation

Transactions in the last hour

IQR = Q3 - Q1

Upper Bound = Q3 + 1.5 × IQR
Lower Bound = Q1 - 1.5 × IQR

Isolation Forest

An unsupervised machine learning algorithm that detects unusual combinations of behavioral features.

contamination = 0.06

Time Anomaly Detection

Uses the previous six hourly activity buckets to identify sudden increases in transaction activity.

Current Activity >
Rolling Mean + 2.5 × Rolling Standard Deviation

Behavioral Signals

The system detects:

New device

New location

New recipient

Rapid transaction burst

Unusually high transaction amount

Risk Scoring
Signal	Score
IQR anomaly	+1
Isolation Forest anomaly	+1
Time anomaly	+1
New device	+1
New location	+1
New recipient	+1
High amount	+1
Risk Levels
Score	Risk Level
0–1	LOW
2–3	MEDIUM
4+	HIGH

The risk score is a ranking signal and should not be interpreted as a calibrated probability of fraud.

Dataset

The project uses approximately 10,000 synthetic UPI transactions.

Features

Sender ID

Receiver ID

Amount

Merchant category

Transaction type

Location

Device type

UPI channel

Timestamp

Simulated Fraud Patterns
high_amount
odd_hour
new_device
new_location
new_receiver
rapid_burst


The simulated fraud labels are used only for evaluation and are not provided as inputs to the unsupervised anomaly detection models.

Model Evaluation
Method	Precision	Recall	F1 Score
IQR	0.264	0.483	0.342
Isolation Forest	0.739	0.528	0.616
Time Anomaly	0.511	0.281	0.363
Combined MEDIUM+	0.493	0.677	0.571
Combined HIGH	0.865	0.096	0.173
ROC-AUC
Combined Risk Score ROC-AUC = 0.873

System Architecture
Streamlit Dashboard
        |
        v
   FastAPI Backend
        |
        v
Feature Engineering
        |
        v
Anomaly Detection
        |
        v
   Risk Scoring
        |
        v
      MySQL

Project Structure
upi-fraud-detection/
|
├── api/
│   ├── main.py
│   ├── routes.py
│   └── database.py
|
├── dashboard/
│   └── app.py
|
├── src/
│   ├── data_generation.py
│   ├── feature_engineering.py
│   ├── anomaly_detection.py
│   ├── predict_service.py
│   └── ...
|
├── models/
├── data/
├── reports/
├── sql/
├── tests/
|
├── run_project.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md

Tech Stack

Python

Pandas

NumPy

Scikit-learn

FastAPI

Pydantic

MySQL

SQLAlchemy

PyMySQL

Streamlit

Joblib

Pytest

Git

Installation
Clone Repository
git clone <repository-url>
cd upi-fraud-detection

Create Virtual Environment
python -m venv venv

Activate Environment
Windows
.\venv\Scripts\Activate.ps1

Linux / macOS
source venv/bin/activate

Install Dependencies
pip install -r requirements.txt

Environment Configuration

Create a .env file in the project root.

DB_HOST=localhost
DB_PORT=3306
DB_NAME=upi_fraud_db
DB_USER=upi_app
DB_PASSWORD=your_password

API_HOST=0.0.0.0
API_PORT=8000
API_BASE_URL=http://localhost:8000


Do not commit .env or production credentials to GitHub.

Running the Project
Run Complete Pipeline
python run_project.py


This runs data generation, feature engineering, anomaly detection, evaluation, and database loading.

Start FastAPI
uvicorn api.main:app --reload


API:

http://localhost:8000


Swagger documentation:

http://localhost:8000/docs

Start Streamlit
streamlit run dashboard/app.py


Dashboard:

http://localhost:8501

Run Tests
pytest tests/ -v

API Endpoints
Method	Endpoint	Description
GET	/health	API, model, and database status
POST	/predict	Predict transaction risk
GET	/alerts	Retrieve fraud alerts
GET	/transactions	Retrieve transactions
Limitations

Uses synthetic transaction data

Fraud labels are simulated

Risk score is not a calibrated probability

Real-world fraud patterns may differ

New users may have limited historical behavior

No direct connection to UPI or banking systems

Future Improvements

Real-time processing with Apache Kafka

Redis-based feature storage

Graph-based fraud detection

SHAP-based explainability

Model drift monitoring

Automated model retraining

Docker and Kubernetes deployment

API authentication and authorization

Conclusion

This project demonstrates an end-to-end UPI fraud detection pipeline combining statistical anomaly detection, unsupervised machine learning, temporal analysis, behavioral signals, risk scoring, REST APIs, database persistence, and an interactive dashboard.

It is designed as a portfolio and learning implementation of a real-world fraud detection architecture.
