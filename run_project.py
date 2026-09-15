"""
run_project.py
================

Runs the full UPI Fraud Detection pipeline end-to-end, in order:

    1. Generate synthetic data
    2. Preprocess
    3. Exploratory data analysis (saves figures)
    4. Feature engineering
    5. IQR + Isolation Forest anomaly detection
    6. Time-based anomaly detection
    7. Fraud risk scoring
    8. Model evaluation (prints metrics, saves ROC curve + confusion matrix)
    9. Load results into MySQL

After this script finishes, start the API and dashboard separately:

    uvicorn api.main:app --reload
    streamlit run dashboard/app.py

Usage:
    python run_project.py
    python run_project.py --skip-mysql   # run everything except the DB load
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.anomaly_detection import run_anomaly_detection
from src.data_generation import generate_dataset
from src.eda import run_eda
from src.evaluation import run_evaluation
from src.feature_engineering import run_feature_engineering
from src.fraud_scoring import run_fraud_scoring
from src.load_to_mysql import run_load
from src.preprocessing import run_preprocessing_pipeline
from src.time_anomaly import run_time_anomaly_detection

BASE = os.path.dirname(__file__)
RAW_PATH = os.path.join(BASE, "data", "raw", "upi_transactions.csv")
CLEAN_PATH = os.path.join(BASE, "data", "processed", "upi_transactions_clean.csv")
FEATURES_PATH = os.path.join(BASE, "data", "processed", "upi_transactions_features.csv")
ANOMALIES_PATH = os.path.join(BASE, "data", "processed", "upi_transactions_anomalies.csv")
TIME_PATH = os.path.join(BASE, "data", "processed", "upi_transactions_time.csv")
SCORED_PATH = os.path.join(BASE, "data", "processed", "upi_transactions_scored.csv")
MODEL_PATH = os.path.join(BASE, "models", "isolation_forest.pkl")
FIGURES_DIR = os.path.join(BASE, "reports", "figures")


def _step(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main(skip_mysql: bool = False) -> None:
    start = time.time()

    _step("STEP 1/8: Generating synthetic UPI transaction data")
    df = generate_dataset()
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    print(f"Generated {len(df)} transactions -> {RAW_PATH}")

    _step("STEP 2/8: Preprocessing")
    run_preprocessing_pipeline(RAW_PATH, CLEAN_PATH)

    _step("STEP 3/8: Exploratory Data Analysis")
    # EDA reads the feature-engineered file for the hour column, so run
    # feature engineering first, then loop back for EDA - but simplest
    # here is to generate hour-independent EDA off the clean file when
    # `hour` isn't required. Since our EDA needs `hour` and
    # `is_fraud_simulated`, we run feature engineering first, then EDA.

    _step("STEP 4/8: Feature Engineering")
    run_feature_engineering(CLEAN_PATH, FEATURES_PATH)

    print("Running EDA now that engineered features are available...")
    run_eda(FEATURES_PATH, FIGURES_DIR)

    _step("STEP 5/8: IQR + Isolation Forest Anomaly Detection")
    run_anomaly_detection(FEATURES_PATH, ANOMALIES_PATH, MODEL_PATH)

    _step("STEP 6/8: Time-Based Anomaly Detection")
    run_time_anomaly_detection(ANOMALIES_PATH, TIME_PATH)

    _step("STEP 7/8: Fraud Risk Scoring")
    run_fraud_scoring(TIME_PATH, SCORED_PATH)

    _step("STEP 8/8: Model Evaluation")
    run_evaluation(SCORED_PATH, FIGURES_DIR)

    if not skip_mysql:
        _step("BONUS: Loading results into MySQL")
        run_load(SCORED_PATH)
    else:
        print("\nSkipping MySQL load (--skip-mysql passed).")

    elapsed = time.time() - start
    print()
    print("=" * 70)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print("Next steps:")
    print("  uvicorn api.main:app --reload       # start the API")
    print("  streamlit run dashboard/app.py       # start the dashboard")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full UPI fraud detection pipeline.")
    parser.add_argument("--skip-mysql", action="store_true",
                         help="Run the pipeline without loading results into MySQL")
    args = parser.parse_args()
    main(skip_mysql=args.skip_mysql)
