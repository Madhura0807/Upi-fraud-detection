"""
Exploratory Data Analysis
============================

Generates a small, purposeful set of plots (not a dump of every
possible chart) saved to reports/figures/, each answering a specific
question about the data:

  1. amount_distribution.png     - is amount skewed? (it is - typical
                                     for financial data, justifies log
                                     scale / IQR approach later)
  2. transactions_by_hour.png    - when do transactions happen, and do
                                     fraud-labeled ones cluster at odd
                                     hours?
  3. merchant_category.png       - which categories see the most volume
  4. fraud_vs_normal_amount.png  - do fraud-labeled transactions really
                                     look different in amount?
  5. transactions_over_time.png  - daily volume trend, sanity check for
                                     the 90-day synthetic window
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def plot_amount_distribution(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["amount"], bins=60, ax=axes[0])
    axes[0].set_title("Transaction Amount Distribution")
    axes[0].set_xlabel("Amount (INR)")

    sns.histplot(df["amount"], bins=60, log_scale=(True, False), ax=axes[1])
    axes[1].set_title("Transaction Amount Distribution (log scale)")
    axes[1].set_xlabel("Amount (INR, log scale)")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "amount_distribution.png"), dpi=120)
    plt.close(fig)


def plot_transactions_by_hour(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    hourly = df.groupby(["hour", "is_fraud_simulated"]).size().unstack(fill_value=0)
    hourly.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Transactions by Hour of Day (normal vs simulated-fraud)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Transaction Count")
    ax.legend(["Normal", "Simulated Fraud"])
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transactions_by_hour.png"), dpi=120)
    plt.close(fig)


def plot_merchant_category(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    order = df["merchant_category"].value_counts().index
    sns.countplot(data=df, y="merchant_category", order=order, ax=ax)
    ax.set_title("Transaction Volume by Merchant Category")
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "merchant_category.png"), dpi=120)
    plt.close(fig)


def plot_fraud_vs_normal_amount(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="is_fraud_simulated", y="amount", ax=ax)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Normal", "Simulated Fraud"])
    ax.set_title("Transaction Amount: Normal vs Simulated Fraud (log scale)")
    ax.set_xlabel("")
    ax.set_ylabel("Amount (INR, log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fraud_vs_normal_amount.png"), dpi=120)
    plt.close(fig)


def plot_transactions_over_time(df: pd.DataFrame, out_dir: str) -> None:
    daily = df.set_index("timestamp").resample("D").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    daily.plot(ax=ax)
    ax.set_title("Daily Transaction Volume")
    ax.set_xlabel("Date")
    ax.set_ylabel("Transaction Count")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "transactions_over_time.png"), dpi=120)
    plt.close(fig)


def run_eda(input_path: str, out_dir: str) -> None:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    os.makedirs(out_dir, exist_ok=True)

    plot_amount_distribution(df, out_dir)
    plot_transactions_by_hour(df, out_dir)
    plot_merchant_category(df, out_dir)
    plot_fraud_vs_normal_amount(df, out_dir)
    plot_transactions_over_time(df, out_dir)

    print(f"Saved 5 EDA figures -> {out_dir}")


if __name__ == "__main__":
    base = os.path.dirname(__file__)
    input_path = os.path.join(base, "..", "data", "processed", "upi_transactions_features.csv")
    out_dir = os.path.join(base, "..", "reports", "figures")
    run_eda(input_path, out_dir)
