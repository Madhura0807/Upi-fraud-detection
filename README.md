# UPI Fraud Detection using Anomaly Detection

An end-to-end UPI fraud detection system that identifies suspicious transactions using behavioral analysis, statistical anomaly detection, Isolation Forest, and a transparent risk-scoring framework.

## Overview

The system analyzes transaction history to detect unusual behavior such as high transaction amounts, rapid transaction bursts, new devices, new locations, and new recipients.

Multiple anomaly signals are combined to assign each transaction a risk score and classify it as Low, Medium, or High risk.

## Detection Methods

### 1. High Amount Anomaly
Compares the current transaction amount with the user's historical average.  
A significantly higher amount generates a suspicious-behavior signal.

### 2. IQR Anomaly
Uses Q1, Q3, and IQR to identify statistical outliers.  
Applied to `amount`, `amount_deviation`, and `transactions_last_hour`.

### 3. Isolation Forest
Uses unsupervised machine learning to identify unusual combinations of behavioral features.  
The model is trained without using the simulated fraud labels.

### 4. Time Anomaly
Compares hourly transaction activity with the user's previous six hourly buckets.  
Activity above `rolling_mean + 2.5 × rolling_std` is treated as anomalous.

### 5. Rapid Burst
Counts transactions made by a sender during the previous hour.  
A sudden increase in transaction frequency can indicate suspicious activity.

### 6. New Device
Compares the current device with previously observed devices for the sender.  
A previously unseen device adds a risk signal.

### 7. New Location
Compares the current location with the sender's historical locations.  
A previously unseen location adds a behavioral risk signal.

### 8. New Recipient
Checks whether the sender has previously transacted with the receiver.  
A first-time recipient contributes an additional risk signal.

## Risk Scoring

| Signal | Score |
|---|---:|
| IQR anomaly | +1 |
| Isolation Forest anomaly | +1 |
| Time anomaly | +1 |
| Device changed | +1 |
| Location changed | +1 |
| New recipient | +1 |
| Amount > 3σ above historical average | +1 |

### Risk Levels
0–1  → LOW
2–3  → MEDIUM
4+   → HIGH
