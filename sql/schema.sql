-- ============================================================
-- UPI Fraud Detection - Database Schema
-- Database: upi_fraud_db
-- ============================================================
-- Design notes (for interview explanation):
--
-- 1. `transactions` stores every raw transaction exactly as it
--    happened. `transaction_id` is a natural primary key because
--    UPI transactions already have globally unique IDs.
--
-- 2. `fraud_alerts` stores the OUTPUT of the detection pipeline
--    for a transaction: a risk score, risk level, individual
--    anomaly flags, and human-readable reasons. It has its own
--    surrogate key (`alert_id`) because in a real system a
--    transaction could theoretically be re-scored and produce
--    more than one alert record over time (e.g. re-evaluation
--    after new behavioral data arrives).
--
-- 3. `transaction_id` in `fraud_alerts` is a FOREIGN KEY back to
--    `transactions`, enforcing that we never store an alert for
--    a transaction that doesn't exist. ON DELETE CASCADE keeps
--    the two tables consistent.
--
-- 4. Indexes are added on columns that are actually filtered/
--    joined on in the analysis queries and the API (sender_id,
--    timestamp, risk_level) - not on every column, since
--    unnecessary indexes slow down writes for no benefit.
-- ============================================================

CREATE DATABASE IF NOT EXISTS upi_fraud_db CHARACTER SET utf8mb4;
USE upi_fraud_db;

-- Drop in dependency order for clean re-runs during development
DROP TABLE IF EXISTS fraud_alerts;
DROP TABLE IF EXISTS transactions;

-- ------------------------------------------------------------
-- transactions: raw transaction log
-- ------------------------------------------------------------
CREATE TABLE transactions (
    transaction_id      VARCHAR(20)     NOT NULL,
    timestamp            DATETIME        NOT NULL,
    sender_id            VARCHAR(20)     NOT NULL,
    receiver_id          VARCHAR(20)     NOT NULL,
    amount                DECIMAL(12, 2)  NOT NULL,
    merchant_category    VARCHAR(50)     NOT NULL,
    transaction_type     VARCHAR(20)     NOT NULL,   -- e.g. P2P, P2M
    location              VARCHAR(50)     NOT NULL,
    device_type           VARCHAR(30)     NOT NULL,
    upi_channel           VARCHAR(30)     NOT NULL,   -- e.g. GPay, PhonePe, Paytm
    created_at            TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (transaction_id),
    CONSTRAINT chk_amount_positive CHECK (amount > 0)
) ENGINE=InnoDB;

CREATE INDEX idx_transactions_sender ON transactions (sender_id);
CREATE INDEX idx_transactions_timestamp ON transactions (timestamp);
CREATE INDEX idx_transactions_receiver ON transactions (receiver_id);
CREATE INDEX idx_transactions_merchant_category ON transactions (merchant_category);

-- ------------------------------------------------------------
-- fraud_alerts: output of the detection pipeline
-- ------------------------------------------------------------
CREATE TABLE fraud_alerts (
    alert_id                 INT             NOT NULL AUTO_INCREMENT,
    transaction_id            VARCHAR(20)     NOT NULL,
    risk_score                 INT             NOT NULL,
    risk_level                  ENUM('LOW', 'MEDIUM', 'HIGH') NOT NULL,
    iqr_flag                    BOOLEAN         NOT NULL DEFAULT FALSE,
    isolation_forest_flag       BOOLEAN         NOT NULL DEFAULT FALSE,
    time_anomaly_flag           BOOLEAN         NOT NULL DEFAULT FALSE,
    device_change                BOOLEAN         NOT NULL DEFAULT FALSE,
    location_change              BOOLEAN         NOT NULL DEFAULT FALSE,
    reasons                      TEXT,
    created_at                   TIMESTAMP       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (alert_id),
    CONSTRAINT fk_alert_transaction
        FOREIGN KEY (transaction_id)
        REFERENCES transactions (transaction_id)
        ON DELETE CASCADE,
    CONSTRAINT chk_risk_score_range CHECK (risk_score >= 0)
) ENGINE=InnoDB;

CREATE INDEX idx_alerts_transaction_id ON fraud_alerts (transaction_id);
CREATE INDEX idx_alerts_risk_level ON fraud_alerts (risk_level);
CREATE INDEX idx_alerts_created_at ON fraud_alerts (created_at);
