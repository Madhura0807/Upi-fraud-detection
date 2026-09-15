-- ============================================================
-- UPI Fraud Detection - Analysis Queries
-- Run against: upi_fraud_db
-- Each query is written to be explainable in a few sentences.
-- ============================================================

USE upi_fraud_db;

-- ------------------------------------------------------------
-- 1. High-risk transactions (most recent first)
--    Joins the two tables to show the transaction alongside why
--    it was flagged.
-- ------------------------------------------------------------
SELECT
    t.transaction_id,
    t.timestamp,
    t.sender_id,
    t.receiver_id,
    t.amount,
    f.risk_score,
    f.risk_level,
    f.reasons
FROM transactions t
JOIN fraud_alerts f ON t.transaction_id = f.transaction_id
WHERE f.risk_level = 'HIGH'
ORDER BY t.timestamp DESC
LIMIT 50;

-- ------------------------------------------------------------
-- 2. Transaction count per user, ranked highest first
--    Useful for spotting unusually active accounts.
-- ------------------------------------------------------------
SELECT
    sender_id,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_amount_sent
FROM transactions
GROUP BY sender_id
ORDER BY transaction_count DESC
LIMIT 20;

-- ------------------------------------------------------------
-- 3. Average transaction amount by merchant category
-- ------------------------------------------------------------
SELECT
    merchant_category,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM transactions
GROUP BY merchant_category
ORDER BY avg_amount DESC;

-- ------------------------------------------------------------
-- 4. Suspicious (MEDIUM/HIGH) transactions by hour of day
--    Reveals whether fraud alerts cluster at particular hours.
-- ------------------------------------------------------------
SELECT
    HOUR(t.timestamp) AS hour_of_day,
    COUNT(*) AS suspicious_count
FROM transactions t
JOIN fraud_alerts f ON t.transaction_id = f.transaction_id
WHERE f.risk_level IN ('MEDIUM', 'HIGH')
GROUP BY HOUR(t.timestamp)
ORDER BY hour_of_day;

-- ------------------------------------------------------------
-- 5. Users with unusually high transaction frequency
--    Uses a subquery to compare each user's count against the
--    overall average + 2 standard deviations - a SQL-native
--    version of the same "how far from normal" idea used in the
--    Python IQR/z-score logic.
-- ------------------------------------------------------------
WITH user_counts AS (
    SELECT sender_id, COUNT(*) AS txn_count
    FROM transactions
    GROUP BY sender_id
),
stats AS (
    SELECT AVG(txn_count) AS mean_count, STDDEV(txn_count) AS std_count
    FROM user_counts
)
SELECT uc.sender_id, uc.txn_count
FROM user_counts uc, stats s
WHERE uc.txn_count > s.mean_count + 2 * s.std_count
ORDER BY uc.txn_count DESC;

-- ------------------------------------------------------------
-- 6. Daily transaction volume (for a time-series chart)
-- ------------------------------------------------------------
SELECT
    DATE(timestamp) AS txn_date,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS total_amount
FROM transactions
GROUP BY DATE(timestamp)
ORDER BY txn_date;

-- ------------------------------------------------------------
-- 7. Risk level distribution with percentage of total
-- ------------------------------------------------------------
SELECT
    risk_level,
    COUNT(*) AS alert_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM fraud_alerts), 2) AS pct_of_total
FROM fraud_alerts
GROUP BY risk_level
ORDER BY FIELD(risk_level, 'LOW', 'MEDIUM', 'HIGH');

-- ------------------------------------------------------------
-- 8. Most common individual fraud reasons
--    reasons is stored as a semicolon-separated string, so we
--    split it out. MySQL 8 supports JSON_TABLE-free splitting via
--    a numbers/recursive CTE approach for a bounded max count.
-- ------------------------------------------------------------
WITH RECURSIVE split_reasons AS (
    SELECT
        alert_id,
        TRIM(SUBSTRING_INDEX(reasons, ';', 1)) AS reason,
        SUBSTRING(reasons, LENGTH(SUBSTRING_INDEX(reasons, ';', 1)) + 2) AS remainder
    FROM fraud_alerts
    WHERE reasons IS NOT NULL AND reasons != '' AND reasons != 'No anomaly signals triggered'

    UNION ALL

    SELECT
        alert_id,
        TRIM(SUBSTRING_INDEX(remainder, ';', 1)),
        SUBSTRING(remainder, LENGTH(SUBSTRING_INDEX(remainder, ';', 1)) + 2)
    FROM split_reasons
    WHERE remainder != ''
)
SELECT reason, COUNT(*) AS occurrences
FROM split_reasons
WHERE reason != ''
GROUP BY reason
ORDER BY occurrences DESC;
