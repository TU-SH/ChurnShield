-- ChurnShield Database Schema
-- Run: psql -U postgres -d churnshield -f database/schema.sql

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS ml;

-- ─── RAW SCHEMA ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS raw.customers (
    customer_id             VARCHAR(20)  PRIMARY KEY,
    state                   VARCHAR(3)   NOT NULL CHECK (state IN ('NSW','VIC','QLD','WA','SA','TAS','ACT','NT')),
    account_length_days     INT          NOT NULL CHECK (account_length_days >= 0),
    area_code               VARCHAR(5),
    international_plan      BOOLEAN      NOT NULL DEFAULT FALSE,
    voicemail_plan          BOOLEAN      NOT NULL DEFAULT FALSE,
    voicemail_messages      INT          NOT NULL DEFAULT 0,
    day_mins                NUMERIC(8,2) NOT NULL DEFAULT 0,
    day_calls               INT          NOT NULL DEFAULT 0,
    day_charge_aud          NUMERIC(8,2) NOT NULL DEFAULT 0,
    evening_mins            NUMERIC(8,2) NOT NULL DEFAULT 0,
    evening_calls           INT          NOT NULL DEFAULT 0,
    evening_charge_aud      NUMERIC(8,2) NOT NULL DEFAULT 0,
    night_mins              NUMERIC(8,2) NOT NULL DEFAULT 0,
    night_calls             INT          NOT NULL DEFAULT 0,
    night_charge_aud        NUMERIC(8,2) NOT NULL DEFAULT 0,
    intl_mins               NUMERIC(8,2) NOT NULL DEFAULT 0,
    intl_calls              INT          NOT NULL DEFAULT 0,
    intl_charge_aud         NUMERIC(8,2) NOT NULL DEFAULT 0,
    customer_service_calls  INT          NOT NULL DEFAULT 0,
    churned                 BOOLEAN      NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─── FEATURES SCHEMA ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS features.customer_features (
    customer_id             VARCHAR(20)  PRIMARY KEY REFERENCES raw.customers(customer_id),
    total_charge_aud        NUMERIC(10,2),
    total_calls             INT,
    total_mins              NUMERIC(10,2),
    avg_charge_per_call     NUMERIC(8,4),
    charge_per_min          NUMERIC(8,4),
    cs_call_ratio           NUMERIC(8,4),
    has_both_plans          BOOLEAN,
    high_day_usage          BOOLEAN,
    state_encoded           INT,
    feature_version         VARCHAR(10)  NOT NULL DEFAULT 'v1.0',
    computed_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ─── ML SCHEMA ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml.model_registry (
    model_id        SERIAL       PRIMARY KEY,
    mlflow_run_id   VARCHAR(64)  UNIQUE NOT NULL,
    model_name      VARCHAR(100) NOT NULL,
    version         VARCHAR(20),
    stage           VARCHAR(20)  NOT NULL DEFAULT 'Staging',
    auc_roc         NUMERIC(6,4),
    precision_score NUMERIC(6,4),
    recall_score    NUMERIC(6,4),
    f1_score        NUMERIC(6,4),
    feature_version VARCHAR(10),
    trained_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    promoted_at     TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ml.predictions (
    prediction_id     BIGSERIAL    PRIMARY KEY,
    customer_id       VARCHAR(20)  REFERENCES raw.customers(customer_id),
    model_id          INT          REFERENCES ml.model_registry(model_id),
    churn_probability NUMERIC(6,4) NOT NULL,
    churn_predicted   BOOLEAN      NOT NULL,
    risk_segment      VARCHAR(10)  NOT NULL CHECK (risk_segment IN ('LOW','MEDIUM','HIGH')),
    shap_values       JSONB,
    predicted_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    request_source    VARCHAR(50)  DEFAULT 'api'
);

CREATE INDEX IF NOT EXISTS idx_predictions_customer  ON ml.predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_risk      ON ml.predictions(risk_segment);
CREATE INDEX IF NOT EXISTS idx_predictions_date      ON ml.predictions(predicted_at DESC);
CREATE INDEX IF NOT EXISTS idx_customers_state       ON raw.customers(state);
CREATE INDEX IF NOT EXISTS idx_customers_churned     ON raw.customers(churned);
