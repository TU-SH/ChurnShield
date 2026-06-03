"""Database engine, session factory, and helper functions."""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config import get_settings

logger = logging.getLogger(__name__)


def get_engine():
    settings = get_settings()
    return create_engine(settings.DATABASE_URL, pool_pre_ping=True, pool_size=5)


def get_session_factory():
    engine = get_engine()
    return sessionmaker(bind=engine)


@contextmanager
def get_session():
    Session = get_session_factory()
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── Data loading ────────────────────────────────────────────────────────────────

def load_raw_customers() -> pd.DataFrame:
    """Load all customers from raw schema."""
    return pd.read_sql("SELECT * FROM raw.customers", get_engine())


def load_features() -> pd.DataFrame:
    """Load pre-computed features joined with labels."""
    sql = """
        SELECT f.*, c.churned
        FROM features.customer_features f
        JOIN raw.customers c ON f.customer_id = c.customer_id
    """
    return pd.read_sql(sql, get_engine())


def load_predictions_with_context() -> pd.DataFrame:
    """Load latest prediction per customer joined with customer & feature data."""
    sql = """
        SELECT
            p.customer_id,
            p.churn_probability,
            p.churn_predicted,
            p.risk_segment,
            p.shap_values,
            p.predicted_at,
            c.state,
            c.account_length_days,
            c.international_plan,
            c.customer_service_calls,
            c.churned AS actual_churn,
            f.total_charge_aud
        FROM ml.predictions p
        JOIN raw.customers c ON p.customer_id = c.customer_id
        LEFT JOIN features.customer_features f ON p.customer_id = f.customer_id
        WHERE p.prediction_id IN (
            SELECT MAX(prediction_id)
            FROM ml.predictions
            GROUP BY customer_id
        )
        ORDER BY p.churn_probability DESC
    """
    return pd.read_sql(sql, get_engine())


# ── Write helpers ────────────────────────────────────────────────────────────────

def upsert_features(df: pd.DataFrame) -> None:
    """Write engineered features to features.customer_features."""
    engine = get_engine()
    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                INSERT INTO features.customer_features
                    (customer_id, total_charge_aud, total_calls, total_mins,
                     avg_charge_per_call, charge_per_min, cs_call_ratio,
                     has_both_plans, high_day_usage, state_encoded, feature_version)
                VALUES
                    (:customer_id, :total_charge_aud, :total_calls, :total_mins,
                     :avg_charge_per_call, :charge_per_min, :cs_call_ratio,
                     :has_both_plans, :high_day_usage, :state_encoded, :feature_version)
                ON CONFLICT (customer_id) DO UPDATE SET
                    total_charge_aud    = EXCLUDED.total_charge_aud,
                    total_calls         = EXCLUDED.total_calls,
                    total_mins          = EXCLUDED.total_mins,
                    avg_charge_per_call = EXCLUDED.avg_charge_per_call,
                    charge_per_min      = EXCLUDED.charge_per_min,
                    cs_call_ratio       = EXCLUDED.cs_call_ratio,
                    has_both_plans      = EXCLUDED.has_both_plans,
                    high_day_usage      = EXCLUDED.high_day_usage,
                    state_encoded       = EXCLUDED.state_encoded,
                    feature_version     = EXCLUDED.feature_version,
                    computed_at         = NOW()
            """), row.to_dict())
    logger.info(f"Upserted {len(df)} feature rows")


def register_model(run_id: str, metrics: dict, feature_version: str = "v1.0") -> int:
    """Insert model into registry, return model_id."""
    settings = get_settings()
    with get_session() as session:
        result = session.execute(text("""
            INSERT INTO ml.model_registry
                (mlflow_run_id, model_name, stage, auc_roc,
                 precision_score, recall_score, f1_score, feature_version)
            VALUES
                (:run_id, :model_name, 'Staging', :auc_roc,
                 :precision_score, :recall_score, :f1_score, :feature_version)
            ON CONFLICT (mlflow_run_id) DO UPDATE SET stage = 'Staging'
            RETURNING model_id
        """), {
            "run_id": run_id,
            "model_name": settings.MODEL_NAME,
            "auc_roc": metrics.get("cv_auc_mean"),
            "precision_score": metrics.get("precision"),
            "recall_score": metrics.get("recall"),
            "f1_score": metrics.get("f1"),
            "feature_version": feature_version,
        })
        model_id = result.fetchone()[0]
    return model_id


def promote_model(model_id: int) -> None:
    """Promote model to Production, archive previous champion."""
    with get_session() as session:
        session.execute(text("""
            UPDATE ml.model_registry
            SET stage = 'Archived'
            WHERE stage = 'Production'
        """))
        session.execute(text("""
            UPDATE ml.model_registry
            SET stage = 'Production', promoted_at = NOW()
            WHERE model_id = :model_id
        """), {"model_id": model_id})
    logger.info(f"Model {model_id} promoted to Production")


def get_production_model_id() -> Optional[int]:
    """Return the current production model_id, or None."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT model_id FROM ml.model_registry WHERE stage='Production' ORDER BY promoted_at DESC LIMIT 1"
        ))
        row = result.fetchone()
    return row[0] if row else None


def log_prediction(
    customer_id: str,
    prob: float,
    risk: str,
    shap_dict: dict,
    source: str = "api",
) -> None:
    """Log a prediction to ml.predictions."""
    model_id = get_production_model_id()
    with get_session() as session:
        session.execute(text("""
            INSERT INTO ml.predictions
                (customer_id, model_id, churn_probability, churn_predicted,
                 risk_segment, shap_values, request_source)
            VALUES
                (:customer_id, :model_id, :prob, :predicted,
                 :risk, :shap, :source)
        """), {
            "customer_id": customer_id,
            "model_id": model_id,
            "prob": round(prob, 4),
            "predicted": prob >= get_settings().CHURN_THRESHOLD,
            "risk": risk,
            "shap": json.dumps(shap_dict),
            "source": source,
        })
