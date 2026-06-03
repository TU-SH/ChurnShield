"""Feature engineering pipeline for ChurnShield.

All transformations are deterministic and unit-tested.
Feature version: v1.0
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AU_STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]
STATE_MAP = {s: i for i, s in enumerate(AU_STATES)}

# Columns fed to the XGBoost model — order matters for SHAP
FEATURE_COLUMNS = [
    "account_length_days",
    "international_plan",
    "voicemail_plan",
    "voicemail_messages",
    "day_mins",
    "day_calls",
    "day_charge_aud",
    "evening_mins",
    "evening_calls",
    "evening_charge_aud",
    "night_mins",
    "night_calls",
    "night_charge_aud",
    "intl_mins",
    "intl_calls",
    "intl_charge_aud",
    "customer_service_calls",
    # engineered
    "total_charge_aud",
    "total_calls",
    "total_mins",
    "avg_charge_per_call",
    "charge_per_min",
    "cs_call_ratio",
    "has_both_plans",
    "high_day_usage",
    "state_encoded",
]


def engineer_features(df: pd.DataFrame, p75_day_mins: float | None = None) -> pd.DataFrame:
    """
    Compute all derived features from a raw customers DataFrame.
    Returns a new DataFrame with customer_id + all engineered columns.

    Args:
        df: Raw customer records (may include 'churned' column).
        p75_day_mins: 75th percentile of day_mins from training set.
                      Pass this at inference time to avoid data leakage.
    """
    feat = df[["customer_id"]].copy()

    # ── Charge aggregations ──────────────────────────────────────────────────
    feat["total_charge_aud"] = (
        df["day_charge_aud"]
        + df["evening_charge_aud"]
        + df["night_charge_aud"]
        + df["intl_charge_aud"]
    ).round(2)

    feat["total_calls"] = (
        df["day_calls"] + df["evening_calls"] + df["night_calls"] + df["intl_calls"]
    )

    feat["total_mins"] = (
        df["day_mins"] + df["evening_mins"] + df["night_mins"] + df["intl_mins"]
    ).round(2)

    # ── Rate features ────────────────────────────────────────────────────────
    feat["avg_charge_per_call"] = np.where(
        feat["total_calls"] > 0,
        (feat["total_charge_aud"] / feat["total_calls"]).round(4),
        0,
    )

    feat["charge_per_min"] = np.where(
        feat["total_mins"] > 0,
        (feat["total_charge_aud"] / feat["total_mins"]).round(4),
        0,
    )

    # ── Customer service signal (strong AU telco churn predictor) ────────────
    feat["cs_call_ratio"] = np.where(
        df["account_length_days"] > 0,
        (df["customer_service_calls"] / df["account_length_days"]).round(6),
        0,
    )

    # ── Plan interaction ─────────────────────────────────────────────────────
    feat["has_both_plans"] = (
        df["international_plan"].astype(bool) & df["voicemail_plan"].astype(bool)
    ).astype(int)

    # ── High day usage flag ──────────────────────────────────────────────────
    if p75_day_mins is None:
        p75_day_mins = df["day_mins"].quantile(0.75)
    feat["high_day_usage"] = (df["day_mins"] > p75_day_mins).astype(int)

    # ── AU state encoding ────────────────────────────────────────────────────
    feat["state_encoded"] = df["state"].map(STATE_MAP).fillna(-1).astype(int)

    feat["feature_version"] = "v1.0"

    # Merge raw columns needed by the model
    raw_cols = [
        "account_length_days", "international_plan", "voicemail_plan",
        "voicemail_messages", "day_mins", "day_calls", "day_charge_aud",
        "evening_mins", "evening_calls", "evening_charge_aud",
        "night_mins", "night_calls", "night_charge_aud",
        "intl_mins", "intl_calls", "intl_charge_aud",
        "customer_service_calls",
    ]
    feat = feat.merge(df[["customer_id"] + raw_cols], on="customer_id")

    logger.debug(f"Engineered features for {len(feat)} customers")
    return feat


def engineer_features_single(df: pd.DataFrame, p75_day_mins: float = 220.5) -> pd.DataFrame:
    """Wrapper for single-row inference — uses training-set p75."""
    return engineer_features(df, p75_day_mins=p75_day_mins)


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the FEATURE_COLUMNS subset, in the correct order."""
    return df[FEATURE_COLUMNS].astype(float)
