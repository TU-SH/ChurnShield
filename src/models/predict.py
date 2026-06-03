"""
Inference wrapper.
Loads the champion model artefact (pickle) and exposes predict().
Falls back to a simple XGBoost load if pickle not found.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from src.features.engineering import FEATURE_COLUMNS, get_feature_matrix

logger = logging.getLogger(__name__)

_artefact: Optional[dict] = None


def _find_latest_pickle() -> Optional[Path]:
    pkls = sorted(Path(".").glob("mlruns_local_model_*.pkl"), key=lambda p: p.stat().st_mtime)
    return pkls[-1] if pkls else None


def load_artefact() -> dict:
    global _artefact
    if _artefact is not None:
        return _artefact

    pkl = _find_latest_pickle()
    if pkl:
        logger.info(f"Loading artefact from {pkl}")
        with open(pkl, "rb") as f:
            _artefact = pickle.load(f)
        return _artefact

    raise FileNotFoundError(
        "No model artefact found. Run: python -m src.models.train"
    )


def predict_proba(X: pd.DataFrame) -> np.ndarray:
    art = load_artefact()
    return art["calibrated_model"].predict_proba(X)[:, 1]


def get_shap_values(X: pd.DataFrame) -> np.ndarray:
    art = load_artefact()
    return art["explainer"].shap_values(X)


def predict_single(feature_row: pd.DataFrame) -> dict:
    """
    Full inference for a single customer.
    Returns prob, risk_segment, top 5 SHAP factors.
    """
    from src.config import get_settings
    settings = get_settings()

    X = get_feature_matrix(feature_row)
    prob      = float(predict_proba(X)[0])
    shap_vals = get_shap_values(X)[0]
    shap_dict = {k: round(float(v), 4) for k, v in zip(FEATURE_COLUMNS, shap_vals)}

    risk = (
        "HIGH"   if prob >= settings.HIGH_RISK_THRESHOLD else
        "MEDIUM" if prob >= settings.CHURN_THRESHOLD     else
        "LOW"
    )

    top5 = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "churn_probability": round(prob, 4),
        "churn_predicted":   prob >= settings.CHURN_THRESHOLD,
        "risk_segment":      risk,
        "shap_values":       shap_dict,
        "top_risk_factors":  top5,
    }
