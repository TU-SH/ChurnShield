"""
Training pipeline for ChurnShield.

Usage:
    python -m src.models.train
    make train
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.config import get_settings
from src.database import register_model, upsert_features
from src.features.engineering import (
    FEATURE_COLUMNS,
    engineer_features,
    get_feature_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_training_data() -> pd.DataFrame:
    """Load from CSV (works without a running Postgres during dev)."""
    csv_path = Path("data/raw/customers.csv")
    if csv_path.exists():
        logger.info(f"Loading from {csv_path}")
        return pd.read_csv(csv_path)
    # fallback: load from Postgres
    from src.database import load_raw_customers
    logger.info("Loading from PostgreSQL")
    return load_raw_customers()


def train(push_to_db: bool = False) -> str:
    """
    Full training pipeline:
      1. Load raw data
      2. Engineer features
      3. 5-fold stratified CV
      4. Train final XGBoost + calibrate
      5. Log everything to MLflow
      6. Optionally register model in Postgres
    Returns MLflow run_id.
    """
    settings = get_settings()
    # Use local SQLite — no MLflow server needed
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    # ── Data ────────────────────────────────────────────────────────────────
    raw = load_training_data()
    logger.info(f"Loaded {len(raw)} customers | Churn rate: {raw['churned'].mean():.1%}")

    feats = engineer_features(raw)

    if push_to_db:
        upsert_features(feats)

    X = get_feature_matrix(feats)
    y = raw["churned"].astype(int)

    # Store training p75 for inference
    p75_day_mins = float(raw["day_mins"].quantile(0.75))

    # Class imbalance
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())
    logger.info(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = {
        "n_estimators": 400,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "auc",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        mlflow.log_params(params)
        mlflow.log_param("n_customers", len(raw))
        mlflow.log_param("churn_rate", round(raw["churned"].mean(), 4))
        mlflow.log_param("feature_version", "v1.0")
        mlflow.log_param("p75_day_mins", p75_day_mins)

        # ── Cross-validation ─────────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_model = xgb.XGBClassifier(**params)
        cv_scores = cross_val_score(base_model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

        mlflow.log_metric("cv_auc_mean", round(cv_scores.mean(), 4))
        mlflow.log_metric("cv_auc_std",  round(cv_scores.std(),  4))
        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # ── Final model ──────────────────────────────────────────────────────
        model = xgb.XGBClassifier(**params)
        model.fit(X, y, verbose=False)

        # Platt scaling calibration (cv=5 instead of deprecated 'prefit')
        calibrated = CalibratedClassifierCV(estimator=xgb.XGBClassifier(**params), method="sigmoid", cv=5)
        calibrated.fit(X, y)

        # ── Metrics on full training set (for logging only) ──────────────────
        y_prob = calibrated.predict_proba(X)[:, 1]
        y_pred = (y_prob >= settings.CHURN_THRESHOLD).astype(int)

        auc      = roc_auc_score(y, y_prob)
        prec     = precision_score(y, y_pred, zero_division=0)
        rec      = recall_score(y, y_pred, zero_division=0)
        f1       = f1_score(y, y_pred, zero_division=0)

        mlflow.log_metric("train_auc",       round(auc,  4))
        mlflow.log_metric("train_precision", round(prec, 4))
        mlflow.log_metric("train_recall",    round(rec,  4))
        mlflow.log_metric("train_f1",        round(f1,   4))

        logger.info(f"Train AUC={auc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")

        # ── SHAP ─────────────────────────────────────────────────────────────
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_importance = pd.DataFrame({
            "feature":       FEATURE_COLUMNS,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        mlflow.log_dict(
            shap_importance.to_dict(orient="records"),
            "shap_feature_importance.json",
        )

        top5 = shap_importance.head(5)["feature"].tolist()
        logger.info(f"Top 5 SHAP features: {top5}")

        # ── Log artefacts ────────────────────────────────────────────────────
        # Save calibrated model + metadata as pickle (for offline use)
        artefact = {
            "calibrated_model": calibrated,
            "xgb_model":        model,
            "explainer":        explainer,
            "feature_columns":  FEATURE_COLUMNS,
            "p75_day_mins":     p75_day_mins,
            "threshold":        settings.CHURN_THRESHOLD,
        }
        artefact_path = f"mlruns_local_model_{run_id[:8]}.pkl"
        with open(artefact_path, "wb") as f:
            pickle.dump(artefact, f)
        mlflow.log_artifact(artefact_path)
        # Keep pkl locally — used by predict.py at inference time

        # Log raw XGBoost model (registered model)
        mlflow.xgboost.log_model(
            model,
            artifact_path="xgb_model",
            registered_model_name=settings.MODEL_NAME,
        )

        # ── Register in Postgres ─────────────────────────────────────────────
        metrics = {
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        }
        if push_to_db:
            model_id = register_model(run_id, metrics)
            logger.info(f"Registered model_id={model_id} in Postgres (Staging)")

        logger.info(f"Training complete. Run ID: {run_id}")
        return run_id


if __name__ == "__main__":
    import sys
    push = "--push-db" in sys.argv
    run_id = train(push_to_db=push)
    print(f"\nRun ID: {run_id}")
    print("View in MLflow: http://localhost:5000")
