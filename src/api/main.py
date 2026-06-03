"""
ChurnShield FastAPI application.

Endpoints:
  GET  /health          — liveness check
  POST /predict         — single customer churn prediction
  POST /batch           — batch predictions (up to 500)
  GET  /docs            — Swagger UI (automatic)
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.schemas import (
    BatchInput,
    BatchResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)
from src.config import get_settings
from src.database import log_prediction
from src.features.engineering import engineer_features_single, get_feature_matrix
from src.models.predict import load_artefact, predict_single

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    logger.info("Loading model artefact…")
    try:
        load_artefact()
        logger.info("Model loaded OK")
    except FileNotFoundError as e:
        logger.warning(f"Model not found at startup: {e}")
        logger.warning("Run 'python -m src.models.train' first, then restart the API.")
    yield
    logger.info("Shutting down ChurnShield API")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=(
        "End-to-end customer churn prediction for Australian telcos. "
        "Returns churn probability, risk segment, and top SHAP risk factors."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper ───────────────────────────────────────────────────────────────────────

def _run_prediction(customer: CustomerInput) -> PredictionResponse:
    t0 = time.perf_counter()
    row = pd.DataFrame([customer.model_dump()])
    feats = engineer_features_single(row)
    result = predict_single(feats)
    latency = round((time.perf_counter() - t0) * 1000, 1)

    # Fire-and-forget log — don't fail the request if DB is down
    try:
        log_prediction(
            customer.customer_id,
            result["churn_probability"],
            result["risk_segment"],
            result["shap_values"],
            source="api",
        )
    except Exception as e:
        logger.warning(f"Failed to log prediction for {customer.customer_id}: {e}")

    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=result["churn_probability"],
        churn_predicted=result["churn_predicted"],
        risk_segment=result["risk_segment"],
        top_risk_factors=result["top_risk_factors"],
        latency_ms=latency,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — confirms API and model are ready."""
    return HealthResponse(
        status="ok",
        model=settings.MODEL_NAME,
        version=settings.API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerInput):
    """
    Predict churn probability for a single Australian telco customer.

    Returns:
    - **churn_probability**: calibrated probability [0, 1]
    - **risk_segment**: LOW / MEDIUM / HIGH
    - **top_risk_factors**: top 5 SHAP-based drivers
    """
    try:
        return _run_prediction(customer)
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python -m src.models.train' first.",
        )
    except Exception as e:
        logger.exception(f"Prediction error for {customer.customer_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch", response_model=BatchResponse, tags=["Prediction"])
async def batch_predict(batch: BatchInput):
    """
    Predict churn for up to 500 customers in one request.
    Failed individual predictions are skipped and counted in 'failed'.
    """
    if len(batch.customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size limit is 500 customers.")

    results, failed = [], 0
    for customer in batch.customers:
        try:
            results.append(_run_prediction(customer))
        except Exception as e:
            logger.warning(f"Skipping {customer.customer_id}: {e}")
            failed += 1

    return BatchResponse(predictions=results, total=len(results), failed=failed)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
