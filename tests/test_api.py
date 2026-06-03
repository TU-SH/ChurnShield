"""FastAPI endpoint tests using TestClient with mocked model."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

SAMPLE = {
    "customer_id":            "AU-TEST-001",
    "state":                  "VIC",
    "account_length_days":    200,
    "international_plan":     False,
    "voicemail_plan":         True,
    "voicemail_messages":     15,
    "day_mins":               180.0,
    "day_calls":              90,
    "day_charge_aud":         30.6,
    "evening_mins":           200.0,
    "evening_calls":          90,
    "evening_charge_aud":     17.0,
    "night_mins":             200.0,
    "night_calls":            90,
    "night_charge_aud":       9.0,
    "intl_mins":              0.0,
    "intl_calls":             0,
    "intl_charge_aud":        0.0,
    "customer_service_calls": 1,
}


@pytest.fixture
def client():
    """Create test client with mocked model."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.72, 0.28]])

    mock_explainer = MagicMock()
    mock_explainer.shap_values.return_value = np.zeros((1, 26))

    mock_artefact = {
        "calibrated_model": mock_model,
        "xgb_model":        mock_model,
        "explainer":        mock_explainer,
        "feature_columns":  [f"f{i}" for i in range(26)],
        "p75_day_mins":     220.5,
        "threshold":        0.45,
    }

    with patch("src.models.predict._artefact", mock_artefact):
        with patch("src.database.log_prediction", return_value=None):
            from src.api.main import app
            with TestClient(app) as c:
                yield c


def test_health_returns_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_returns_200(client):
    resp = client.post("/predict", json=SAMPLE)
    assert resp.status_code == 200


def test_predict_response_fields(client):
    resp = client.post("/predict", json=SAMPLE)
    data = resp.json()
    assert "churn_probability"  in data
    assert "churn_predicted"    in data
    assert "risk_segment"       in data
    assert "top_risk_factors"   in data
    assert "latency_ms"         in data


def test_predict_probability_in_range(client):
    resp = client.post("/predict", json=SAMPLE)
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_risk_segment_valid(client):
    resp = client.post("/predict", json=SAMPLE)
    risk = resp.json()["risk_segment"]
    assert risk in ("LOW", "MEDIUM", "HIGH")


def test_predict_top_risk_factors_length(client):
    resp = client.post("/predict", json=SAMPLE)
    factors = resp.json()["top_risk_factors"]
    assert len(factors) == 5


def test_invalid_state_returns_422(client):
    bad = {**SAMPLE, "state": "ZZ"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_negative_day_mins_returns_422(client):
    bad = {**SAMPLE, "day_mins": -1.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_batch_predict(client):
    batch = {"customers": [SAMPLE, {**SAMPLE, "customer_id": "AU-TEST-002"}]}
    resp = client.post("/batch", json=batch)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["predictions"]) == 2


def test_batch_size_limit(client):
    big_batch = {"customers": [{**SAMPLE, "customer_id": f"AU-{i:06d}"} for i in range(501)]}
    resp = client.post("/batch", json=big_batch)
    assert resp.status_code == 400
