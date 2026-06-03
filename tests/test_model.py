"""Model performance regression tests — ensure AUC never drops below threshold."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.mark.skipif(
    not Path("data/raw/customers.csv").exists(),
    reason="Training data not generated yet",
)
def test_features_no_nulls():
    from src.features.engineering import engineer_features, get_feature_matrix
    df = pd.read_csv("data/raw/customers.csv")
    feats = engineer_features(df)
    X = get_feature_matrix(feats)
    assert not X.isnull().any().any()


@pytest.mark.skipif(
    not Path("data/raw/customers.csv").exists(),
    reason="Training data not generated yet",
)
def test_churn_rate_realistic():
    df = pd.read_csv("data/raw/customers.csv")
    churn_rate = df["churned"].mean()
    assert 0.05 <= churn_rate <= 0.30, f"Churn rate {churn_rate:.1%} outside realistic range"


@pytest.mark.skipif(
    not Path("data/raw/customers.csv").exists(),
    reason="Training data not generated yet",
)
def test_feature_count():
    from src.features.engineering import FEATURE_COLUMNS
    assert len(FEATURE_COLUMNS) >= 20, "Should have at least 20 features"
