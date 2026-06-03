"""Unit tests for feature engineering."""
import pandas as pd
import pytest
from src.features.engineering import engineer_features, FEATURE_COLUMNS

BASE = {
    "customer_id":            "AU-TEST-001",
    "state":                  "NSW",
    "account_length_days":    365,
    "area_code":              "02",
    "international_plan":     True,
    "voicemail_plan":         False,
    "voicemail_messages":     0,
    "day_mins":               200.0,
    "day_calls":              100,
    "day_charge_aud":         34.0,
    "evening_mins":           150.0,
    "evening_calls":          80,
    "evening_charge_aud":     12.75,
    "night_mins":             120.0,
    "night_calls":            60,
    "night_charge_aud":       5.40,
    "intl_mins":              10.0,
    "intl_calls":             5,
    "intl_charge_aud":        2.70,
    "customer_service_calls": 3,
    "churned":                False,
}


def make_df(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


def test_all_feature_columns_present():
    feats = engineer_features(make_df())
    for col in FEATURE_COLUMNS:
        assert col in feats.columns, f"Missing feature: {col}"


def test_total_charge_correct():
    feats = engineer_features(make_df())
    expected = 34.0 + 12.75 + 5.40 + 2.70
    assert abs(feats["total_charge_aud"].iloc[0] - expected) < 0.01


def test_total_calls_correct():
    feats = engineer_features(make_df())
    expected = 100 + 80 + 60 + 5
    assert feats["total_calls"].iloc[0] == expected


def test_cs_ratio_zero_for_zero_account_length():
    feats = engineer_features(make_df(account_length_days=0))
    assert feats["cs_call_ratio"].iloc[0] == 0.0


def test_cs_ratio_correct():
    feats = engineer_features(make_df(customer_service_calls=3, account_length_days=100))
    assert abs(feats["cs_call_ratio"].iloc[0] - 0.03) < 1e-6


def test_has_both_plans_true():
    feats = engineer_features(make_df(international_plan=True, voicemail_plan=True))
    assert feats["has_both_plans"].iloc[0] == 1


def test_has_both_plans_false():
    feats = engineer_features(make_df(international_plan=True, voicemail_plan=False))
    assert feats["has_both_plans"].iloc[0] == 0


def test_state_encoding_nsw_is_zero():
    feats = engineer_features(make_df(state="NSW"))
    assert feats["state_encoded"].iloc[0] == 0


def test_state_encoding_vic_is_one():
    feats = engineer_features(make_df(state="VIC"))
    assert feats["state_encoded"].iloc[0] == 1


def test_unknown_state_is_minus_one():
    feats = engineer_features(make_df(state="XX"))
    assert feats["state_encoded"].iloc[0] == -1


def test_avg_charge_per_call_nonzero():
    feats = engineer_features(make_df())
    assert feats["avg_charge_per_call"].iloc[0] > 0


def test_charge_per_min_zero_when_no_usage():
    feats = engineer_features(make_df(
        day_mins=0, evening_mins=0, night_mins=0, intl_mins=0,
        day_charge_aud=0, evening_charge_aud=0, night_charge_aud=0, intl_charge_aud=0,
    ))
    assert feats["charge_per_min"].iloc[0] == 0.0


def test_feature_version_is_set():
    feats = engineer_features(make_df())
    assert feats["feature_version"].iloc[0] == "v1.0"


def test_no_null_values_in_feature_matrix():
    from src.features.engineering import get_feature_matrix
    feats = engineer_features(make_df())
    X = get_feature_matrix(feats)
    assert not X.isnull().any().any(), "Null values found in feature matrix"
