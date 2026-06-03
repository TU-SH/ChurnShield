"""Pydantic request / response models."""
from __future__ import annotations

from typing import List, Tuple

from pydantic import BaseModel, Field, field_validator

AU_STATES = ["NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT"]


class CustomerInput(BaseModel):
    customer_id:            str   = Field(..., example="AU-000001")
    state:                  str   = Field(..., example="NSW")
    account_length_days:    int   = Field(..., ge=0, le=3650, example=365)
    international_plan:     bool  = Field(..., example=False)
    voicemail_plan:         bool  = Field(..., example=True)
    voicemail_messages:     int   = Field(0,   ge=0, example=15)
    day_mins:               float = Field(..., ge=0, example=180.5)
    day_calls:              int   = Field(..., ge=0, example=90)
    day_charge_aud:         float = Field(..., ge=0, example=30.69)
    evening_mins:           float = Field(..., ge=0, example=200.1)
    evening_calls:          int   = Field(..., ge=0, example=90)
    evening_charge_aud:     float = Field(..., ge=0, example=17.01)
    night_mins:             float = Field(..., ge=0, example=201.4)
    night_calls:            int   = Field(..., ge=0, example=90)
    night_charge_aud:       float = Field(..., ge=0, example=9.06)
    intl_mins:              float = Field(..., ge=0, example=2.7)
    intl_calls:             int   = Field(..., ge=0, example=3)
    intl_charge_aud:        float = Field(..., ge=0, example=0.73)
    customer_service_calls: int   = Field(..., ge=0, example=2)

    @field_validator("state")
    @classmethod
    def validate_state(cls, v: str) -> str:
        if v not in AU_STATES:
            raise ValueError(f"state must be one of {AU_STATES}")
        return v


class PredictionResponse(BaseModel):
    customer_id:        str
    churn_probability:  float
    churn_predicted:    bool
    risk_segment:       str
    top_risk_factors:   List[Tuple[str, float]]
    latency_ms:         float


class BatchInput(BaseModel):
    customers: List[CustomerInput]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total:        int
    failed:       int = 0


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str
