"""Central configuration via pydantic-settings — reads from .env file."""
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/churnshield"

    # MLflow
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "churn-prediction"

    # Model
    MODEL_NAME: str = "churn-xgboost"
    CHURN_THRESHOLD: float = 0.45
    HIGH_RISK_THRESHOLD: float = 0.70

    # API
    API_TITLE: str = "ChurnShield API"
    API_VERSION: str = "1.0.0"
    API_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
