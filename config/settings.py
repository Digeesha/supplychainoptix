"""
Configuration management for SupplyChainOptix.

Handles environment-specific settings, secrets management, and feature flags.
Uses pydantic for validation and environment variable loading.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from functools import lru_cache

from pydantic import BaseSettings, Field, validator, PostgresDsn, AnyHttpUrl


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    Loads from .env file and environment variables with validation.
    Environment variables take precedence over .env file values.
    """
    
    # Core Application Settings
    APP_NAME: str = "SupplyChainOptix"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Environment = Field(default=Environment.DEVELOPMENT, env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    LOG_LEVEL: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8501, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # Database Configuration
    DB_HOST: str = Field(default="localhost", env="DB_HOST")
    DB_PORT: int = Field(default=5432, env="DB_PORT")
    DB_NAME: str = Field(default="supplychainoptix", env="DB_NAME")
    DB_USER: str = Field(default="postgres", env="DB_USER")
    DB_PASSWORD: str = Field(default="postgres", env="DB_PASSWORD")
    DB_POOL_SIZE: int = Field(default=20, env="DB_POOL_SIZE")
    DB_MAX_OVERFLOW: int = Field(default=40, env="DB_MAX_OVERFLOW")
    DB_POOL_TIMEOUT: int = Field(default=30, env="DB_POOL_TIMEOUT")
    DB_SSL_MODE: str = Field(default="prefer", env="DB_SSL_MODE")
    
    # Redis Configuration (for caching and session management)
    REDIS_HOST: str = Field(default="localhost", env="REDIS_HOST")
    REDIS_PORT: int = Field(default=6379, env="REDIS_PORT")
    REDIS_DB: int = Field(default=0, env="REDIS_DB")
    REDIS_PASSWORD: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    CACHE_TTL: int = Field(default=3600, env="CACHE_TTL")  # seconds
    
    # ML Model Configuration
    MODEL_PATH: Path = Field(default=Path("models"), env="MODEL_PATH")
    PROPHET_MODEL_PATH: Path = Field(default=Path("models/prophet"), env="PROPHET_MODEL_PATH")
    ISOLATION_FOREST_MODEL_PATH: Path = Field(default=Path("models/isolation_forest"), env="ISOLATION_FOREST_MODEL_PATH")
    MODEL_CACHE_SIZE: int = Field(default=10, env="MODEL_CACHE_SIZE")
    MODEL_RETRAIN_INTERVAL_DAYS: int = Field(default=7, env="MODEL_RETRAIN_INTERVAL_DAYS")
    PREDICTION_HORIZON_DAYS: int = Field(default=14, env="PREDICTION_HORIZON_DAYS")
    MIN_PREDICTION_CONFIDENCE: float = Field(default=0.65, env="MIN_PREDICTION_CONFIDENCE")
    
    # Anomaly Detection Settings
    ANOMALY_CONTAMINATION: float = Field(default=0.1, env="ANOMALY_CONTAMINATION")
    ANOMALY_N_ESTIMATORS: int = Field(default=100, env="ANOMALY_N_ESTIMATORS")
    ANOMALY_MAX_SAMPLES: str = Field(default="auto", env="ANOMALY_MAX_SAMPLES")
    ANOMALY_THRESHOLD: float = Field(default=-0.5, env="ANOMALY_THRESHOLD")
    
    # Data Processing Configuration
    BATCH_SIZE: int = Field(default=1000, env="BATCH_SIZE")
    MAX_CONCURRENT_JOBS: int = Field(default=5, env="MAX_CONCURRENT_JOBS")
    DATA_RETENTION_DAYS: int = Field(default=730, env="DATA_RETENTION_DAYS")  # 2 years
    DATA_REFRESH_INTERVAL_MINUTES: int = Field(default=15, env="DATA_REFRESH_INTERVAL_MINUTES")
    
    # External API Configuration
    GEOPOLITICAL_API_KEY: Optional[str] = Field(default=None, env="GEOPOLITICAL_API_KEY")
    GEOPOLITICAL_API_URL: Optional[AnyHttpUrl] = Field(default=None, env="GEOPOLITICAL_API_URL")
    WEATHER_API_KEY: Optional[str] = Field(default=None, env="WEATHER_API_KEY")
    WEATHER_API_URL: Optional[AnyHttpUrl] = Field(default=None, env="WEATHER_API_URL")
    SHIPPING_API_KEY: Optional[str] = Field(default=None, env="SHIPPING_API_KEY")
    SHIPPING_API_URL: Optional[AnyHttpUrl] = Field(default=None, env="SHIPPING_API_URL")
    API_TIMEOUT_SECONDS: int = Field(default=30, env="API_TIMEOUT_SECONDS")
    API_MAX_RETRIES: int = Field(default=3, env="API_MAX_RETRIES")
    
    # Security Configuration
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    SESSION_LIFETIME_HOURS: int = Field(default=24, env="SESSION_LIFETIME_HOURS")
    ENABLE_CORS: bool = Field(default=True, env="ENABLE_CORS")
    ALLOWED_ORIGINS: List[str] = Field(default=["http://localhost:8501"], env="ALLOWED_ORIGINS")
    
    # Feature Flags
    ENABLE_GEOPOLITICAL_TRACKING: bool = Field(default=True, env="ENABLE_GEOPOLITICAL_TRACKING")
    ENABLE_WEATHER_INTEGRATION: bool = Field(default=True, env="ENABLE_WEATHER_INTEGRATION")
    ENABLE_REAL_TIME_ALERTS: bool = Field(default=True, env="ENABLE_REAL_TIME_ALERTS")
    ENABLE_AUTO_RETRAINING: bool = Field(default=False, env="ENABLE_AUTO_RETRAINING")
    ENABLE_EXPERIMENTAL_FEATURES: bool = Field(default=False, env="ENABLE_EXPERIMENTAL_FEATURES")
    
    # Monitoring and Observability
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    ENABLE_TRACING: bool = Field(default=False, env="ENABLE_TRACING")
    SENTRY_DSN: Optional[str] = Field(default=None, env="SENTRY_DSN")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    # Notification Configuration
    SMTP_HOST: Optional[str] = Field(default=None, env="SMTP_HOST")
    SMTP_PORT: int = Field(default=587, env="SMTP_PORT")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    SMTP_FROM_EMAIL: str = Field(default="noreply@supplychainoptix.com", env="SMTP_FROM_EMAIL")
    SLACK_WEBHOOK_URL: Optional[AnyHttpUrl] = Field(default=None, env="SLACK_WEBHOOK_URL")
    ALERT_EMAIL_RECIPIENTS: List[str] = Field(default=[], env="ALERT_EMAIL_RECIPIENTS")
    
    # Dashboard Configuration
    DEFAULT_DASHBOARD_REFRESH_SECONDS: int = Field(default=300, env="DEFAULT_DASHBOARD_REFRESH_SECONDS")
    MAX_DASHBOARD_ITEMS: int = Field(default=100, env="MAX_DASHBOARD_ITEMS")
    CHART_HEIGHT: int = Field(default=400, env="CHART_HEIGHT")
    ENABLE_DARK_MODE: bool = Field(default=False, env="ENABLE_DARK_MODE")
    
    @validator("MODEL_PATH", "PROPHET_MODEL_PATH", "ISOLATION_FOREST_MODEL_PATH", pre=True)
    def validate_path(cls, v: Any) -> Path:
        """Ensure paths are Path objects and create directories if they don't exist."""
        path = Path(v) if not isinstance(v, Path) else v
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v: str, values: Dict[str, Any]) -> str:
        """Ensure secret key is not default in production."""
        env = values.get("ENVIRONMENT", Environment.DEVELOPMENT)
        if env == Environment.PRODUCTION and v == "dev-secret-key-change-in-production":
            raise ValueError("SECRET_KEY must be changed in production environment")
        return v
    
    @validator("MIN_PREDICTION_CONFIDENCE")
    def validate_confidence(cls, v: float) -> float:
        """Ensure confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("MIN_PREDICTION_CONFIDENCE must be between 0 and 1")
        return v
    
    @validator("ANOMALY_CONTAMINATION")
    def validate_contamination(cls, v: float) -> float:
        """Ensure contamination parameter is valid for isolation forest."""
        if not 0 < v < 0.5:
            raise ValueError("ANOMALY_CONTAMINATION must be between 0 and 0.5")
        return v
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v: Any) -> List[str]:
        """Parse comma-separated origins string into list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("ALERT_EMAIL_RECIPIENTS", pre=True)
    def parse_email_recipients(cls, v: Any) -> List[str]:
        """Parse comma-separated email string into list."""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        return v
    
    @property
    def database_url(self) -> str:
        """Construct PostgreSQL database URL."""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            f"?sslmode={self.DB_SSL_MODE}"
        )
    
    @property
    def async_database_url(self) -> str:
        """Construct async PostgreSQL database URL."""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
    
    @property
    def redis_url(self) -> str:
        """Construct Redis connection URL