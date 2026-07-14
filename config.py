"""
PrepGalaxy — Configuration Module
==================================

Centralized environment variable loading, validation, and global constants.

This module contains NO business logic. It is responsible exclusively for:
    - Loading environment variables from the process environment / `.env` file
    - Validating that all mandatory configuration is present and well-formed
    - Exposing a single, immutable, typed configuration object to the rest
      of the application via `Config.load()`
    - Defining configuration-level enums shared across the codebase
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Final, List

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment Bootstrap
# ---------------------------------------------------------------------------

BASE_DIR: Final[Path] = Path(__file__).resolve().parent
ENV_FILE: Final[Path] = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=False)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when mandatory configuration is missing or invalid at startup."""

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    def to_logging_level(self) -> int:
        return getattr(logging, self.value)

# ---------------------------------------------------------------------------
# Application-wide Constants
# ---------------------------------------------------------------------------

APP_NAME: Final[str] = "PrepGalaxy"
APP_VERSION: Final[str] = "1.3.0-phase4"

DEFAULT_MONGO_DB_NAME: Final[str] = "prepgalaxy"
DEFAULT_LOG_LEVEL: Final[str] = LogLevel.INFO.value
DEFAULT_RATE_LIMIT: Final[int] = 3

LOG_DIRECTORY: Final[Path] = BASE_DIR / "logs"
LOG_FILE_PATH: Final[Path] = LOG_DIRECTORY / "prepgalaxy.log"
LOG_MAX_BYTES: Final[int] = 10 * 1024 * 1024
LOG_BACKUP_COUNT: Final[int] = 5

DEFAULT_MONGO_MIN_POOL_SIZE: Final[int] = 10
DEFAULT_MONGO_MAX_POOL_SIZE: Final[int] = 100
DEFAULT_MONGO_CONNECT_TIMEOUT_MS: Final[int] = 10_000
DEFAULT_MONGO_SERVER_SELECTION_TIMEOUT_MS: Final[int] = 10_000
DEFAULT_MONGO_MAX_RETRIES: Final[int] = 5
DEFAULT_MONGO_RETRY_BASE_DELAY_SECONDS: Final[float] = 1.0

# ---------------------------------------------------------------------------
# Environment Variable Parsing Helpers
# ---------------------------------------------------------------------------

def _get_required_env(key: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        raise ConfigurationError(f"Missing mandatory environment variable: '{key}'")
    return value.strip()

def _get_optional_env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value.strip() if value and value.strip() else default

def _get_optional_int_env(key: str, default: int) -> int:
    raw_value = os.getenv(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable '{key}' must be an integer, got: '{raw_value}'") from exc

def _get_optional_float_env(key: str, default: float) -> float:
    raw_value = os.getenv(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable '{key}' must be a number, got: '{raw_value}'") from exc

def _get_optional_int_list_env(key: str, default: List[int]) -> List[int]:
    raw_value = os.getenv(key)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return [int(v.strip()) for v in raw_value.split(",") if v.strip()]
    except ValueError as exc:
        raise ConfigurationError(f"Environment variable '{key}' must be a comma-separated list of integers.") from exc

# ---------------------------------------------------------------------------
# Configuration Container
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Config:
    environment: Environment
    log_level: LogLevel
    admin_ids: List[int] = field(default_factory=list)
    rate_limit_mps: int = DEFAULT_RATE_LIMIT
    bot_token: str
    mongo_uri: str
    mongo_db_name: str
    mongo_min_pool_size: int
    mongo_max_pool_size: int
    mongo_connect_timeout_ms: int
    mongo_server_selection_timeout_ms: int
    mongo_max_retries: int
    mongo_retry_base_delay_seconds: float

    @staticmethod
    def load() -> "Config":
        environment_raw = _get_optional_env("ENVIRONMENT", Environment.PRODUCTION.value)
        try:
            environment = Environment(environment_raw.lower())
        except ValueError as exc:
            valid = ", ".join(e.value for e in Environment)
            raise ConfigurationError(f"Invalid ENVIRONMENT value '{environment_raw}'. Must be one of: {valid}") from exc

        log_level_raw = _get_optional_env("LOG_LEVEL", DEFAULT_LOG_LEVEL)
        try:
            log_level = LogLevel(log_level_raw.upper())
        except ValueError as exc:
            valid = ", ".join(l.value for l in LogLevel)
            raise ConfigurationError(f"Invalid LOG_LEVEL value '{log_level_raw}'. Must be one of: {valid}") from exc

        admin_ids = _get_optional_int_list_env("ADMIN_IDS", [])
        rate_limit_mps = _get_optional_int_env("RATE_LIMIT_MESSAGES_PER_SECOND", DEFAULT_RATE_LIMIT)

        bot_token = _get_required_env("BOT_TOKEN")
        if ":" not in bot_token:
            raise ConfigurationError("BOT_TOKEN appears malformed. Expected format '<bot_id>:<secret>'.")

        mongo_uri = _get_required_env("MONGO_URI")
        if not (mongo_uri.startswith("mongodb://") or mongo_uri.startswith("mongodb+srv://")):
            raise ConfigurationError("MONGO_URI must start with 'mongodb://' or 'mongodb+srv://'.")

        mongo_db_name = _get_optional_env("MONGO_DB_NAME", DEFAULT_MONGO_DB_NAME)
        mongo_min_pool_size = _get_optional_int_env("MONGO_MIN_POOL_SIZE", DEFAULT_MONGO_MIN_POOL_SIZE)
        mongo_max_pool_size = _get_optional_int_env("MONGO_MAX_POOL_SIZE", DEFAULT_MONGO_MAX_POOL_SIZE)
        
        if mongo_min_pool_size < 0 or mongo_max_pool_size < 1 or mongo_min_pool_size > mongo_max_pool_size:
            raise ConfigurationError("Invalid MongoDB pool sizing limits.")

        mongo_connect_timeout_ms = _get_optional_int_env("MONGO_CONNECT_TIMEOUT_MS", DEFAULT_MONGO_CONNECT_TIMEOUT_MS)
        mongo_server_selection_timeout_ms = _get_optional_int_env("MONGO_SERVER_SELECTION_TIMEOUT_MS", DEFAULT_MONGO_SERVER_SELECTION_TIMEOUT_MS)
        
        mongo_max_retries = _get_optional_int_env("MONGO_MAX_RETRIES", DEFAULT_MONGO_MAX_RETRIES)
        mongo_retry_base_delay_seconds = _get_optional_float_env("MONGO_RETRY_BASE_DELAY_SECONDS", DEFAULT_MONGO_RETRY_BASE_DELAY_SECONDS)

        return Config(
            environment=environment,
            log_level=log_level,
            admin_ids=admin_ids,
            rate_limit_mps=rate_limit_mps,
            bot_token=bot_token,
            mongo_uri=mongo_uri,
            mongo_db_name=mongo_db_name,
            mongo_min_pool_size=mongo_min_pool_size,
            mongo_max_pool_size=mongo_max_pool_size,
            mongo_connect_timeout_ms=mongo_connect_timeout_ms,
            mongo_server_selection_timeout_ms=mongo_server_selection_timeout_ms,
            mongo_max_retries=mongo_max_retries,
            mongo_retry_base_delay_seconds=mongo_retry_base_delay_seconds,
        )

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION
