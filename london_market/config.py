"""Environment configuration parsed once, with credentials excluded from records."""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import ConfigDict, Field, SecretStr, field_validator, model_validator

from .domain import Limits, Mode, Model


class Settings(Model):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    mode: Mode = "fixture"
    database_path: Path = Path("data/market.sqlite3")
    main_model: str = Field(default="deepseek/deepseek-v4.1-flash", min_length=1)
    main_provider_ignore: str = Field(default="Wafer", max_length=300)
    router_model: str = Field(default="cloudflare/clef-flash", min_length=1)
    openrouter_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    tavily_api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    limits: Limits = Field(default_factory=Limits)
    timezone: str = "Europe/London"
    cw_permission: bool = False
    history_days: int = Field(default=180, ge=1, le=3650)
    request_timeout_seconds: float = Field(default=60, gt=0, le=120)
    max_search_results: int = Field(default=5, ge=1, le=10)
    main_input_usd_per_token: float | None = Field(default=None, gt=0)
    main_output_usd_per_token: float | None = Field(default=None, ge=0)
    router_input_usd_per_token: float | None = Field(default=None, gt=0)
    router_output_usd_per_token: float | None = Field(default=None, ge=0)
    max_document_chars: int = Field(default=18000, ge=1000, le=100000)

    @field_validator("timezone")
    @classmethod
    def valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("unknown report timezone") from error
        return value

    @model_validator(mode="after")
    def usable_path(self) -> Settings:
        if not str(self.database_path).strip():
            raise ValueError("database path must not be empty")
        return self

    def require_live_credentials(self, *, search: bool = True) -> None:
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if search and not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if missing:
            raise ValueError("Live mode requires " + ", ".join(missing) + "; configure securely in environment settings")


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    values: dict[str, str] = {}
    if env_file is not None and env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError(".env lines must use NAME=value")
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    values.update(os.environ)
    names = {
        "LONDON_MARKET_MODE": "mode", "LONDON_MARKET_DB": "database_path",
        "OPENROUTER_MAIN_MODEL": "main_model", "OPENROUTER_ROUTER_MODEL": "router_model",
        "OPENROUTER_API_KEY": "openrouter_api_key", "TAVILY_API_KEY": "tavily_api_key",
        "LONDON_MARKET_TIMEZONE": "timezone",
        "OPENROUTER_MAIN_INPUT_USD_PER_TOKEN": "main_input_usd_per_token",
        "OPENROUTER_MAIN_OUTPUT_USD_PER_TOKEN": "main_output_usd_per_token",
        "OPENROUTER_ROUTER_INPUT_USD_PER_TOKEN": "router_input_usd_per_token",
        "OPENROUTER_ROUTER_OUTPUT_USD_PER_TOKEN": "router_output_usd_per_token",
        "LONDON_MARKET_CW_PERMISSION": "cw_permission",
        "LONDON_MARKET_HISTORY_DAYS": "history_days", "LONDON_MARKET_REQUEST_TIMEOUT": "request_timeout_seconds",
    }
    limit_names = {
        "LONDON_MARKET_MAX_STEPS": "max_steps", "LONDON_MARKET_MAX_SECONDS": "max_seconds",
        "LONDON_MARKET_MAX_TOKENS": "max_tokens", "LONDON_MARKET_MAX_SPEND_USD": "max_spend_usd",
    }
    payload = {field: values[key] for key, field in names.items() if values.get(key)}
    if "OPENROUTER_MAIN_PROVIDER_IGNORE" in values:
        payload["main_provider_ignore"] = values["OPENROUTER_MAIN_PROVIDER_IGNORE"]
    payload["limits"] = {field: values[key] for key, field in limit_names.items() if values.get(key)}
    return Settings.model_validate(payload)
