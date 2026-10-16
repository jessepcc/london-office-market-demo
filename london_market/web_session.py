"""Private browser-session data and explicit visitor-supplied credentials."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import SecretStr

from .config import Settings
from .domain import Limits, Mode
from .fixtures import load_fixtures
from .store import Store


class SessionData:
    def __init__(self) -> None:
        self.directory = TemporaryDirectory(prefix="london-market-")
        self.database_path = Path(self.directory.name) / "market.sqlite3"
        self.demo_initialized = False

    def initialize_demo(self) -> None:
        if not self.demo_initialized:
            with Store(self.database_path) as store:
                if not store.counts("fixture")["documents"]:
                    load_fixtures(store)
            self.demo_initialized = True

    def close(self) -> None:
        self.directory.cleanup()


def session_settings(data: SessionData, *, mode: Mode, openrouter_key: str = "", tavily_key: str = "", main_model: str | None = None, router_model: str | None = None) -> Settings:
    return Settings(
        mode=mode,
        database_path=data.database_path,
        openrouter_api_key=SecretStr(openrouter_key.strip()) if openrouter_key.strip() else None,
        tavily_api_key=SecretStr(tavily_key.strip()) if tavily_key.strip() else None,
        main_model=main_model if main_model is not None else Settings.model_fields["main_model"].default,
        router_model=router_model if router_model is not None else Settings.model_fields["router_model"].default,
        limits=Limits(max_steps=96, max_seconds=180, max_tokens=250000, max_spend_usd=0.25),
    )
