"""Visitor keys and evidence remain inside one temporary browser session."""
from pathlib import Path
import subprocess
import sys

import pytest
from streamlit.testing.v1 import AppTest

from london_market.config import Settings
from london_market.domain import Task, RunRecord
from london_market.store import Store
from london_market.web_session import SessionData, session_settings

APP = Path(__file__).parents[1] / "app.py"


def test_sessions_isolate_storage_credentials_and_idempotent_fixture_start(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-router-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "host-search-secret")
    monkeypatch.setenv("OPENROUTER_MAIN_MODEL", "host/model-must-not-be-used")
    monkeypatch.setenv("LONDON_MARKET_DB", "/tmp/host-database-must-not-be-opened.sqlite3")
    first, second = SessionData(), SessionData()
    try:
        first.initialize_demo()
        with Store(first.database_path) as store:
            run = RunRecord(task=Task(objective="First visitor only", entry_skill="research_market"), main_model="fixture", router_model="fixture")
            store.save_run(run)
            counts = store.counts("fixture")
        first.initialize_demo()
        with Store(first.database_path) as store:
            assert store.counts("fixture") == counts
        with Store(second.database_path) as store:
            assert store.counts("fixture")["documents"] == 0
            assert store.list_runs("fixture") == []
        own = session_settings(first, mode="live", openrouter_key="visitor-router", tavily_key="visitor-search")
        empty = session_settings(second, mode="live")
        assert own.openrouter_api_key.get_secret_value() == "visitor-router"
        assert empty.openrouter_api_key is None and empty.tavily_api_key is None
        assert empty.main_model == Settings.model_fields["main_model"].default
        assert first.database_path != second.database_path
        assert "visitor-router" not in own.model_dump_json() and "visitor-search" not in repr(own)
        assert own.limits.max_spend_usd == 0.25
    finally:
        first.close()
        second.close()


def test_real_app_initializes_demo_isolates_sessions_and_clears_keys_and_data(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-router-secret")
    monkeypatch.setenv("TAVILY_API_KEY", "host-search-secret")
    monkeypatch.setenv("LONDON_MARKET_REQUIRE_PASSWORD", "true")
    first = AppTest.from_file(str(APP), default_timeout=10).run()
    second = AppTest.from_file(str(APP), default_timeout=10).run()
    assert not first.exception and not second.exception
    old_data = first.session_state["session-data"]
    other_data = second.session_state["session-data"]
    with Store(old_data.database_path) as store:
        assert store.counts("fixture")["observations"] > 0
    first.chat_input(key="question-fixture").set_value("What is City vacancy?").run()
    assert len(first.chat_message) == 2 and len(second.chat_message) == 0
    first.selectbox(key="data-mode").select("live").run()
    assert first.chat_input(key="question-live").disabled
    assert first.text_input(key="openrouter-key").value == ""
    first.text_input(key="openrouter-key").set_value("visitor-router").run()
    assert first.chat_input(key="question-live").disabled
    first.text_input(key="tavily-key").set_value("visitor-search").run()
    assert not first.chat_input(key="question-live").disabled
    first.radio(key="view").set_value("Controls").run()
    assert not first.button(key="collect").disabled and not first.button(key="briefing").disabled
    second.selectbox(key="data-mode").select("live").run()
    assert second.text_input(key="openrouter-key").value == ""
    assert second.text_input(key="tavily-key").value == ""
    first.button(key="clear-session").click().run()
    assert not first.exception
    assert not old_data.database_path.parent.exists()
    assert first.session_state["session-data"].database_path != old_data.database_path
    assert other_data.database_path.parent.exists()
    assert len(first.chat_message) == 0
    first.selectbox(key="data-mode").select("live").run()
    assert first.text_input(key="openrouter-key").value == ""
    assert first.text_input(key="tavily-key").value == ""
    assert first.chat_input(key="question-live").disabled


def test_cli_rejects_removed_scheduler_and_poll_controls():
    for args in (("scheduler",), ("status", "--poll-seconds", "10"), ("status", "--once")):
        result = subprocess.run([sys.executable, "-m", "london_market", *args], cwd=APP.parent, text=True, capture_output=True)
        assert result.returncode == 2
        assert "invalid choice" in result.stderr or "unrecognized arguments" in result.stderr


def test_web_rejects_models_without_supported_billing_and_router_contract():
    data = SessionData()
    try:
        with pytest.raises(ValueError, match="research model"):
            session_settings(data, mode="live", main_model="unknown/unpriced-model")
        with pytest.raises(ValueError, match="router"):
            session_settings(data, mode="live", router_model="unknown/router")
    finally:
        data.close()


def test_opening_saved_briefing_has_no_widget_state_warning(monkeypatch):
    from streamlit.elements.lib import policies

    monkeypatch.setattr(policies, "_shown_default_value_warning", False)
    app = AppTest.from_file(str(APP), default_timeout=15).run()
    app.radio(key="view").set_value("Controls").run()
    app.button(key="briefing").click().run()
    app.button(key="controls-report-fixture").click().run()
    assert not app.exception
    assert not any("created with a default value" in warning.value for warning in app.warning)
