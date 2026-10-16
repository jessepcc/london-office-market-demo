"""Drive the real Streamlit app without paid services using AppTest."""

from datetime import datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from london_market.config import Settings
from london_market.domain import ChartPayload, Claim, Evidence, Observation, RunRecord, RunResult, Task, utc_now
from london_market.fixtures import fixture_batches
from london_market.providers import FixtureWebProvider, ProviderError
from london_market.runner import Runner
from london_market.jobs import ManualJobs
from london_market.store import Store

APP = Path(__file__).parents[1] / "app.py"


@pytest.fixture
def app_environment(tmp_path, monkeypatch):
    database = tmp_path / "market.sqlite3"
    monkeypatch.setenv("LONDON_MARKET_DB", str(database))
    monkeypatch.setenv("LONDON_MARKET_MODE", "fixture")
    # Empty process values override a developer's local .env credentials.
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("TAVILY_API_KEY", "")
    return database


def open_app(database):
    from ui_support import app_test
    app = app_test(database)
    assert not app.exception
    return app


def test_empty_states_and_research_without_a_report_preserve_context(app_environment):
    app = open_app(app_environment)
    assert app.radio(key="view").options == ["Weekly briefings", "Chat", "Controls", "Sources and coverage"]
    assert app.selectbox(key="data-mode").options == ["Synthetic demo", "Live with your keys"]
    assert any("SYNTHETIC" in warning.value and "scripted" in warning.value for warning in app.warning)
    assert any("on demand" in caption.value for caption in app.sidebar.caption)
    app.radio(key="view").set_value("Weekly reports").run()
    assert any("No weekly briefings saved" in info.value for info in app.info)
    app.radio(key="view").set_value("Sources").run()
    assert any("No sources saved" in info.value for info in app.info)
    app.radio(key="view").set_value("Research").run()
    app.chat_input(key="question-fixture").set_value("What is City office vacancy?").run()
    app.chat_input(key="question-fixture").set_value("What about Canary Wharf?").run()
    assert not app.exception and len(app.chat_message) == 4
    assert any("No validated Canary Wharf" in markdown.value for markdown in app.markdown)
    with Store(app_environment) as store:
        runs = store.list_runs("fixture")
        assert len(runs) == 2 and store.list_reports("fixture") == []
        assert [message.role for message in runs[0].task.history] == ["user", "assistant"]
        assert runs[0].task.history[0].content == "What is City office vacancy?"
        assert "Observed fact" in runs[0].task.history[1].content
        assert len(store.list_evidence("fixture")) == 1


def test_collect_brief_history_chart_and_source_dates(app_environment):
    app = open_app(app_environment)
    app.radio(key="view").set_value("Controls").run()
    app.button(key="collect").click().run()
    app.button(key="briefing").click().run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert any("Observed fact" in markdown.value for markdown in app.markdown)
    assert any("Landlord analysis" in markdown.value for markdown in app.markdown)
    assert any("Coverage gaps" in markdown.value for markdown in app.markdown)
    assert {item.value for item in app.subheader} >= {"Market evidence", "Landlord analysis", "What to watch", "Market exhibits"}
    app.button(key="briefing").click().run()
    assert any("first saved report was retained" in info.value for info in app.info)
    with Store(app_environment) as store:
        existing = store.list_reports("fixture")[0]
        jobs = ManualJobs(Settings(database_path=app_environment), store)
        jobs.run_manual("briefing", datetime.fromisoformat("2026-10-19T09:00:00+01:00"))
    app.button(key="controls-report-fixture").click().run()
    assert not app.exception
    selector = app.selectbox(key="report-select-fixture")
    assert len(selector.options) == 2
    selector.select(existing["id"]).run()
    assert len(app.get("plotly_chart")) == 2
    app.radio(key="view").set_value("Sources").run()
    app.selectbox(key="source-select-fixture").select(fixture_batches()[0][1][0].id).run()
    assert not app.exception
    assert all("ev_" not in label for label in app.selectbox(key="source-select-fixture").options)
    assert any(item.value == "Reviewed for monitoring" for item in app.caption)
    dates = app.table[0].value.set_index("Field")["Value"].to_dict()
    assert dates["Publication date"] == "2026-09-30"
    assert dates["Event date"] == "2026-03-31"
    assert dates["Retrieved at"].startswith("2026-10-09")
    assert dates["Source geography"] == "City" and dates["Coverage grouping"] == "City"
    observation = next(table.value for table in app.dataframe if "Observation ID" in table.value.columns).iloc[0]
    assert observation["Unit"] == "%" and "Unoccupied" in observation["Definition"]
    assert observation["Reference period"] == "2026-01-01 to 2026-03-31"
    assert observation["Measure"] == "Vacancy rate"
    assert observation["Observation ID"] == fixture_batches()[0][2][0].id


def test_mode_switch_partitions_chat_and_exposes_live_blockers(app_environment):
    app = open_app(app_environment)
    app.chat_input(key="question-fixture").set_value("What is City office vacancy?").run()
    app.selectbox(key="data-mode").select("live").run()
    assert not app.exception and len(app.chat_message) == 0
    assert app.chat_input(key="question-live").disabled
    assert any("Unavailable services" in warning.value for warning in app.warning)
    app.radio(key="view").set_value("Sources").run()
    assert any("reuse permission" in caption.value for caption in app.caption)
    app.radio(key="view").set_value("Controls").run()
    assert app.button(key="collect").disabled and app.button(key="briefing").disabled
    assert any("No manual collection or briefing" in info.value for info in app.info)
    app.radio(key="view").set_value("Weekly reports").run()
    assert any("No weekly briefings" in info.value for info in app.info)
    app.selectbox(key="data-mode").select("fixture").run()
    app.radio(key="view").set_value("Research").run()
    assert len(app.chat_message) == 2
    with Store(app_environment) as store:
        assert len(store.list_runs("fixture")) == 1 and store.list_runs("live") == []


def test_visual_purpose_gap_and_source_failure_remain_visible(app_environment):
    class MissingCityQuarter(FixtureWebProvider):
        def retrieve(self, url, timeout):
            if "city-q2" in url:
                raise ProviderError("synthetic Q2 source unavailable")
            return super().retrieve(url, timeout)
    with Store(app_environment) as store:
        settings = Settings(database_path=app_environment)
        jobs = ManualJobs(settings, store, Runner(settings, store, web=MissingCityQuarter()))
        jobs.run_manual("collection")
        jobs.run_manual("briefing")
    app = open_app(app_environment)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception and len(app.get("plotly_chart")) == 0
    assert any("SYNTHETIC visual-purpose gap: No compatible contiguous history" in item.value for item in app.markdown)
    assert any("synthetic Q2 source unavailable" in item.value for item in app.markdown)
    assert any(item.label == "Evidence limitations" for item in app.expander)
    assert any("Value" in frame.value.columns and len(frame.value) == 1 for frame in app.dataframe)
    app.button(key="open-run-history").click().run()
    assert not app.exception
    assert any("Completed with gaps" in frame.value["Status"].tolist() for frame in app.dataframe if "Status" in frame.value.columns)
    with Store(app_environment) as store:
        collection_id = next(run.id for run in store.list_runs("fixture") if run.task.entry_skill == "collect_market_evidence")
    app.selectbox(key="run-select-fixture").select(collection_id).run()
    assert any("synthetic Q2 source unavailable" in frame.value.to_string() for frame in app.dataframe)


def test_missing_citation_and_insufficient_payload_are_honest(app_environment):
    result = RunResult(status="partial", answer="Synthetic unsupported citation fixture", reason="Missing source fixture", evidence_ids=["missing-source"], claims=[Claim(kind="fact", text="Synthetic unsupported fact", evidence_ids=["missing-source"])], charts=[ChartPayload(chart_type="insufficient_data", title="No comparable history", purpose="Honest fallback")])
    run = RunRecord(task=Task(objective="Fixture report", entry_skill="prepare_weekly_briefing"), status="partial", main_model="fixture", router_model="fixture", result=result, finished_at=utc_now())
    with Store(app_environment) as store:
        store.save_run(run)
        store.save_report(run.id, datetime(2026, 10, 5).date(), result)
    app = open_app(app_environment)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception and len(app.get("plotly_chart")) == 0
    assert any("Insufficient comparable observations" in info.value for info in app.info)
    assert any("Evidence unavailable: missing-source" in warning.value for warning in app.warning)
    assert any("Unsupported fact" in markdown.value for markdown in app.markdown)


def test_activity_retains_failed_run_and_source_actions(app_environment):
    class FailedModel:
        def complete(self, *args, **kwargs):
            raise ProviderError("synthetic model outage")
    with Store(app_environment) as store:
        failed = Runner(Settings(), store, model=FailedModel()).run(Task(objective="Synthetic research", entry_skill="research_market", trigger="manual"))
    app = open_app(app_environment)
    app.button(key="open-run-history").click().run()
    assert not app.exception
    assert any("synthetic model outage" in markdown.value for markdown in app.markdown)
    assert any("Failed" in frame.value["Status"].tolist() for frame in app.dataframe if "Status" in frame.value.columns)
    assert any(failed.id in caption.value for caption in app.caption)
    assert any("Observable actions" in expander.label for expander in app.expander)
    app.checkbox(key=f"saved-result-{failed.id}").check().run()
    assert not app.exception
    assert any(failed.result.answer == warning.value for warning in app.warning)
    assert any(failed.result.reason == caption.value for caption in app.caption)


def test_dated_key_figure_and_monitoring_qualification_render_together(app_environment):
    document, sources, observations = fixture_batches()[0]
    source = Evidence(**{**sources[0].model_dump(exclude={"id"}), "excerpt": "Synthetic Central London vacancy was 8.0% in Q1 2026.", "native_geography": "Central London", "mapped_geography": "Central London", "validation_reason": "Synthetic publisher extract; wider London coverage remains missing."})
    observation = Observation(**{**observations[0].model_dump(exclude={"id"}), "native_geography": "Central London", "mapped_geography": "Central London", "evidence_ids": [source.id]})
    result = RunResult(status="partial", answer="Synthetic briefing", reason="test", observation_ids=[observation.id], evidence_ids=[source.id], claims=[Claim(kind="fact", text=source.excerpt, evidence_ids=[source.id])])
    run = RunRecord(task=Task(objective="Synthetic briefing", entry_skill="prepare_weekly_briefing"), status="partial", main_model="fixture", router_model="fixture", result=result, finished_at=utc_now())
    with Store(app_environment) as store:
        store.ingest(document, [source], [observation])
        store.save_run(run)
        store.save_report(run.id, datetime(2026, 10, 5).date(), result)
    app = open_app(app_environment)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert 'market-key-figure' in rendered and 'Vacancy rate · Central London' in rendered
    assert '2026-01-01 to 2026-03-31' in rendered and observation.definition in rendered
    assert any(source.validation_reason in item.value for item in app.caption)


def test_repeat_source_check_distinguishes_no_new_content_from_market_change(app_environment):
    with Store(app_environment) as store:
        jobs = ManualJobs(Settings(database_path=app_environment), store)
        jobs.run_manual("collection")
        repeated, _ = jobs.run_manual("collection")
        assert repeated.status == "partial"
        assert repeated.result.collection_counts["evidence"] == 0
    app = open_app(app_environment)
    app.button(key="open-run-history").click().run()
    app.selectbox(key="run-select-fixture").select(repeated.id).run()
    checks = next(frame.value for frame in app.dataframe if "Checked" in frame.value.columns)
    assert checks["Status"].tolist() == ["No new source content"] * 5
    app.checkbox(key=f"saved-result-{repeated.id}").check().run()
    assert any("do not establish unchanged market conditions" in item.value for item in app.markdown)


def test_run_time_display_converts_to_utc_and_retains_exact_details(app_environment):
    from app import display_run_time

    started = datetime.fromisoformat("2026-10-10T19:30:45.123456+08:00")
    assert display_run_time(started) == "10 Oct 2026 11:30 UTC"
    assert display_run_time(None) == "Unknown"
    result = RunResult(status="partial", answer="Synthetic test", reason="Preserve timestamps")
    run = RunRecord(task=Task(objective="Synthetic test", entry_skill="research_market"), started_at=started, finished_at=started, status="partial", main_model="fixture", router_model="fixture", result=result)
    with Store(app_environment) as store:
        store.save_run(run)
    app = open_app(app_environment)
    app.button(key="open-run-history").click().run()
    runs = next(frame.value for frame in app.dataframe if "Started" in frame.value.columns)
    assert runs["Started"].tolist() == ["10 Oct 2026 11:30 UTC"]
    assert app.selectbox(key="run-select-fixture").options[0].startswith("Run 1 · 10 Oct 2026 11:30 UTC")
    assert any(started.isoformat() in item.value for item in app.caption)


def test_controls_jobs_navigation_and_retained_report_preview(app_environment):
    app = open_app(app_environment)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not any(item.key in ("collect", "briefing") for item in app.button)
    app.button(key="reports-controls-fixture").click().run()
    assert not app.exception and app.radio(key="view").value == "Controls"
    assert any("on demand" in item.value for item in app.caption)
    app.button(key="collect").click().run()
    with Store(app_environment) as store:
        assert store.list_reports("fixture") == []
        collection = store.list_runs("fixture")[0]
        assert collection.task.entry_skill == "collect_market_evidence"
    app.run()
    assert any(collection.id in item.value for item in app.caption)
    app.selectbox(key="data-mode").select("live").run()
    assert not app.exception
    assert not any(collection.id in item.value for item in app.caption)
    app.selectbox(key="data-mode").select("fixture").run()
    app.button(key="controls-chat-fixture").click().run()
    assert not app.exception and app.radio(key="view").value == "Research"
    app.radio(key="view").set_value("Controls").run()
    assert any(collection.id in item.value for item in app.caption)
    app.button(key="briefing").click().run()
    with Store(app_environment) as store:
        first = store.list_reports("fixture")[0]
        first_result = first["result"].model_dump(mode="json")
    app.button(key="controls-report-fixture").click().run()
    assert not app.exception
    assert not any(item.key == "report-select-fixture" for item in app.selectbox)
    assert not any(item.key in ("collect", "briefing") for item in app.button)
    assert any("filed under week of " + first["week_start"] in item.value for item in app.caption)
    app.radio(key="view").set_value("Controls").run()
    app.button(key="briefing").click().run()
    with Store(app_environment) as store:
        latest = store.list_runs("fixture")[0]
        saved = store.list_reports("fixture")
        assert len(saved) == 1 and saved[0]["run_id"] == first["run_id"]
        assert saved[0]["result"].model_dump(mode="json") == first_result
        assert latest.id != first["run_id"]
    assert any("first saved report was retained" in item.value for item in app.info)
    assert any(latest.id in item.value for item in app.caption)
    assert not any(first["run_id"] in item.value for item in app.caption)
    app.button(key="controls-run-fixture").click().run()
    assert not app.exception and app.session_state["show-run-history"]
    assert app.selectbox(key="run-select-fixture").value == latest.id


def test_controls_preserve_collection_source_failure_and_coverage(app_environment):
    class MissingCityQuarter(FixtureWebProvider):
        def retrieve(self, url, timeout):
            if "city-q2" in url:
                raise ProviderError("synthetic collection source unavailable")
            return super().retrieve(url, timeout)

    with Store(app_environment) as store:
        settings = Settings(database_path=app_environment)
        run, report_id = ManualJobs(settings, store, Runner(settings, store, web=MissingCityQuarter())).run_manual("collection")
        assert report_id is None and run.status == "partial"
    app = open_app(app_environment)
    app.radio(key="view").set_value("Controls").run()
    assert not app.exception
    checks = next(frame.value for frame in app.dataframe if "Checked" in frame.value.columns)
    assert "Check failed" in checks["Status"].tolist()
    assert "synthetic collection source unavailable" in checks["Error"].tolist()
    gaps = next(table.value for table in app.table if "Coverage area" in table.value.columns)
    assert "Canary Wharf" in gaps["Coverage area"].tolist()
    assert any(run.id in item.value for item in app.caption)
