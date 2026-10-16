"""U1 behavior: persisted provenance, repeat imports, conflicts and offline fixtures."""

import hashlib
import json
import socket
import subprocess
import sys
from datetime import date

import pytest
from pydantic import ValidationError

from london_market.config import Settings, load_settings
from london_market.domain import Document, Evidence, Observation, RunRecord, RunResult, SourceAttempt, Task, TraceEvent, compatibility, utc_now
from london_market.fixtures import coverage_gaps, fixture_batches, load_fixtures
from london_market.store import Store


def republication(batch, slug, value=None):
    document, evidence, observations = batch
    document = Document.model_validate({**document.model_dump(exclude={"id"}), "url": f"https://fixtures.example.invalid/{slug}", "content_hash": hashlib.sha256(slug.encode()).hexdigest(), "published_at": date(2026, 10, 1)})
    item = Evidence.model_validate({**evidence[0].model_dump(exclude={"id"}), "document_id": document.id, "source_url": document.url, "published_at": document.published_at})
    observation = Observation.model_validate({**observations[0].model_dump(exclude={"id"}), "evidence_ids": [item.id], "value": value if value is not None else observations[0].value})
    return document, [item], [observation]


def test_repeat_ingestion_and_restart_readback(tmp_path):
    path = tmp_path / "market.sqlite3"
    with Store(path) as store:
        assert load_fixtures(store) == {"documents": 5, "evidence": 6, "observations": 4, "revisions": 0}
        assert load_fixtures(store) == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
    with Store(path) as store:
        assert store.counts("fixture") == {"documents": 5, "evidence": 6, "observations": 4, "runs": 0, "reports": 0}
        city = next(item for item in store.list_observations("fixture") if item.mapped_geography == "City" and item.period_end == date(2026, 3, 31))
        assert (city.value, city.unit, city.publisher, city.definition, city.category, city.native_geography, city.mapped_geography) == (8.0, "%", "Synthetic Market Publisher", "Unoccupied office space as a share of office stock", "vacancy", "City", "City")
        source = store.get_evidence(city.evidence_ids[0])
        assert source.document_id == fixture_batches()[0][0].id
        assert str(source.source_url) == "https://fixtures.example.invalid/city-q1"
        assert source.excerpt.startswith("Synthetic City:")
        assert source.validation_status == "validated"


def test_same_measurement_across_publications_merges_provenance(tmp_path):
    first = fixture_batches()[0]
    repeat = republication(first, "city-q1-republished")
    with Store(tmp_path / "db") as store:
        store.ingest(*first)
        assert store.ingest(*repeat) == {"documents": 1, "evidence": 1, "observations": 0, "revisions": 0}
        observations = store.list_observations("fixture")
        assert len(observations) == 1
        assert observations[0].id == first[2][0].id == repeat[2][0].id
        assert set(observations[0].evidence_ids) == {first[1][0].id, repeat[1][0].id}


def test_revised_value_and_reversal_preserve_conflict_history(tmp_path):
    first = fixture_batches()[0]
    correction = republication(first, "corrected", 7.8)
    reversal = republication(first, "reversed", 8.0)
    with Store(tmp_path / "db") as store:
        store.ingest(*first)
        assert store.ingest(*correction)["revisions"] == 1
        assert store.ingest(*reversal)["observations"] == 0
        assert store.ingest(*first)["observations"] == 0
        values = store.list_observations("fixture")
        assert len(values) == 2
        assert {item.value for item in values} == {8.0, 7.8}
        revised = next(item for item in values if item.value == 7.8)
        assert revised.revision_of == first[2][0].id
        original = store.get_observation(first[2][0].id)
        assert set(original.evidence_ids) == {first[1][0].id, reversal[1][0].id}


def test_new_publication_is_not_a_new_reference_period_or_event(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        observation = store.get_observation(fixture_batches()[0][2][0].id)
        evidence = store.get_evidence(observation.evidence_ids[0])
        assert observation.period_start == date(2026, 1, 1)
        assert observation.period_end == date(2026, 3, 31)
        assert evidence.event_date == date(2026, 3, 31)
        assert evidence.published_at == date(2026, 9, 30)
        assert evidence.retrieved_at.isoformat() == "2026-10-09T09:00:00+00:00"


def test_incompatible_measurements_are_explicit():
    observations = fixture_batches()[1][2]
    assert compatibility(*observations) == ["metric", "definition", "category"]
    with pytest.raises(ValidationError, match="definition"):
        Observation.model_validate(observations[0].model_dump(exclude={"id", "definition"}))
    with pytest.raises(ValidationError, match="reference period"):
        Observation.model_validate({**observations[0].model_dump(exclude={"id"}), "period_end": "2026-01-01"})


def test_candidate_cannot_become_normalized_without_monitoring_validation(tmp_path):
    doc, evidence, observations = fixture_batches()[0]
    candidate = Evidence.model_validate({**evidence[0].model_dump(), "validation_status": "candidate", "validation_reason": None})
    with Store(tmp_path / "db") as store:
        store.ingest(doc, [candidate], [])
        with pytest.raises(ValueError, match="validated monitoring evidence"):
            store.ingest(doc, [], observations)
        assert len(store.list_observations("fixture")) == 0
        store.set_evidence_validation(candidate.id, "validated", "Synthetic fixture explicitly checked")
        assert store.ingest(doc, [], observations)["observations"] == 1


def test_failed_batch_is_atomic(tmp_path):
    doc, evidence, observations = fixture_batches()[0]
    bad = Observation.model_validate({**observations[0].model_dump(), "evidence_ids": ["missing"]})
    with Store(tmp_path / "db") as store:
        with pytest.raises(ValueError, match="stored evidence"):
            store.ingest(doc, evidence, [bad])
        assert store.counts("fixture")["documents"] == 0
        assert store.counts("fixture")["evidence"] == 0


def test_live_and_fixture_evidence_are_separate_and_aggregate_is_not_coverage(tmp_path):
    doc, evidence, observations = fixture_batches()[0]
    live_doc = Document.model_validate({**doc.model_dump(exclude={"id"}), "mode": "live", "synthetic": False})
    live_evidence = Evidence.model_validate({**evidence[0].model_dump(exclude={"id"}), "document_id": live_doc.id, "mode": "live", "native_geography": "Central London", "mapped_geography": "Central London"})
    live_observation = Observation.model_validate({**observations[0].model_dump(exclude={"id"}), "mode": "live", "native_geography": "Central London", "mapped_geography": "Central London", "evidence_ids": [live_evidence.id]})
    with Store(tmp_path / "db") as store:
        store.ingest(live_doc, [live_evidence], [live_observation])
        load_fixtures(store)
        assert len(store.list_observations("live")) == 1
        assert len(store.list_observations("fixture")) == 4
        assert live_doc.id != doc.id
        assert store.get_evidence(live_evidence.id, "fixture") is None
        assert len(coverage_gaps(store, "live")) == 5
        assert {gap.geography for gap in coverage_gaps(store, "fixture")} == {"Canary Wharf", "Midtown and Fringe", "Wider London"}
    with pytest.raises(ValidationError, match="aggregate source geography"):
        Observation.model_validate({**live_observation.model_dump(exclude={"id"}), "mapped_geography": "City"})


def test_runs_reports_traces_source_failures_survive_reopen(tmp_path):
    path = tmp_path / "db"
    run = RunRecord(task=Task(objective="Synthetic briefing", entry_skill="prepare_weekly_briefing"), main_model="fixture-script", router_model="fixture-script")
    result = RunResult(status="partial", answer="Synthetic result with evidence gaps", reason="Missing submarket observations")
    with Store(path) as store:
        store.save_run(run)
        store.add_trace(run.id, TraceEvent(kind="tool", name="retrieve", outcome="failed", data={"authorization": "never-persist-me", "error": "unavailable"}))
        store.add_source_attempt(run.id, SourceAttempt(url="https://fixtures.example.invalid/unavailable", status="failed", error="synthetic timeout"))
        store.save_run(run.model_copy(update={"status": "partial", "result": result, "finished_at": utc_now()}))
        report = store.save_report(run.id, date(2026, 10, 5), result)
        assert store.save_report(run.id, date(2026, 10, 5), result) == report
    with Store(path) as store:
        assert store.get_run(run.id).result.answer == "Synthetic result with evidence gaps"
        assert store.list_runs("fixture")[0].status == "partial"
        assert store.list_traces(run.id)[0].data == {"authorization": "[redacted]", "error": "unavailable"}
        assert store.list_source_attempts(run.id)[0].error == "synthetic timeout"
        assert len(store.list_reports("fixture")) == 1
        assert store.list_reports("fixture")[0]["result"].status == "partial"
        assert store.list_reports("live") == []


def test_configuration_credentials_are_not_serialized(monkeypatch):
    settings = Settings(openrouter_api_key="private-model-key", tavily_api_key="private-search-key")
    assert "private" not in repr(settings)
    assert "private" not in settings.model_dump_json()
    assert "openrouter_api_key" not in settings.model_dump()
    Settings(openrouter_api_key="model-only").require_live_credentials(search=False)
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        Settings(openrouter_api_key="model-only").require_live_credentials()
    monkeypatch.setenv("LONDON_MARKET_MAX_STEPS", "0")
    with pytest.raises(ValidationError):
        load_settings(env_file=None)
    with pytest.raises(ValidationError):
        Settings(timezone="Moon/Base")


def test_fixture_load_has_no_network_dependency(tmp_path, monkeypatch):
    def deny_network(*args, **kwargs):
        raise AssertionError("fixture attempted network")
    monkeypatch.setattr(socket, "socket", deny_network)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    with Store(tmp_path / "db") as store:
        assert load_fixtures(store)["observations"] == 4


def test_actual_cli_repeated_load_and_readback(tmp_path):
    path = str(tmp_path / "db")
    first = subprocess.run([sys.executable, "-m", "london_market", "init-fixtures", "--db", path], check=True, capture_output=True, text=True)
    second = subprocess.run([sys.executable, "-m", "london_market", "init-fixtures", "--db", path], check=True, capture_output=True, text=True)
    readback = subprocess.run([sys.executable, "-m", "london_market", "inspect", "--db", path, "--mode", "fixture"], check=True, capture_output=True, text=True)
    assert first.stdout.startswith("Demo mode uses synthetic sample data")
    assert json.loads(second.stdout.split("\n", 1)[1])["new"]["observations"] == 0
    snapshot = json.loads(readback.stdout.split("\n", 1)[1])
    assert snapshot["stored"]["observations"] == 4
    assert len(snapshot["coverage_gaps"]) == 3
