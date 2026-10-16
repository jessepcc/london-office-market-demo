from authoring_support import analysis_intents, analysis_sources, seed_analyses
import hashlib

import json
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import Claim, Document, Evidence, Limits, Observation, RunRecord, RunResult, SourceAttempt, Task, TraceEvent, Usage
from london_market.fixtures import fixture_batches, load_fixtures, scripted_coverage
from london_market.providers import FixtureWebProvider, ModelReply, OpenRouterProvider, ProviderError, SignalCard, SignalChoice, SignalReply, ToolCall, WebProvider, signal_payload
from london_market.runner import BriefingFinishArgs, Runner
from london_market.sources import Measurement, MonitoredSource
from london_market.store import Store
from london_market.tools import CompareArgs, QueryArgs, SearchArgs, ToolContext, compare, query_evidence, search
from london_market.weekly import signal_card, weekly_context, weekly_summary

NOW = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)


def task(mode="fixture", **kwargs):
    return Task(objective="Prepare weekly London office intelligence", entry_skill="prepare_weekly_briefing", mode=mode, briefing_date=date(2026, 10, 10), report_week_start=date(2026, 10, 5), **kwargs)


def test_metadata_legacy_and_bounds():
    legacy = Claim(kind="fact", text="Dated observation", evidence_ids=["ev"])
    assert legacy.signal is None and legacy.topic_tags == []
    for changes in ({"topic_tags": ["rents", "rents"]}, {"topic_tags": ["supply", "demand", "rents", "macro"]}, {"topic_tags": ["other"]}, {"kind": "forecast", "signal": "neutral"}):
        with pytest.raises(ValidationError):
            Claim.model_validate({**legacy.model_dump(), **changes})
    with pytest.raises(ValidationError, match="Every new weekly card"):
        BriefingFinishArgs(claims=[legacy], coverage=scripted_coverage([], [], []))
    current = legacy.model_copy(update={"topic_tags": ["demand", "rents", "ESG"]})
    assert BriefingFinishArgs(claims=[current], coverage=scripted_coverage([], [], [])).claims[0].topic_tags == ["demand", "rents", "ESG"]


def test_ingestion_stamp_retry_and_legacy_unknown(tmp_path):
    document, evidence, observations = fixture_batches()[0]
    with Store(tmp_path / "db") as store:
        store.ingest(document, evidence, observations)
        original = store.get_evidence(evidence[0].id)
        assert original.id == evidence[0].id and original.first_ingested_at.tzinfo is not None
        assert store.ingest(document, evidence, observations)["evidence"] == 0
        assert store.get_evidence(original.id).first_ingested_at == original.first_ingested_at
        legacy = original.model_dump(mode="json")
        legacy.pop("first_ingested_at")
        with store.connection:
            store.connection.execute("UPDATE evidence SET body=? WHERE id=?", (json.dumps(legacy), original.id))
        store.ingest(document, evidence, observations)
        assert store.get_evidence(original.id).first_ingested_at is None
        assert store.counts("fixture")["evidence"] == 1


def save_report(store, week, day, mode="fixture", ids=None):
    result = RunResult(status="partial", answer="Original", briefing_date=day, reason="Retained context", evidence_ids=ids or [])
    run = RunRecord(task=task(mode).model_copy(update={"report_week_start": week, "briefing_date": day}), main_model="test", router_model="test", status="partial", result=result, finished_at=NOW)
    store.save_run(run)
    return store.save_report(run.id, week, result)


def test_context_earlier_same_mode_week_and_timezone(tmp_path):
    with Store(tmp_path / "db") as store:
        prior = save_report(store, date(2026, 9, 28), date(2026, 10, 2), ids=["old"])
        same = save_report(store, date(2026, 10, 5), date(2026, 10, 9))
        save_report(store, date(2026, 9, 28), date(2026, 10, 3), mode="live")
        context = weekly_context(store, task(), "Europe/London", NOW)
        assert (context.window_start, context.window_end, context.prior_report_id, context.prior_evidence_ids) == (date(2026, 10, 4), date(2026, 10, 10), prior, ["old"])
        assert context.prior_report_id != same
        context = weekly_context(store, task().model_copy(update={"briefing_date": None}), "Asia/Hong_Kong", datetime(2026, 10, 10, 23, tzinfo=timezone.utc))
        assert context.briefing_date == date(2026, 10, 11)


def test_summary_old_period_new_publication_revision_and_frozen_snapshot(tmp_path):
    document, evidence, observations = fixture_batches()[0]
    document = document.model_copy(update={"published_at": date(2026, 10, 9)})
    evidence = [item.model_copy(update={"published_at": document.published_at, "first_ingested_at": NOW}) for item in evidence]
    with Store(tmp_path / "db") as store:
        store.ingest(document, evidence, observations)
        revision = Observation.model_validate({**observations[0].model_dump(exclude={"id", "revision_of"}), "value": 7.9})
        store.ingest(document, [], [revision])
        run = RunRecord(task=task(), main_model="test", router_model="test", started_at=NOW)
        store.save_run(run)
        summary = weekly_summary(store, "fixture", run.id, weekly_context(store, task(), "UTC", NOW), [], [evidence[0].id], NOW + timedelta(seconds=3))
        update = summary.evidence_updates[0]
        assert (update.publication_in_window, update.event_in_window, update.first_ingested_in_window, update.previously_cited) == (True, False, True, None)
        assert update.reference_periods[0].period_end == date(2026, 3, 31)
        assert update.reference_periods[1].revision_of == observations[0].id
        serialized = summary.model_dump_json()
        store.ingest(document, evidence, observations)
        assert summary.model_dump_json() == serialized
        assert "missing weekly evidence does not establish no market movement" in summary.indicator_gaps[0]


def test_same_url_revision_is_one_publication(tmp_path):
    document, evidence, _ = fixture_batches()[0]
    changed = Document.model_validate({**document.model_dump(exclude={"id"}), "content_hash": hashlib.sha256(b"revision").hexdigest()})
    revised = Evidence.model_validate({**evidence[0].model_dump(exclude={"id"}), "document_id": changed.id, "excerpt": "A revised source excerpt"})
    with Store(tmp_path / "db") as store:
        store.ingest(document, evidence, [])
        store.ingest(changed, [revised], [])
        run = RunRecord(task=task(), main_model="test", router_model="test", started_at=NOW)
        store.save_run(run)
        summary = weekly_summary(store, "fixture", run.id, weekly_context(store, task(), "UTC", NOW), [], [evidence[0].id, revised.id], NOW)
        assert (summary.cited_publication_count, summary.cited_excerpt_count, summary.cited_domain_count) == (1, 2, 1)
        assert summary.source_concentration[0].publication_count == 1


def test_current_collection_attempts_exclude_stale_latest(tmp_path):
    with Store(tmp_path / "db") as store:
        current = RunRecord(task=task(), main_model="test", router_model="test", started_at=NOW)
        recent = RunRecord(task=task().model_copy(update={"entry_skill": "collect_market_evidence"}), main_model="test", router_model="test", started_at=NOW - timedelta(days=1))
        stale = RunRecord(task=recent.task, main_model="test", router_model="test", started_at=NOW - timedelta(days=30))
        for record in (current, recent, stale):
            store.save_run(record)
            store.add_source_attempt(record.id, SourceAttempt(url="https://example.com/source", status="failed", at=record.started_at, error="observed timeout"))
        store.add_trace(recent.id, TraceEvent(at=recent.started_at, kind="tool", name="search", outcome="success", data={"arguments": {"query": "London office current announcement"}, "output": {"start_date": "2026-10-04", "end_date": "2026-10-11"}}))
        summary = weekly_summary(store, "fixture", current.id, weekly_context(store, task(), "UTC", NOW), [], [], NOW)
        assert {item.run_id for item in summary.source_attempts} == {current.id, recent.id}
        assert (summary.fresh_searches[0].query, summary.fresh_searches[0].start_date) == ("London office current announcement", date(2026, 10, 4))


def test_weekly_search_window_and_publication_query(tmp_path):
    payloads = []
    def respond(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"results": []})
    with Store(tmp_path / "db") as store, httpx.MockTransport(respond) as transport:
        ctx = ToolContext(Settings(), store, WebProvider(Settings(mode="live", tavily_api_key="mock"), transport), "fixture", "test", 1, weekly=weekly_context(store, task(), "UTC", NOW))
        result = search(ctx, SearchArgs(query="London office news"))
        assert (payloads[0]["start_date"], payloads[0]["end_date"], result.start_date, result.end_date) == ("2026-10-04", "2026-10-11", date(2026, 10, 4), date(2026, 10, 11))
        load_fixtures(store)
        result = query_evidence(ctx, QueryArgs(published_since=date(2026, 9, 1), published_until=date(2026, 10, 10)))
        assert [item.published_at for item in result.evidence] == [date(2026, 9, 30)]
        assert [item.period_end for item in result.observations] == [date(2026, 3, 31)]


def test_clef_exact_choices_and_compact_utf8_boundary():
    card = SignalCard(fingerprint="card_0011", text="Availability fell in comparable quarters.", qualifications=["City; Q1 and Q2."])
    def respond(request):
        payload = json.loads(request.content)
        assert len(payload["state"][0].encode()) <= 1500
        assert set(payload["questions"]) == {"card_0"}
        return httpx.Response(200, json={"model": "cloudflare/clef-flash", "provider": "test", "answers": {"card_0": {"type": "choice", "choice": "improves_landlord_leasing_conditions", "probabilities": {"improves_landlord_leasing_conditions": .8, "worsens_landlord_leasing_conditions": .1, "no_directional_leasing_effect": .1}, "confidence": .7}}, "usage": {"input_tokens": 60, "output_tokens": 0, "cost": .00001}})
    with httpx.MockTransport(respond) as transport:
        reply = OpenRouterProvider(Settings(openrouter_api_key="mock"), transport).classify_signals([card], 1)
        assert (reply.choices[card.fingerprint].choice, reply.provider, reply.usage.input_tokens) == ("bullish", "test", 60)
    with pytest.raises(ProviderError, match="compact"):
        signal_payload([card.model_copy(update={"text": "界" * 700})], "clef")
    with pytest.raises(ValidationError, match="contradicts"):
        SignalChoice(choice="neutral", probabilities={"bullish": .8, "bearish": .1, "neutral": .1}, confidence=.7)


class WeeklyModel:
    def __init__(self, claim, error=None):
        self.claim, self.error, self.classification_calls = claim, error, 0

    def complete(self, messages, tools, max_output_tokens, timeout):
        schema = next(item for item in tools if item["function"]["name"] == "finish")["function"]["parameters"]["$defs"]["Claim"]
        assert "signal" not in schema["properties"]
        if not any(message.get("role") == "tool" for message in messages):
            return ModelReply(model="test-main", provider="mock", usage=Usage(), calls=[ToolCall(id="window", name="query_evidence", arguments=json.dumps({"published_since": "2026-10-04", "published_until": "2026-10-10"}))])
        return ModelReply(model="test-main", provider="mock", usage=Usage(), calls=[ToolCall(id="finish", name="finish", arguments=json.dumps({"claims": [self.claim], "coverage": scripted_coverage([self.claim], [item.model_dump(mode="json") for item in fixture_batches()[0][2]], [])}))])

    def classify_signals(self, cards, timeout):
        self.classification_calls += 1
        if self.error:
            raise self.error
        return SignalReply(choices={card.fingerprint: SignalChoice(choice="bullish", probabilities={"bullish": .8, "bearish": .1, "neutral": .1}, confidence=.7) for card in cards}, model="clef-test", provider="mock", usage=Usage(input_tokens=50, spend_usd=.00001))


@pytest.mark.parametrize("error", [None, ProviderError("Observed classifier failure")])
def test_runner_classifies_after_audit_no_neutral_fallback(tmp_path, error):
    with Store(tmp_path / "db") as store:
        store.ingest(*fixture_batches()[0])
        evidence = store.list_evidence("fixture")[0]
        payload = {"kind": "fact", "text": evidence.excerpt, "evidence_ids": [evidence.id], "topic_tags": ["availability"]}
        model = WeeklyModel(payload, error)
        run = Runner(Settings(), store, model=model).run(task())
        assert (run.status, model.classification_calls, run.result.claims[0].signal) == ("partial", 1, "bullish" if error is None else None)
        assert run.result.weekly.unassessed_fact_count == (1 if error else 0)
        assert run.usage.input_tokens == (50 if error is None else 0)
        invalid_model = WeeklyModel({**payload, "text": "Vacancy reached 999%."})
        rejected = Runner(Settings(), store, model=invalid_model).run(task())
        assert invalid_model.classification_calls == 0 and rejected.status == "stopped"




def test_classification_capacity_rejects_before_any_calls_or_report(tmp_path):
    with Store(tmp_path / "db") as store:
        store.ingest(*fixture_batches()[0])
        evidence = store.list_evidence("fixture")[0]
        model = WeeklyModel({"kind": "fact", "text": evidence.excerpt, "evidence_ids": [evidence.id], "topic_tags": ["availability"]})
        run = Runner(Settings(), store, model=model).run(task(limits=Limits(max_steps=5)))
        assert run.status == "stopped" and model.classification_calls == 0
        assert run.result.claims == [] and store.list_reports("fixture") == []
        assert any("shortfall 1" in str(item.data) for item in store.list_traces(run.id))
        assert store.list_traces(run.id)[-1].data["steps"] == 5


def test_material_qualifications_oversize_is_not_truncated(tmp_path):
    with Store(tmp_path / "db") as store:
        document, evidence, observations = fixture_batches()[0]
        evidence = [evidence[0].model_copy(update={"validation_reason": "Reviewed. Source qualifications. " + "Planned, not completed. " * 100})]
        store.ingest(document, evidence, observations)
        claim = Claim(kind="fact", text=evidence[0].excerpt, evidence_ids=[evidence[0].id], topic_tags=["availability"])
        card = signal_card(claim, store, "fixture")
        assert "Planned, not completed. " * 100 in card.qualifications
        model = WeeklyModel(claim.model_dump())
        run = Runner(Settings(), store, model=model).run(task())
        assert run.status == "stopped" and model.classification_calls == 0
        assert run.result.claims == [] and store.list_reports("fixture") == []
        assert any("oversized_claim_indexes" in item.data for item in store.list_traces(run.id))


@pytest.mark.parametrize("initial_query", [None, {"category": "vacancy"}, {"offset": 1}, {"published_since": "2026-10-05"}, {"limit": 99}])
def test_weekly_requires_observed_complete_window_query_and_repairs_empty_result(tmp_path, initial_query):
    class RepairWindowModel(WeeklyModel):
        def __init__(self, claim):
            super().__init__(claim)
            self.phase = 0
            self.audit_issues = []
            self.empty_window_observed = False

        def complete(self, messages, tools, max_output_tokens, timeout):
            self.phase += 1
            window = {"published_since": "2026-10-04", "published_until": "2026-10-10"}
            if self.phase == 1 and initial_query is not None:
                name, args = "query_evidence", {**window, **initial_query}
            elif self.phase == (2 if initial_query is not None else 1):
                name, args = "finish", {"claims": [self.claim], "coverage": scripted_coverage([self.claim], [item.model_dump(mode="json") for item in fixture_batches()[0][2]], [])}
            elif self.phase == (3 if initial_query is not None else 2):
                feedback = json.loads(next(message["content"] for message in reversed(messages) if message.get("role") == "tool"))
                self.audit_issues = feedback["audit_issues"]
                assert "query_evidence" in {tool["function"]["name"] for tool in tools}
                name, args = "query_evidence", window
            else:
                output = json.loads(next(message["content"] for message in reversed(messages) if message.get("role") == "tool"))
                self.empty_window_observed = output["evidence_total"] == 0
                name, args = "finish", {"claims": [self.claim], "coverage": scripted_coverage([self.claim], [item.model_dump(mode="json") for item in fixture_batches()[0][2]], []), "limitations": ["No publications were found within the briefing window; cited facts are dated historical context."]}
            return ModelReply(model="test-main", provider="mock", usage=Usage(), calls=[ToolCall(id=str(self.phase), name=name, arguments=json.dumps(args))])

    with Store(tmp_path / "db") as store:
        store.ingest(*fixture_batches()[0])
        evidence = store.list_evidence("fixture")[0]
        model = RepairWindowModel({"kind": "fact", "text": evidence.excerpt, "evidence_ids": [evidence.id], "topic_tags": ["availability"]})
        run = Runner(Settings(), store, model=model).run(task())
        assert run.status == "partial" and run.result.claims[0].text == evidence.excerpt
        assert model.empty_window_observed and any("pinned publication window" in issue for issue in model.audit_issues)
        assert [event.outcome for event in store.list_traces(run.id) if event.name == "finish"] == ["failed", "success"]
        assert model.classification_calls == 1


def test_signal_card_consolidates_measurements_without_losing_periods(tmp_path):
    document, evidence, _ = fixture_batches()[0]
    periods = [(date(2025, 7, 1), date(2025, 9, 30)), (date(2025, 10, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 3, 31)), (date(2026, 4, 1), date(2026, 6, 30))]
    definition = "Office vacancy rate at reference-period end; not availability."
    observations = [Observation(mode="fixture", metric="vacancy_rate", value=7.2, unit="%", publisher=document.publisher, definition=definition, category="vacancy", native_geography="Central London", mapped_geography="Central London", period_start=start, period_end=end, evidence_ids=[evidence[0].id]) for start, end in periods]
    with Store(tmp_path / "db") as store:
        store.ingest(document, evidence, observations)
        claim = Claim(kind="fact", text=evidence[0].excerpt, evidence_ids=[evidence[0].id])
        card = signal_card(claim, store, "fixture")
        measurements = [item for item in card.qualifications if item.startswith("Measurement:")]
        assert len(measurements) == 1 and measurements[0].count(definition) == 1
        for start, end in periods:
            assert f"{start} to {end}" in measurements[0]
        assert len(signal_payload([card], "clef")["state"][0].encode("utf-8")) <= 1500


def test_clef_contract_requires_neutral_static_levels_preserves_announced_demand():
    cards = [SignalCard(fingerprint="level", text="Central London vacancy was 8.4% in Q1.", qualifications=[]), SignalCard(fingerprint="demand", text="An occupier announced a planned London headquarters move.", qualifications=[])]
    payload = signal_payload(cards, "clef")
    for question in payload["questions"].values():
        assert "A standalone level or total" in question["criteria"]["no_directional_leasing_effect"]
        assert "announced/planned/completed status" in question["instructions"]
        assert "Do not infer benchmarks, trends or office effects" in question["instructions"]


def test_clef_reassessment_contract_preserves_exact_qualified_facts_and_returned_choices():
    cards = [
        SignalCard(fingerprint="premium", text="Savills reported H1 2026 average rents in BREEAM 'Excellent' or 'Outstanding' buildings were 49% higher than non-rated buildings in the West End and 39% higher in the City. A level premium, not evidence of a widening gap.", qualifications=["Publication: 2026-08-06; Central London; qualitative evidence, not a normalized series."]),
        SignalCard(fingerprint="housing", text="Lendlease submitted plans to Newham for Stratford Cross's final phase: approximately 2,000 homes, a public park and retail. A submitted residential-led proposal, not consented or completed office supply.", qualifications=["Submitted residential-led proposal, not consented development or completed office supply; relevant competing land-use news around an established commercial district."]),
        SignalCard(fingerprint="offices", text="An occupier announced a planned London office headquarters move.", qualifications=["Planned occupation, not completed take-up or a signed lease."]),
    ]
    def respond(request):
        payload = json.loads(request.content)
        state = json.loads(payload["state"][0])
        assert list(state) == ["landlord_leasing_definition", "card_0", "card_1", "card_2"]
        assert state["landlord_leasing_definition"].startswith("Landlord office-leasing perspective:")
        assert {key: value for key, value in state.items() if key != "landlord_leasing_definition"} == {f"card_{i}": {"fact": card.text, "qualifications": card.qualifications} for i, card in enumerate(cards)}
        assert len(payload["state"][0].encode("utf-8")) <= 2000
        for question in payload["questions"].values():
            assert set(question) == {"type", "instructions", "criteria"}
            assert set(question["criteria"]) == {"improves_landlord_leasing_conditions", "worsens_landlord_leasing_conditions", "no_directional_leasing_effect"}
            assert "a static certification rent premium" in question["criteria"]["no_directional_leasing_effect"]
            assert "Housing, parks or retail plans alone establish no office-market effect" in question["criteria"]["no_directional_leasing_effect"]
            assert "Do not infer benchmarks, trends or office effects" in question["instructions"]
        # This is a scripted provider receipt; real model semantics require live proof.
        answers = {f"card_{i}": {"type": "choice", "choice": choice, "probabilities": {name: .98 if name == choice else .01 for name in ("improves_landlord_leasing_conditions", "worsens_landlord_leasing_conditions", "no_directional_leasing_effect")}, "confidence": .98} for i, choice in enumerate(("no_directional_leasing_effect", "no_directional_leasing_effect", "improves_landlord_leasing_conditions"))}
        return httpx.Response(200, json={"model": "cloudflare/clef-flash", "provider": "scripted-contract-test", "answers": answers, "usage": {"input_tokens": 240, "output_tokens": 0, "cost": .00001}})
    with httpx.MockTransport(respond) as transport:
        reply = OpenRouterProvider(Settings(openrouter_api_key="mock"), transport).classify_signals(cards, 1)
    assert {fingerprint: choice.choice for fingerprint, choice in reply.choices.items()} == {"premium": "neutral", "housing": "neutral", "offices": "bullish"}
    assert reply.provider == "scripted-contract-test" and reply.usage.input_tokens == 240 and reply.usage.spend_usd == .00001
    assert all(set(choice.model_dump()) == {"choice", "probabilities", "confidence"} for choice in reply.choices.values())


@pytest.mark.parametrize("basis,periods,expected", [
    ("period_total", [(date(2026, 9, 27), date(2026, 10, 3)), (date(2026, 10, 4), date(2026, 10, 10))], True),
    ("snapshot", [(date(2026, 10, 3), date(2026, 10, 3)), (date(2026, 10, 10), date(2026, 10, 10))], True),
    ("period_total", [(date(2026, 10, 3), date(2026, 10, 3)), (date(2026, 10, 10), date(2026, 10, 10))], False),
    (None, [(date(2026, 9, 27), date(2026, 10, 3)), (date(2026, 10, 4), date(2026, 10, 10))], False),
])
def test_weekly_source_basis_total_vs_snapshot(tmp_path, monkeypatch, basis, periods, expected):
    import london_market.tools as tools
    measurement = Measurement("office_flow", "", "City", "sq ft", "Exact weekly source definition", "take-up", weekly_basis=basis)
    source = MonitoredSource("test", "https://example.com/weekly", "", None, None, (measurement,), (), ())
    monkeypatch.setattr(tools, "MONITORED_SOURCES", {"test": source})
    with Store(tmp_path / "db") as store:
        document = Document(mode="fixture", synthetic=True, url=source.url, title="Synthetic weekly evidence", publisher="Test", content_hash=hashlib.sha256(b"weekly").hexdigest())
        evidence = Evidence(document_id=document.id, mode="fixture", excerpt="Synthetic weekly values 100 and 120 sq ft.", retrieved_at=NOW, source_url=document.url, native_geography="City", mapped_geography="City", category="take-up", validation_status="validated", validation_reason="Reviewed test source")
        observations = [Observation(mode="fixture", metric=measurement.metric, value=value, unit=measurement.unit, publisher="Test", definition=measurement.definition, category="take-up", native_geography="City", mapped_geography="City", period_start=start, period_end=end, evidence_ids=[evidence.id]) for (start, end), value in zip(periods, (100, 120))]
        store.ingest(document, [evidence], observations)
        ctx = ToolContext(Settings(), store, FixtureWebProvider(), "fixture", "run", 1)
        output = compare(ctx, CompareArgs(observation_ids=[item.id for item in observations]))
        assert output.compatible and output.absolute_change == 20 and output.week_over_week == expected
        assert output.weekly_basis == (basis if expected else None)
        altered = Observation.model_validate({**observations[1].model_dump(exclude={"id"}), "definition": "Different definition"})
        store.ingest(document, [], [altered])
        refused = compare(ctx, CompareArgs(observation_ids=[observations[0].id, altered.id]))
        assert not refused.compatible and "definition" in refused.reason
