"""Scripted selection proof through the real Runner; no live model-quality claim."""

import hashlib
import json
from datetime import date, datetime, timezone

import pytest

from london_market.config import Settings
from london_market.domain import Document, Evidence, Limits, Observation, RunResult, Task, Usage
from london_market.fixtures import load_fixtures, scripted_coverage
from london_market.providers import ModelReply, ToolCall
from london_market.report_demo import seed_report_demo
from london_market.runner import Runner
from london_market.skills import default_registry
from london_market.store import Store

TOPIC_CASES = [
    ("analyze_office_rents", "rents", "Fictional City asking rent excludes incentives; achieved rent is unknown."),
    ("analyze_leasing_activity", "take-up", "Fictional City letting was signed in Q2; it does not establish net absorption."),
    ("analyze_supply_pipeline", "pipeline", "Fictional City project has planning consent; construction and prelets are unconfirmed."),
    ("analyze_submarkets", "submarkets", "Fictional City evidence covers City only; Wider London is unobserved."),
    ("analyze_macro_drivers", "macro", "Fictional UK employment release published in September covers Q1; London office effects are untested."),
    ("analyze_occupier_demand", "ESG", "Fictional City occupier leased a certified building; this one deal cannot establish an ESG premium."),
    ("analyze_market_events", "news", "Fictional City lease announced in September occurred in Q1; repeated articles describe the same event."),
]


def seed_topic_evidence(store, category, excerpt):
    """One clearly fictional, source-native qualitative record with no normalization."""
    now = datetime(2026, 10, 9, 9, tzinfo=timezone.utc)
    document = Document(mode="fixture", synthetic=True, url=f"https://topic-proof.example.invalid/{category}", title=f"Fictional {category} topic proof", publisher="Fictional Topic Publisher", content_hash=hashlib.sha256(excerpt.encode()).hexdigest(), published_at=date(2026, 9, 30), retrieved_at=now)
    evidence = Evidence(document_id=document.id, mode="fixture", excerpt=excerpt, source_url=document.url, native_geography="City", mapped_geography="City", category=category, published_at=document.published_at, retrieved_at=now)
    store.ingest(document, [evidence], [])
    return evidence


class TopicProofModel:
    """Public reusable fixture provider: inspect real results before cited finish.

    Root verification can import this module with runpy and persist the same runs.
    Selection and wording are scripted; factual excerpts come from tool output.
    """

    def __init__(self, skill_id, *, geography="City", comparison_ids=None, forbidden=False):
        self.skill_id = skill_id
        self.geography = geography
        self.comparison_ids = comparison_ids
        self.forbidden = forbidden
        self.rounds = 0
        self.inspected_query = None
        self.inspected_comparison = None

    def complete(self, messages, tools, max_output_tokens, timeout):
        if self.rounds == 0:
            pinned = next((line.removeprefix("Pinned weekly context: ") for message in messages for line in (message.get("content") or "").splitlines() if line.startswith("Pinned weekly context: ")), None)
            if pinned and not any(message.get("role") == "tool" for message in messages):
                context = json.loads(pinned)
                return ModelReply(model="synthetic-topic-proof", provider="synthetic-scripted", usage=Usage(), calls=[ToolCall(id="window", name="query_evidence", arguments=json.dumps({"published_since": context["window_start"], "published_until": context["window_end"]}))])
        self.rounds += 1
        outputs = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if self.rounds == 1:
            name, args = "load_skill", {"skill_id": self.skill_id}
        elif self.forbidden:
            name, args = "normalize_observation", {}
        elif self.rounds == 2:
            name, args = "query_evidence", {"geography": self.geography, "limit": 12}
        elif self.rounds == 3:
            self.inspected_query = outputs[-1]
            assert "evidence" in self.inspected_query
            if self.comparison_ids:
                name, args = "compare", {"observation_ids": self.comparison_ids}
            else:
                name, args = "finish", self.finish_query()
        else:
            self.inspected_comparison = outputs[-1]
            assert self.inspected_comparison["compatible"] is False
            name, args = "finish", {"claims": [{"kind": "watchpoint", "text": "Fictional comparison refused: " + self.inspected_comparison["reason"]}], "topic_gaps": ["No compatible change can be reported"]}
        if name == "finish" and any("Pinned weekly context:" in str(item.get("content")) for item in messages):
            args["coverage"] = scripted_coverage(args.get("claims", []), self.inspected_query.get("observations", []) if self.inspected_query else [], [], evidence=self.inspected_query.get("evidence", []) if self.inspected_query else [])
        return ModelReply(model="synthetic-topic-proof", provider="synthetic-scripted", usage=Usage(), calls=[ToolCall(id=f"topic-{self.rounds}", name=name, arguments=json.dumps(args))])

    def finish_query(self):
        evidence = self.inspected_query["evidence"]
        if not evidence:
            return {"response_type": "insufficient_data", "message": f"No evidence found for {self.geography} within this fixture query.", "topic_gaps": [f"{self.geography}: no evidence in queried scope"]}
        item = evidence[0]
        return {"claims": [{"kind": "fact", "text": item["excerpt"], "evidence_ids": [item["id"]]}], "limitations": ["Scripted synthetic topic-selection proof; no professional live research validation"]}


def run_topic_proof(store, skill_id, *, entry_skill="research_market", limits=None, **kwargs):
    provider = TopicProofModel(skill_id, **kwargs)
    task = Task(objective="Investigate the requested London office topic", entry_skill=entry_skill, trigger="manual", mode="fixture", limits=limits or Limits())
    return Runner(Settings(), store, model=provider).run(task), provider


@pytest.mark.parametrize("skill_id,category,excerpt", TOPIC_CASES)
def test_all_topics_load_inspect_evidence_and_finish_in_shared_runner(tmp_path, skill_id, category, excerpt):
    with Store(tmp_path / "db") as store:
        evidence = seed_topic_evidence(store, category, excerpt)
        run, model = run_topic_proof(store, skill_id)
        assert run.status == "partial" and store.get_run(run.id).result == run.result
        assert model.inspected_query["evidence_total"] == 1
        assert run.result.claims[0].text == evidence.excerpt
        assert run.result.evidence_ids == [evidence.id]
        traces = store.list_traces(run.id)
        selected = next(event for event in traces if event.kind == "skill" and event.name == skill_id)
        loaded = default_registry().load(skill_id)
        assert selected.data["version"] == ("7" if skill_id == "analyze_macro_drivers" else "6" if skill_id in ("analyze_leasing_activity", "analyze_office_rents") else "5")
        assert selected.data["instruction_hash"] == loaded["instruction_hash"]
        assert [event.name for event in traces if event.kind == "tool"] == ["load_skill", "query_evidence", "finish"]
        assert all(event.outcome == "success" for event in traces if event.kind == "tool")
        assert traces[-1].kind == "termination"
        assert store.list_observations("fixture") == []


def test_weekly_selects_topic_and_specialist_entry_cannot_normalize(tmp_path):
    with Store(tmp_path / "db") as store:
        seed_topic_evidence(store, "news", TOPIC_CASES[-1][2])
        weekly, _ = run_topic_proof(store, "analyze_market_events", entry_skill="prepare_weekly_briefing")
        assert weekly.result.evidence_ids
        assert {event.name for event in store.list_traces(weekly.id) if event.kind == "skill"} == {"prepare_weekly_briefing", "analyze_market_events"}
        denied, _ = run_topic_proof(store, "analyze_office_rents", entry_skill="analyze_office_rents", forbidden=True)
        errors = [event.data["error"] for event in store.list_traces(denied.id) if event.kind == "tool" and event.outcome == "failed"]
        assert errors and all("not an approved binding" in error for error in errors)
        assert denied.status == "stopped"
        assert store.list_observations("fixture") == []


def test_explicit_low_action_cap_stops_topic_run_without_extra_calls(tmp_path):
    with Store(tmp_path / "db") as store:
        run, model = run_topic_proof(store, "analyze_office_rents", limits=Limits(max_steps=2))
        assert run.status == "stopped" and "Step limit" in run.result.reason
        assert model.rounds == 1
        assert not any(event.name == "query_evidence" for event in store.list_traces(run.id))
        assert store.get_run(run.id).finished_at is not None


def test_missing_wider_london_evidence_finishes_with_honest_gap(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        run, model = run_topic_proof(store, "analyze_submarkets", geography="Wider London")
        assert model.inspected_query["evidence_total"] == 0
        assert run.status == "partial" and run.result.topic_gaps
        assert not run.result.claims and not run.result.evidence_ids
        assert "Wider London" in run.result.answer


@pytest.mark.parametrize("case", ["rent_basis", "vacancy_availability"])
def test_real_comparison_refuses_incompatible_measurements(tmp_path, case):
    with Store(tmp_path / "db") as store:
        if case == "rent_basis":
            seed_report_demo(store)
            rents = sorted([item for item in store.list_observations("fixture") if item.metric == "prime_asking_rent" and item.mapped_geography == "City"], key=lambda item: item.period_start)
            baseline = rents[0]
            achieved = Observation(**{**rents[1].model_dump(exclude={"id"}), "definition": "Achieved headline rent, different from asking", "value": 100})
            evidence = store.get_evidence(achieved.evidence_ids[0], "fixture")
            store.ingest(store.get_document(evidence.document_id, "fixture"), [], [achieved])
            ids = [baseline.id, achieved.id]
            topic = "analyze_office_rents"
        else:
            load_fixtures(store)
            items = [item for item in store.list_observations("fixture") if item.mapped_geography == "City" and item.period_start == date(2026, 4, 1)]
            ids = [item.id for item in items]
            topic = "analyze_leasing_activity"
        run, model = run_topic_proof(store, topic, comparison_ids=ids)
        assert not model.inspected_comparison["compatible"]
        assert model.inspected_comparison["absolute_change"] is None
        assert run.result.topic_gaps == ["No compatible change can be reported"]
        assert all(claim.kind == "watchpoint" for claim in run.result.claims)
        assert any(event.name == "compare" and event.outcome == "success" for event in store.list_traces(run.id))


def test_specialist_registry_contract_is_exact_and_unique():
    registry = default_registry()
    assert len(registry.catalog()) == 14
    for skill_id, _, _ in TOPIC_CASES:
        skill = registry.skills[skill_id]
        assert skill.input_schema is Task and skill.output_schema is RunResult
        assert {binding.name for binding in skill.tool_bindings} == {"search", "retrieve", "query_evidence", "cache_evidence", "compare"}
