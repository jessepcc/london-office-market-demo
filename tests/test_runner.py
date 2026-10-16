"""Bounded shared-loop behavior, contextual routing and deterministic charts."""

import json
import socket
from datetime import date

import pytest

from london_market.config import Settings
from london_market.domain import ChatMessage, Claim, Document, Evidence, Limits, Observation, RunRecord, Task, Usage
from london_market.fixtures import fixture_batches, load_fixtures
from london_market.providers import FixtureModelProvider, FixtureWebProvider, ModelReply, ProviderError, RouteChoice, RouteReply, ToolCall
from london_market.runner import Runner
from london_market.skills import default_registry
from london_market.store import Store
from london_market.tools import CompareArgs, ToolContext, VisualizationArgs, PrepareArgs, NormalizeArgs, cache_evidence, CacheArgs, choose_visualization, compare, normalize_observation, prepare_visualization_data


class ScriptModel:
    def __init__(self, actions, route=None):
        self.actions = actions
        self.count = 0
        self.route_reply = route

    def route(self, task, timeout):
        if isinstance(self.route_reply, Exception):
            raise self.route_reply
        return self.route_reply or FixtureModelProvider(task).route(task, timeout)

    def complete(self, messages, tools, max_output_tokens, timeout):
        self.count += 1
        name, payload = self.actions[min(self.count - 1, len(self.actions) - 1)]
        return ModelReply(model="synthetic-test-model", provider="synthetic", calls=[ToolCall(id=f"test{self.count}", name=name, arguments=payload if isinstance(payload, str) else json.dumps(payload))], usage=Usage())


def task(question="What is City office vacancy?", **kwargs):
    return Task(objective=question, entry_skill="research_market", trigger="chat", **kwargs)


def tool_context(store):
    run = RunRecord(task=task(), main_model="fixture-scripted-main", router_model="fixture-scripted-router")
    store.save_run(run)
    return ToolContext(Settings(), store, FixtureWebProvider(), "fixture", run.id, 10)




def test_city_chart_uses_two_compatible_periods_and_loaded_skills(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        run = Runner(Settings(), store).run(task("Chart City office vacancy trend"))
        assert run.status == "partial"
        chart = run.result.charts[0]
        assert chart.chart_type == "line"
        assert [point.value for point in chart.points] == [8.0, 7.5]
        assert [point.period_end for point in chart.points] == [date(2026, 3, 31), date(2026, 6, 30)]
        assert all(point.unit == "%" and point.definition for point in chart.points)
        assert all(store.get_observation(point.observation_id).value == point.value for point in chart.points)
        skills = [event for event in store.list_traces(run.id) if event.kind == "skill"]
        assert {event.name for event in skills} == {"research_market", "choose_visualization", "prepare_visualization_data"}
        assert {event.name: event.data["version"] for event in skills} == {name: default_registry().skills[name].version for name in ("research_market", "choose_visualization", "prepare_visualization_data")}
        assert all(event.data["instruction_hash"] for event in skills)
        assert store.list_traces(run.id)[-1].data["steps"] == 12


@pytest.mark.parametrize("question,history", [
    ("What about Canary Wharf?", [ChatMessage(role="user", content="Discuss London office demand")]),
    ("How do interest rates affect demand?", []),
    ("What does employment imply for office demand?", []),
])
def test_relevant_evidence_poor_questions_are_not_declined(tmp_path, question, history):
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store).run(task(question, history=history))
        assert run.status == "partial"
        assert "outside that scope" not in run.result.answer
        assert any(event.kind == "model" and event.outcome == "response" for event in store.list_traces(run.id))
        assert run.result.topic_gaps


def test_ambiguous_clarifies_and_clearly_unrelated_declines(tmp_path):
    with Store(tmp_path / "db") as store:
        runner = Runner(Settings(), store)
        ambiguous = runner.run(task("Tell me more"))
        assert "Which London office" in ambiguous.result.answer
        unrelated = runner.run(task("Give me a pasta recipe"))
        assert unrelated.status == "completed"
        assert "outside that scope" in unrelated.result.answer
        assert not any(event.kind == "model" for event in store.list_traces(unrelated.id))


@pytest.mark.parametrize("route", [ProviderError("synthetic router unavailable"), RouteReply(route=RouteChoice(choice="unrelated", probabilities={"in_scope": .2, "ambiguous": .3, "unrelated": .5, "greeting": 0}, confidence=.5), model="uncertain", provider="synthetic", usage=Usage())])
def test_router_failure_or_uncertainty_defers_to_main(tmp_path, route):
    model = ScriptModel([("finish", {"response_type": "insufficient_data", "message": "Relevant London office question; more evidence needed"})], route=route)
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store, model=model).run(task("What about Canary Wharf?"))
        assert model.count == 1
        assert run.status == "partial"
        assert "Relevant London office" in run.result.answer


def test_oversized_question_bypasses_compact_router(tmp_path):
    model = ScriptModel([("finish", {"response_type": "clarification", "message": "Which office metric should be investigated?"})], route=AssertionError("router must be bypassed"))
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store, model=model).run(task("London offices " + "x" * 2000))
        assert run.status == "partial"
        assert any(event.kind == "router" and event.outcome == "fallback" for event in store.list_traces(run.id))


@pytest.mark.parametrize("name,payload", [("shell", {"command": "echo arbitrary"}), ("load_skill", {"skill_id": "unknown"}), ("query_evidence", {"geography": "Mars"}), ("query_evidence", "not JSON")])
def test_unknown_bindings_skills_and_malformed_args_fail_and_stop(tmp_path, name, payload):
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store, model=ScriptModel([(name, payload)])).run(task())
        assert run.status == "stopped"
        assert "Repeated-action" in run.result.reason
        assert any(event.kind == "tool" and event.outcome == "failed" for event in store.list_traces(run.id))
        assert len(store.list_evidence("fixture")) == 0


def test_claims_must_cite_stored_same_mode_evidence(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        model = ScriptModel([("finish", {"claims": [{"kind": "fact", "text": "Unsupported rent rise", "evidence_ids": ["not-stored"]}]})])
        run = Runner(Settings(), store, model=model).run(task())
        assert run.status == "stopped"
        assert not run.result.claims
        assert "Unsupported rent rise" not in run.result.answer
        assert any("unknown evidence" in event.data.get("error", "") for event in store.list_traces(run.id))


def test_unrestricted_answer_and_model_chart_points_are_rejected(tmp_path):
    with Store(tmp_path / "db") as store:
        model = ScriptModel([("finish", {"answer": "Rents doubled", "charts": [{"value": 1000}], "message": "Rents doubled"})])
        run = Runner(Settings(), store, model=model).run(task())
        assert run.status == "stopped"
        assert "Rents doubled" not in run.result.answer
        assert run.result.charts == []


def test_step_and_elapsed_time_caps_preserve_terminal_runs(tmp_path):
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store).run(task(limits=Limits(max_steps=2)))
        assert run.status == "stopped" and "Step limit" in run.result.reason
        current = [0.0]
        class SlowModel(ScriptModel):
            def complete(self, *args, **kwargs):
                current[0] = 10
                return super().complete(*args, **kwargs)
        model = SlowModel([("query_evidence", {})])
        run = Runner(Settings(), store, model=model, clock=lambda: current[0]).run(task(limits=Limits(max_seconds=1)))
        assert run.status == "stopped" and "Elapsed-time" in run.result.reason
        assert store.get_run(run.id).finished_at is not None


def test_failed_provider_is_persisted(tmp_path):
    class FailedModel(ScriptModel):
        def complete(self, *args, **kwargs):
            raise ProviderError("synthetic model failure")
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store, model=FailedModel([])).run(task())
        assert run.status == "failed"
        assert run.result.reason == "synthetic model failure"
        assert store.get_run(run.id).status == "failed"
        assert store.list_traces(run.id)[-1].kind == "termination"


def test_exact_excerpts_and_chart_plan_boundary(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = tool_context(store)
        page = FixtureWebProvider().retrieve(str(fixture_batches()[0][0].url), 10)
        ctx.fetched[page.document.id] = page
        with pytest.raises(ValueError, match="occur exactly"):
            cache_evidence(ctx, CacheArgs(retrieved_id=page.document.id, excerpt="Synthetic vacancy 99%", native_geography="City", mapped_geography="City", category="vacancy"))
        with pytest.raises(ValueError, match="plan produced"):
            prepare_visualization_data(ctx, PrepareArgs(plan_id="model-invented"))


def test_incompatible_chart_is_table_and_sparse_history_is_honest(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        ctx = tool_context(store)
        city_q2 = fixture_batches()[1][2]
        plan = choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in city_q2], title="Different measurements", purpose="Keep definitions separate"))
        assert plan.plan.chart_type == "table"
        assert "definition" in plan.plan.caveats[0]
        payload = prepare_visualization_data(ctx, PrepareArgs(plan_id=plan.plan_id)).payload
        assert [point.metric for point in payload.points] == ["vacancy_rate", "availability_rate"]
        assert [point.value for point in payload.points] == [7.5, 9.2]
        sparse = choose_visualization(ctx, VisualizationArgs(observation_ids=[fixture_batches()[2][2][0].id], title="West End", purpose="Sparse history"))
        assert sparse.plan.chart_type == "table"
        assert "insufficient" in sparse.plan.caveats[0]
        empty = choose_visualization(ctx, VisualizationArgs(title="No history", purpose="No fabricated points"))
        assert prepare_visualization_data(ctx, PrepareArgs(plan_id=empty.plan_id)).payload.points == []


def test_cadence_conflicts_and_submarket_delta_direction(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        ctx = tool_context(store)
        city_q1 = fixture_batches()[0][2][0]
        city_q2 = fixture_batches()[1][2][0]
        west_end = fixture_batches()[2][2][0]
        trend = compare(ctx, CompareArgs(observation_ids=[city_q2.id, city_q1.id]))
        assert trend.compatible and trend.absolute_change == -.5 and trend.unit == "percentage points"
        assert trend.from_observation_id == city_q1.id
        comparison = compare(ctx, CompareArgs(observation_ids=[west_end.id, city_q2.id], comparison="submarkets"))
        assert comparison.absolute_change == 3.0 and comparison.from_observation_id == west_end.id
        year = Observation.model_validate({**city_q2.model_dump(exclude={"id"}), "period_start": "2025-01-01", "period_end": "2025-12-31"})
        store.ingest(fixture_batches()[1][0], [], [year])
        result = compare(ctx, CompareArgs(observation_ids=[year.id, city_q2.id]))
        assert not result.compatible and "cadence" in result.reason
        conflicting = Observation.model_validate({**city_q2.model_dump(exclude={"id"}), "value": 7.6})
        store.ingest(fixture_batches()[1][0], [], [conflicting])
        result = compare(ctx, CompareArgs(observation_ids=[city_q1.id, city_q2.id]))
        assert not result.compatible and "conflicting" in result.reason


def test_synthetic_normalization_requires_metric_period_and_matching_excerpt(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = tool_context(store)
        page = FixtureWebProvider().retrieve(str(fixture_batches()[0][0].url), 10)
        ctx.fetched[page.document.id] = page
        evidence = cache_evidence(ctx, CacheArgs(retrieved_id=page.document.id, excerpt=page.text, native_geography="City", mapped_geography="City", category="vacancy")).evidence
        fields = fixture_batches()[1][2][0].model_dump(exclude={"id", "mode", "publisher", "evidence_ids", "revision_of"})
        with pytest.raises(ValueError, match="approved synthetic"):
            normalize_observation(ctx, NormalizeArgs(evidence_id=evidence.id, **fields))
        fields = fixture_batches()[0][2][0].model_dump(exclude={"id", "mode", "publisher", "evidence_ids", "revision_of"})
        result = normalize_observation(ctx, NormalizeArgs(evidence_id=evidence.id, **fields))
        assert result.observation.value == 8.0
        assert store.get_evidence(evidence.id).validation_status == "validated"
