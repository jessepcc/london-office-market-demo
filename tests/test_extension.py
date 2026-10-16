"""A local fixture-only skill extends the shared runner through registration."""

import json

from pydantic import Field

from london_market.config import Settings
from london_market.domain import Geography, Model, RunResult, Task, Usage
from london_market.fixtures import load_fixtures
from london_market.providers import ModelReply, ToolCall
from london_market.runner import Runner
from london_market.skills import SkillSpec, default_registry
from london_market.store import Store
from london_market.tools import ToolBinding


class CoverageInput(Model):
    geography: Geography


class CoverageOutput(Model):
    geography: Geography
    observations: int = Field(ge=0)
    statement: str


def fixture_coverage(ctx, args: CoverageInput) -> CoverageOutput:
    count = sum(item.mapped_geography == args.geography for item in ctx.store.list_observations(ctx.mode))
    return CoverageOutput(geography=args.geography, observations=count, statement=f"Synthetic coverage check inspected {count} {args.geography} observations.")


class ExtensionModel:
    def __init__(self, malformed=False):
        self.phase = 0
        self.malformed = malformed

    def complete(self, messages, tools, max_output_tokens, timeout):
        names = {tool["function"]["name"] for tool in tools}
        if self.phase == 0:
            assert "fixture_coverage" not in names
            name, args = "load_skill", {"skill_id": "fixture_coverage_review"}
        elif self.phase == 1 or self.malformed:
            assert "fixture_coverage" in names
            name, args = "fixture_coverage", {"geography": "Mars" if self.malformed else "City"}
        else:
            output = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "tool"))
            assert output["observations"] == 3
            name, args = "finish", {"claims": [{"kind": "watchpoint", "text": "Synthetic coverage check inspected City observations.", "evidence_ids": []}], "limitations": ["Fixture-only extension proof"]}
        self.phase += 1
        return ModelReply(model="fixture-extension-script", provider="synthetic", calls=[ToolCall(id=f"extension-{self.phase}", name=name, arguments=json.dumps(args))], usage=Usage())


def extension_registry(tmp_path):
    instructions = tmp_path / "SKILL.md"
    instructions.write_text("# Synthetic coverage review\nCall fixture_coverage for City and inspect the typed result. This is a fixture-only extension demonstration, not live market evidence.\n")
    registry = default_registry()
    binding = ToolBinding(name="fixture_coverage", description="Inspect controlled synthetic observations by geography", input_schema=CoverageInput, output_schema=CoverageOutput, handler=fixture_coverage, fixture_only=True)
    registry.register(SkillSpec(id="fixture_coverage_review", description="Fixture-only extension proof", version="fixture-1", instructions_path=instructions, input_schema=Task, output_schema=RunResult, tool_bindings=(binding,)))
    return registry


def test_registered_extension_executes_through_unchanged_runner(tmp_path):
    registry = extension_registry(tmp_path)
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        run = Runner(Settings(), store, registry=registry, model=ExtensionModel()).run(Task(objective="Inspect synthetic City coverage through the registered extension", entry_skill="research_market", trigger="manual"))
        assert run.status == "partial"
        assert run.main_model == "fixture-extension-script"
        assert "inspected City observations" in run.result.answer
        assert next(event for event in store.list_traces(run.id) if event.name == "fixture_coverage" and event.outcome == "success").data["output"]["observations"] == 3
        traces = store.list_traces(run.id)
        selected = next(trace for trace in traces if trace.kind == "skill" and trace.name == "fixture_coverage_review")
        assert selected.data["version"] == "fixture-1" and selected.data["instruction_hash"]
        tool = next(trace for trace in traces if trace.kind == "tool" and trace.name == "fixture_coverage")
        assert tool.outcome == "success" and tool.data["output"]["observations"] == 3
        assert tool.data["arguments"] == {"geography": "City"}
    assert "fixture_coverage_review" not in default_registry().skills
    assert "fixture_coverage" not in registry.allowed_tools({"fixture_coverage_review"}, "live")


def test_extension_uses_existing_validation_and_repetition_limits(tmp_path):
    with Store(tmp_path / "db") as store:
        run = Runner(Settings(), store, registry=extension_registry(tmp_path), model=ExtensionModel(malformed=True)).run(Task(objective="Malformed synthetic extension input", entry_skill="research_market", trigger="manual"))
        assert run.status == "stopped"
        assert "Repeated-action limit" in run.result.reason
        failures = [trace for trace in store.list_traces(run.id) if trace.kind == "tool" and trace.name == "fixture_coverage"]
        assert len(failures) == 2 and all(trace.outcome == "failed" for trace in failures)
        assert not run.result.claims
