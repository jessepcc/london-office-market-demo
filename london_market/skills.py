"""Small explicit runtime-skill registry; instruction files cannot grant tools."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .domain import RunResult, Task
from .tools import PayloadResult, PrepareArgs, ToolBinding, VisualizationArgs, PlanResult, bindings


@dataclass(frozen=True)
class SkillSpec:
    id: str
    description: str
    version: str
    instructions_path: Path
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    tool_bindings: tuple[ToolBinding, ...]


class SkillRegistry:
    def __init__(self, skills: list[SkillSpec]):
        self.skills: dict[str, SkillSpec] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: SkillSpec) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,60}", skill.id) or skill.id in self.skills:
            raise ValueError("skill ID must be unique and use lowercase identifiers")
        if not skill.instructions_path.is_file() or skill.instructions_path.stat().st_size > 12000:
            raise ValueError("skill instructions must be an explicit bounded local file")
        for existing in self.skills.values():
            prior = {binding.name: binding for binding in existing.tool_bindings}
            for binding in skill.tool_bindings:
                if binding.name in prior and binding != prior[binding.name]:
                    raise ValueError("one tool name cannot grant different Python bindings")
        self.skills[skill.id] = skill

    def catalog(self) -> list[dict[str, str]]:
        return [{"id": skill.id, "description": skill.description, "version": skill.version} for skill in self.skills.values()]

    def load(self, skill_id: str) -> dict[str, Any]:
        skill = self.skills.get(skill_id)
        if skill is None:
            raise ValueError("unknown registered skill ID")
        instructions = skill.instructions_path.read_text()
        return {"id": skill.id, "version": skill.version, "instruction_hash": hashlib.sha256(instructions.encode()).hexdigest(), "instructions": instructions, "approved_tools": [binding.name for binding in skill.tool_bindings]}

    def allowed_tools(self, loaded: set[str], mode: str) -> dict[str, ToolBinding]:
        return {binding.name: binding for skill_id in loaded for binding in self.skills[skill_id].tool_bindings if mode == "fixture" or not binding.fixture_only}


def default_registry() -> SkillRegistry:
    tool_map = bindings()
    root = Path(__file__).parent / "runtime_skills"
    definitions = [
        ("research_market", "29", "Independent London office and relevant macro research, without requiring reports", Task, RunResult, ("market_inventory", "search", "retrieve", "read_document", "query_evidence", "cache_evidence", "compare", "choose_visualization", "prepare_visualization_data")),
        ("collect_market_evidence", "6", "Choose monitored source checks and discover missing evidence; normalize approved definitions only", Task, RunResult, ("market_inventory", "monitoring_sources", "collect_monitored_source", "search", "retrieve", "read_document", "query_evidence", "cache_evidence", "normalize_observation", "collect_fixture_evidence")),
        ("prepare_weekly_briefing", "45", "Landlord briefing with current-source checks, fact signals, weekly novelty and supported dated comparisons", Task, RunResult, ("market_inventory", "query_evidence", "search", "retrieve", "read_document", "cache_evidence", "compare", "choose_visualization", "prepare_visualization_data")),
        ("audit_market_data", "3", "Data quality, histories, conflicts, dimensions and measurement comparability", Task, RunResult, ("market_inventory", "query_evidence", "compare")),
        ("analyze_market_trends", "5", "Deterministic dated endpoint changes with percentage-point and relative-growth semantics", Task, RunResult, ("query_evidence", "compare")),
        ("choose_visualization", "8", "Choose an evidence-supported trend, comparison, table or insufficient-data state", VisualizationArgs, PlanResult, ("query_evidence", "choose_visualization")),
        ("prepare_visualization_data", "5", "Produce chart values and provenance through deterministic approved tools", PrepareArgs, PayloadResult, ("prepare_visualization_data",)),
        ("analyze_office_rents", "6", "Prime/Grade A rents, incentives, asking versus achieved and net effective terms", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_leasing_activity", "6", "Vacancy, availability, take-up, absorption and demand with distinct stocks and flows", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_supply_pipeline", "5", "Development/refurbishment status, completion timing, prelets and competing supply", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_submarkets", "5", "City, West End, Canary Wharf, Midtown and Fringe, Wider London coverage and boundaries", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_macro_drivers", "7", "Rates, inflation, economic activity and employment through a London-office mechanism", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_occupier_demand", "5", "Flight to quality, ESG, hybrid work, occupier sectors and evidence of leasing preferences", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
        ("analyze_market_events", "5", "Dated material London office news, status, actors, original sources and follow-ups", Task, RunResult, ("search", "retrieve", "query_evidence", "cache_evidence", "compare")),
    ]
    return SkillRegistry([SkillSpec(skill_id, description, version, root / skill_id / "SKILL.md", input_schema, output_schema, tuple(tool_map[name] for name in names)) for skill_id, version, description, input_schema, output_schema, names in definitions])
