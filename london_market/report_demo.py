"""Opt-in fictional history and scripted report, using the shared Runner."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

from .domain import Document, Evidence, Observation, Task, Usage, Limits
from .fixtures import scripted_coverage
from .sources import fixture_analysis_sources
from .providers import ModelReply, ToolCall
from .runner import Runner

PERIODS = [(date(2025, 7, 1), date(2025, 9, 30)), (date(2025, 10, 1), date(2025, 12, 31)), (date(2026, 1, 1), date(2026, 3, 31)), (date(2026, 4, 1), date(2026, 6, 30))]
SERIES = {
    "vacancy_rate": ("%", "Unoccupied office space as a share of office stock", "vacancy", {"City": [8, 7.8, 7.5, 7], "West End": [5, 4.8, 4.5, 4.2]}),
    "prime_asking_rent": ("GBP per sq ft per year", "Quoted annual prime office headline rent; excludes incentives", "rents", {"City": [90, 92, 95, 98], "West End": [140, 143, 147, 150]}),
    "take_up": ("million sq ft", "Office space leased during the reference quarter", "take-up", {"City": [.8, .9, .95, 1.1], "West End": [.4, .45, .5, .55]}),
    "available_space": ("million sq ft", "Office space marketed as available at quarter end", "availability", {"City": [7, 6.8, 6.4, 6], "West End": [2.8, 2.7, 2.6, 2.5]}),
}


def seed_report_demo(store):
    retrieved = datetime(2026, 10, 9, 9, tzinfo=timezone.utc)
    for metric, (unit, definition, category, series) in SERIES.items():
        for geography, values in series.items():
            for (start, end), value in zip(PERIODS, values):
                excerpt = f"SYNTHETIC: {geography} {metric} was {value:g} {unit} for {start} to {end}. Definition: {definition}."
                document = Document(mode="fixture", synthetic=True, url=f"https://professional-demo.example.invalid/{metric}/{geography.replace(' ', '-')}/{end}", title=f"Fictional {geography} {metric} {end}", publisher="Fictional London Office Observatory", content_hash=hashlib.sha256(excerpt.encode()).hexdigest(), published_at=date(2026, 7, 15), retrieved_at=retrieved)
                evidence = Evidence(document_id=document.id, mode="fixture", source_url=document.url, excerpt=excerpt, published_at=document.published_at, retrieved_at=retrieved, native_geography=geography, mapped_geography=geography, category=category, validation_status="validated", validation_reason="Controlled fictional report-demo measurement; no live market authority")
                observation = Observation(mode="fixture", metric=metric, value=value, unit=unit, publisher=document.publisher, definition=definition, category=category, native_geography=geography, mapped_geography=geography, period_start=start, period_end=end, evidence_ids=[evidence.id])
                store.ingest(document, [evidence], [observation])
    for document, evidence, observations in fixture_analysis_sources():
        store.ingest(document, evidence, observations)
    examples = [("City", "On 18 June 2026, fictional occupier Meridian Analytics leased 42,000 sq ft at fictional building Lantern House, City, in this synthetic scenario. The fictional transaction concerns refurbished offices and does not establish a market-wide refurbishment premium."), ("West End", "On 24 June 2026, fictional occupier Orchard Design agreed a fictional 18,000 sq ft pre-let at fictional building Juniper Court, West End. This synthetic example illustrates future supply commitments and does not measure aggregate pre-let demand.")]
    for geography, text in examples:
        excerpt = "SYNTHETIC FICTIONAL EXAMPLE. " + text
        event_date = date(2026, 6, 18 if geography == "City" else 24)
        document = Document(mode="fixture", synthetic=True, url=f"https://professional-demo.example.invalid/examples/{geography.replace(' ', '-')}", title="Fictional named leasing example", publisher="Fictional London Office Observatory", content_hash=hashlib.sha256(excerpt.encode()).hexdigest(), published_at=date(2026, 7, 15), retrieved_at=retrieved)
        evidence = Evidence(document_id=document.id, mode="fixture", source_url=document.url, excerpt=excerpt, event_date=event_date, published_at=document.published_at, retrieved_at=retrieved, native_geography=geography, mapped_geography=geography, category="leasing_example")
        store.ingest(document, [evidence], [])


class ReportDemoModel:
    def __init__(self):
        self.phase = 0

    def complete(self, messages, tools, max_output_tokens, timeout):
        names, outputs = {}, {}
        for message in messages:
            for call in message.get("tool_calls", []):
                names[call["id"]] = call["function"]["name"]
            if message.get("role") == "tool":
                outputs[message["tool_call_id"]] = (names[message["tool_call_id"]], json.loads(message["content"]))
        def reply(actions):
            phase = self.phase
            self.phase += 1
            return ModelReply(model="fixture-professional-report-script", provider="synthetic", usage=Usage(), calls=[ToolCall(id=f"demo-{phase}-{index}", name=name, arguments=json.dumps(arguments)) for index, (name, arguments) in enumerate(actions)])
        if self.phase == 0:
            pinned = next(line.removeprefix("Pinned weekly context: ") for message in messages for line in (message.get("content") or "").splitlines() if line.startswith("Pinned weekly context: "))
            context = json.loads(pinned)
            window_query = ("query_evidence", {"published_since": context["window_start"], "published_until": context["window_end"], "limit": 12})
            return reply([window_query] + [("query_evidence", {"metric": metric, "limit": 12}) for metric in [*SERIES, "leasing_example"]] + [("query_evidence", {"evidence_ids": [evidence[0].id for _, evidence, _ in fixture_analysis_sources()]})] + [("load_skill", {"skill_id": "choose_visualization"}), ("load_skill", {"skill_id": "prepare_visualization_data"})])
        queries = [value for name, value in outputs.values() if name == "query_evidence"]
        observations = [item for query in queries for item in query["observations"]]
        evidence = {item["id"]: item for query in queries for item in query["evidence"]}
        if self.phase == 1:
            actions = []
            for metric, title, purpose, preferred in [
                ("vacancy_rate", "Vacancy declined in both synthetic submarkets", "Compare the direction and level of compatible City and West End vacancy histories", "trend"),
                ("prime_asking_rent", "Prime asking rents rose across four synthetic quarters", "Compare quoted rents across submarkets without claiming achieved rents or net effective values", "trend"),
                ("take_up", "City leasing volume exceeded West End in synthetic Q2", "Compare latest-quarter leased space using the same definition and period", "submarkets"),
                ("available_space", "City retained more marketed space in synthetic Q2", "Compare available space at quarter end; do not confuse it with construction pipeline", "submarkets"),
            ]:
                selected = [item for item in observations if item["metric"] == metric and (preferred == "trend" or item["period_end"] == "2026-06-30")]
                actions.append(("choose_visualization", {"observation_ids": [item["id"] for item in selected], "title": title, "purpose": purpose, "preferred": preferred}))
            vacancy = sorted([item for item in observations if item["metric"] == "vacancy_rate" and item["mapped_geography"] == "City"], key=lambda item: item["period_start"])
            take_up = sorted([item for item in observations if item["metric"] == "take_up" and item["mapped_geography"] == "City"], key=lambda item: item["period_start"])
            actions += [
                ("choose_visualization", {"pairs": [{"x_observation_id": x["id"], "y_observation_id": y["id"]} for x, y in zip(vacancy, take_up)], "preferred": "relation", "title": "Synthetic vacancy and leasing association", "purpose": "Compare aligned quarterly measurements descriptively; no causal inference"}),
                ("choose_visualization", {"observation_ids": [item["id"] for item in take_up], "preferred": "total", "title": "Synthetic displayed leasing flows", "purpose": "Calculate the sum of displayed nonoverlapping quarters, not net stock change"}),
            ]
            for metric in ("vacancy_rate", "prime_asking_rent"):
                selected = sorted([item for item in observations if item["metric"] == metric and item["mapped_geography"] == "City"], key=lambda item: item["period_start"])
                actions.append(("compare", {"observation_ids": [selected[0]["id"], selected[-1]["id"]]}))
            return reply(actions)
        if self.phase == 2:
            return reply([("prepare_visualization_data", {"plan_id": value["plan_id"]}) for name, value in outputs.values() if name == "choose_visualization"])
        charts = [value for name, value in outputs.values() if name == "prepare_visualization_data"]
        comparisons = [value for name, value in outputs.values() if name == "compare"]
        claims = []
        for comparison in comparisons:
            end = next(item for item in observations if item["id"] == comparison["to_observation_id"])
            claims.append({"kind": "fact", "text": f"SYNTHETIC City {end['metric'].replace('_', ' ')} {'fell' if comparison['absolute_change'] < 0 else 'rose' if comparison['absolute_change'] > 0 else 'was unchanged'} by {abs(comparison['absolute_change']):g} {comparison['unit'].removesuffix('s') if abs(comparison['absolute_change']) == 1 else comparison['unit']} from {comparison['baseline_period_start']}–{comparison['baseline_period_end']} to {comparison['endpoint_period_start']}–{comparison['endpoint_period_end']}. Relative growth was {comparison['relative_growth_percent']:g}% of the nonzero baseline. These are endpoint changes across four quarterly observations, not current market findings.", "evidence_ids": comparison["evidence_ids"]})
        latest = sorted([item for item in observations if item["metric"] == "take_up" and item["period_end"] == "2026-06-30"], key=lambda item: item["mapped_geography"])
        claims.append({"kind": "fact", "text": "SYNTHETIC Q2 2026 take-up was " + "; ".join(f"{item['mapped_geography']} {item['value']:g} {item['unit']}" for item in latest) + ". The comparison measures leased space, not deal counts or demand quality.", "evidence_ids": [eid for item in latest for eid in item["evidence_ids"]]})
        for item in evidence.values():
            if item["category"] == "leasing_example":
                claims.append({"kind": "fact", "text": item["excerpt"], "evidence_ids": [item["id"]]})
        for _, supported, _ in fixture_analysis_sources():
            item = supported[0]
            analysis = {"kind": "delivery_uncertainty", "forecast_evidence_id": item.id} if item.category == "pipeline" else {"kind": "lease_commitment", "transaction_evidence_id": item.id}
            claims.append({"kind": "interpretation", "interpretation_role": "landlord_analysis", "analysis": analysis})
        claims += [{"kind": "watchpoint", "text": "Obtain later comparable releases, achieved rents, incentives and completion schedules before applying the scenario to real London offices.", "evidence_ids": comparisons[0]["evidence_ids"]}, {"kind": "watchpoint", "text": "All evidence is synthetic and historical. Seek current local releases for Canary Wharf, Midtown and Fringe, and wider London before assessing those areas.", "evidence_ids": comparisons[0]["evidence_ids"]}]
        for comparison in comparisons:
            baseline = next(item for item in observations if item["id"] == comparison["from_observation_id"])
            endpoint = next(item for item in observations if item["id"] == comparison["to_observation_id"])
            claims.append({"kind": "fact", "text": f"SYNTHETIC City {endpoint['metric'].replace('_', ' ')} {'fell' if comparison['absolute_change'] < 0 else 'rose' if comparison['absolute_change'] > 0 else 'was unchanged'} from {baseline['value']:g} to {endpoint['value']:g} {endpoint['unit']} between quarters ending {baseline['period_end']} and {endpoint['period_end']}. This fictional history provides leasing context only.", "evidence_ids": comparison["evidence_ids"]})
        sections = [{"kind": "executive_findings", "claim_indexes": [9, 10]}, {"kind": "key_trends", "claim_indexes": [0, 1], "chart_indexes": [0, 1]}, {"kind": "submarket_dynamics", "claim_indexes": [2], "chart_indexes": [2, 3]}, {"kind": "concrete_examples", "claim_indexes": [3, 4]}, {"kind": "landlord_implications", "claim_indexes": [5, 6]}, {"kind": "outlook", "claim_indexes": [7]}, {"kind": "sources_and_gaps", "claim_indexes": [8]}]
        coverage = scripted_coverage(claims, observations, [], evidence=list(evidence.values()))
        return reply([("finish", {"coverage": coverage, "claims": claims, "chart_ids": [chart["chart_id"] for chart in charts], "sections": sections, "limitations": ["Fictional history and examples; no external requests or live agency", "Prime rents are asking rents; supply exhibit is marketed availability, not the construction pipeline"], "topic_gaps": ["Achieved rents and incentives", "Grade A premiums", "Construction, refurbishment and aggregate pre-let pipeline", "Macroeconomic and occupier demand evidence"]})])


def generate_report_demo(settings, store, *, max_steps=None):
    if settings.mode != "fixture":
        raise ValueError("report-demo requires fixture mode")
    seed_report_demo(store)
    task = Task(objective="Prepare a sourced professional London office report from the synthetic report-demo history", entry_skill="prepare_weekly_briefing", mode="fixture", trigger="fixture", briefing_date=date(2026, 10, 9), report_week_start=date(2026, 10, 5), limits=Limits.model_validate({**settings.limits.model_dump(), **({"max_steps": max_steps} if max_steps is not None else {})}))
    run = Runner(settings, store, model=ReportDemoModel()).run(task)
    report_id = store.save_report(run.id, task.report_week_start, run.result) if run.status in ("completed", "partial") else None
    return run, report_id
