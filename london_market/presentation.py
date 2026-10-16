"""Business display labels; canonical evidence and run values stay unchanged."""

from datetime import date
from collections import Counter
from zoneinfo import ZoneInfo

from .domain import Claim, Evidence, Observation, RunResult, TraceEvent, stable_id
from .weekly import saved_claim_source_ids

SIGNAL_LABELS = {"bullish": "Bullish", "bearish": "Bearish", "neutral": "Neutral", None: "Signal unavailable"}

CLAIM_LABELS = {"fact": "Observed fact", "interpretation": "Interpretation", "forecast": "Source forecast", "watchpoint": "Watchpoint"}
VIEW_LABELS = {"Research": "Chat", "Weekly reports": "Weekly briefings", "Controls": "Controls", "Sources": "Sources and coverage", "Activity": "Run history"}
STATUS_LABELS = {"completed": "Completed", "partial": "Completed with gaps", "failed": "Failed", "stopped": "Stopped before completion", "running": "In progress"}
MONITORING_LABELS = {"candidate": "Awaiting monitoring review", "validated": "Reviewed for monitoring", "rejected": "Excluded from monitoring"}
SOURCE_CHECK_LABELS = {"success": "Source checked", "failed": "Check failed", "unchanged": "No new source content", "blocked": "Access blocked"}
MEASURE_LABELS = {
    "vacancy_rate": "Vacancy rate", "availability_rate": "Availability rate", "prime_rent": "Prime rent", "grade_a_rent": "Grade A rent",
    "asking_rent": "Asking rent", "achieved_rent": "Achieved rent", "take_up": "Office take-up", "take-up": "Office take-up", "reported_supply": "Source-reported supply",
    "development_completions": "Development completions", "under_construction": "Space under construction", "occupier_take_up_share": "Occupier-sector share of take-up",
    "take_up_grade_share": "Office-grade share of take-up", "supply_grade_share": "Office-grade share of supply", "take_up_share": "Share of take-up", "bank_rate": "Bank Rate", "hybrid_work_share": "Hybrid working share", "space_acquired": "Office space acquired",
    "vacancy": "Vacancy", "availability": "Availability", "vacancy_availability": "Vacancy and availability", "pipeline": "Development pipeline",
    "occupier_demand": "Occupier demand", "macro": "Economic context", "news": "Market news", "submarket_dynamics": "Submarket conditions",
}
ACTION_LABELS = {
    "research_market": "Research market evidence", "collect_market_evidence": "Collect market evidence", "prepare_weekly_briefing": "Prepare weekly briefing",
    "search": "Search market sources", "retrieve": "Retrieve source content", "monitoring_sources": "Review monitoring sources", "collect_monitored_source": "Check a monitoring source",
    "read_document": "Read source content", "query_evidence": "Inspect stored evidence", "market_inventory": "Review measures and coverage", "cache_evidence": "Save a source excerpt",
    "normalize_observation": "Validate a measurement", "compare": "Compare compatible measurements", "choose_visualization": "Choose an exhibit", "prepare_visualization_data": "Prepare exhibit data",
    "collect_fixture_evidence": "Load demonstration evidence", "load_skill": "Load research instructions", "finish": "Complete findings", "run": "Finish run",
    "revise_finish": "Revise rejected findings", "classify_signals": "Assess factual leasing signals",
}


def claim_label(claim: Claim) -> str:
    if claim.kind == "interpretation" and claim.interpretation_role:
        return {"source_view": "Source view", "landlord_analysis": "Landlord analysis"}[claim.interpretation_role]
    return CLAIM_LABELS[claim.kind]


def measure_label(value: str) -> str:
    return MEASURE_LABELS.get(value, value.replace("_", " "))


def action_label(value: str) -> str:
    return ACTION_LABELS.get(value, value.replace("_", " "))


def research_pipeline_rows(traces: list[TraceEvent]) -> list[dict[str, str]]:
    work = {
        'Source checks': ('monitoring_sources', 'collect_monitored_source', 'collect_fixture_evidence'),
        'Source research': ('search', 'retrieve', 'read_document'),
        'Evidence analysis': ('query_evidence', 'market_inventory', 'cache_evidence', 'normalize_observation', 'compare'),
        'Exhibits': ('choose_visualization', 'prepare_visualization_data'),
        'Finalization': ('finish', 'revise_finish', 'classify_signals'),
    }
    groups: dict[str, list[tuple[str, str]]] = {}
    for event in traces:
        if event.kind == 'signal' and event.outcome in ('classified', 'unavailable'):
            name, outcome = 'classify_signals', event.outcome
        elif event.kind == 'tool' and event.outcome in ('success', 'failed'):
            name, outcome = event.name, event.outcome
        else:
            continue
        if name == 'load_skill':
            continue
        label = next((label for label, names in work.items() if name in names), 'Other recorded work')
        groups.setdefault(label, []).append((name, 'returned' if outcome == 'success' else outcome))
    if not any(label in groups for label in ('Source checks', 'Source research', 'Evidence analysis', 'Exhibits')):
        return []
    return [
        {'Work': label,
         'Recorded actions': '; '.join(f'{action_label(name)} ({count})' for name, count in Counter(name for name, _ in events).items()),
         'Recorded outcomes': '; '.join(f'{count} {outcome}' for outcome, count in Counter(outcome for _, outcome in events).items())}
        for label, events in groups.items()
    ]


def runtime_skill_rows(traces: list[TraceEvent]) -> list[dict[str, str]]:
    loaded = dict.fromkeys((event.name, str(event.data.get('version', 'Not recorded'))) for event in traces if event.kind == 'skill' and event.outcome == 'loaded')
    return [{'Skill': action_label(name), 'Version': version} for name, version in loaded]


def date_label(value: date) -> str:
    return value.strftime("%d %b %Y").lstrip("0")


def count_label(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def weekly_window(result: RunResult) -> str:
    context = result.weekly.context
    return f"{date_label(context.window_start)} to {date_label(context.window_end)}"


def claim_timing(claim: Claim, result: RunResult) -> str:
    if result.weekly is None:
        return "Timing unavailable"
    source_ids = saved_claim_source_ids(claim, result)
    updates = [item for item in result.weekly.evidence_updates if item.evidence_id in source_ids]
    if any(item.event_in_window for item in updates):
        return "Event this week"
    if any(item.publication_in_window for item in updates):
        return "New publication this week"
    if any(item.first_ingested_in_window for item in updates):
        return "Evidence added this week"
    return "Retained context" if updates else "Timing unavailable"


def market_evidence_groups(result: RunResult, unavailable_ids: set[str] | None = None) -> list[tuple[str | None, str, list[int]]]:
    unavailable_ids = unavailable_ids or set()
    groups = {signal: [] for signal in SIGNAL_LABELS}
    for index, claim in enumerate(result.claims):
        if claim.kind == "fact":
            signal = claim.signal if result.status not in ("failed", "stopped") and not unavailable_ids.intersection(saved_claim_source_ids(claim, result)) else None
            groups[signal].append(index)
    timing_order = {"Event this week": 0, "New publication this week": 1, "Evidence added this week": 2, "Retained context": 3, "Timing unavailable": 4}
    return [(signal, label, sorted(groups[signal], key=lambda index: timing_order[claim_timing(result.claims[index], result)])) for signal, label in SIGNAL_LABELS.items() if groups[signal] or signal is not None]


def partition_market_evidence(result: RunResult, indexes: list[int]) -> tuple[list[int], list[int]]:
    if result.weekly is None:
        return indexes, []
    current, retained = [], []
    for index in indexes:
        target = current if claim_timing(result.claims[index], result) in ("Event this week", "New publication this week") else retained
        target.append(index)
    return current, retained


def claim_metadata(claim: Claim, result: RunResult, evidence: list[Evidence], observations: list[Observation]) -> list[str]:
    source_ids = saved_claim_source_ids(claim, result)
    sources = [item for item in evidence if item.id in source_ids]
    updates = [item for item in result.weekly.evidence_updates if item.evidence_id in source_ids] if result.weekly else []
    metadata = [claim_timing(claim, result)] if result.weekly else []
    published = sorted({item.published_at for item in (updates if result.weekly else sources) if item.published_at})
    metadata.append("Published " + ", ".join(date_label(value) for value in published) if published else "Publication date unknown")
    events = sorted({item.event_date for item in (updates if result.weekly else sources) if item.event_date})
    if events:
        metadata.append("Event " + ", ".join(date_label(value) for value in events))
    periods = {(period.period_start, period.period_end) for item in updates for period in item.reference_periods} if result.weekly else {(item.period_start, item.period_end) for item in observations if set(item.evidence_ids).intersection(source_ids)}
    for chart in result.charts:
        if stable_id("chart", chart.model_dump(mode="json")) in claim.chart_ids:
            points = chart.points + [point for pair in chart.pairs for point in (pair.x, pair.y)] + ([chart.denominator] if chart.denominator else [])
            periods.update((point.period_start, point.period_end) for point in points)
    if periods:
        metadata.append("Reference period " + "; ".join(f"{date_label(start)} to {date_label(end)}" for start, end in sorted(periods)))
    return metadata


def weekly_summary_copy(result: RunResult) -> dict[str, str]:
    summary = result.weekly
    context = summary.context
    cutoff = context.cutoff_at.astimezone(ZoneInfo(context.timezone)).strftime("%d %b %Y %H:%M")
    publications = {item.document_id for item in summary.evidence_updates if item.publication_in_window}
    events = {item.evidence_id for item in summary.evidence_updates if item.event_in_window}
    checks = len(summary.source_attempts)
    failures = sum(item.status in ("failed", "blocked") for item in summary.source_attempts)
    return {
        "window": weekly_window(result),
        "cutoff": f"Research cutoff {cutoff} {context.timezone}",
        "prior": f"Compared with the briefing dated {date_label(context.prior_briefing_date)}." if context.prior_briefing_date else "First briefing in this history. No prior weekly briefing is available.",
        "coverage": f"{count_label(summary.cited_publication_count, 'cited publication')} · {count_label(summary.cited_excerpt_count, 'excerpt')} · {count_label(summary.cited_domain_count, 'domain')}",
        "concentration": "Publication and excerpt counts describe the evidence used. Several excerpts from one publication are not independent sources.",
        "activity": f"{count_label(len(publications), 'cited publication')} dated in this window · {count_label(len(events), 'cited excerpt')} with dated events in this window. Retained evidence does not establish unchanged market conditions.",
        "status": f"{STATUS_LABELS[result.status]} · {count_label(checks, 'source check')}, {failures} failed or blocked." if checks else f"{STATUS_LABELS[result.status]} · No source checks recorded.",
        "measurements": "No comparable weekly measurements are available." if not summary.week_over_week else f"{count_label(len(summary.week_over_week), 'supported weekly comparison')}. Source periods and definitions are shown below.",
    }


def weekly_indicator_rows(result: RunResult, observations: list[Observation]) -> list[dict[str, str]]:
    by_id = {item.id: item for item in observations}
    rows = []
    for indicator in result.weekly.week_over_week:
        baseline, endpoint = by_id.get(indicator.from_observation_id), by_id.get(indicator.to_observation_id)
        scope = indicator.native_geography
        if scope != indicator.mapped_geography:
            scope += f" ({indicator.mapped_geography})"
        rows.append({"Measure": measure_label(indicator.metric), "Geography": scope,
                     "Previous period": f"{date_label(indicator.baseline_period_start)} to {date_label(indicator.baseline_period_end)}",
                     "Previous value": f"{baseline.value:g} {baseline.unit}" if baseline else "Value unavailable",
                     "Latest period": f"{date_label(indicator.endpoint_period_start)} to {date_label(indicator.endpoint_period_end)}",
                     "Latest value": f"{endpoint.value:g} {endpoint.unit}" if endpoint else "Value unavailable",
                     "Change": f"{indicator.absolute_change:+g} {indicator.unit}", "Publisher": indicator.publisher,
                     "Definition": indicator.definition})
    return rows


def weekly_indicator_groups(result: RunResult, observations: list[Observation]) -> list[tuple[str, list[dict[str, str]]]]:
    groups = {"London office indicators": [], "UK economic context": []}
    for indicator, row in zip(result.weekly.week_over_week, weekly_indicator_rows(result, observations), strict=True):
        uk = indicator.native_geography.strip().lower() in ("uk", "united kingdom", "great britain", "national", "uk national")
        groups["UK economic context" if uk else "London office indicators"].append(row)
    return [(label, rows) for label, rows in groups.items() if rows]
