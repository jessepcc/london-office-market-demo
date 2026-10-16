"""Deterministic briefing chronology and saved provenance summaries."""

from collections import defaultdict
from dataclasses import dataclass
import re
from datetime import date, datetime, time, timedelta
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from .domain import SourceReportedChange, Claim, RunResult, Task, WeeklyContext, WeeklyEvidenceUpdate, WeeklyIndicator, WeeklyReferencePeriod, WeeklySearch, WeeklySourceAttempt, WeeklySourceCount, WeeklySummary, stable_id
from .providers import ProviderError, SignalCard, signal_payload
from .sources import MONITORED_SOURCES, reviewed_source_qualifications
from .store import Store


def weekly_context(store: Store, task: Task, timezone: str, now: datetime) -> WeeklyContext:
    day = task.briefing_date or now.astimezone(ZoneInfo(timezone)).date()
    week = task.report_week_start or day - timedelta(days=day.weekday())
    previous = next((report for report in store.list_reports(task.mode) if date.fromisoformat(report["week_start"]) < week), None)
    return WeeklyContext(briefing_date=day, report_week_start=week, window_start=day - timedelta(days=6), window_end=day, cutoff_at=now, timezone=timezone,
                         prior_report_id=previous["id"] if previous else None, prior_run_id=previous["run_id"] if previous else None,
                         prior_briefing_date=previous["result"].briefing_date if previous else None,
                         prior_evidence_ids=sorted(set(previous["result"].evidence_ids) | {eid for claim in previous["result"].claims for eid in claim.evidence_ids}) if previous else [])


def claim_source_ids(claim: Claim, charts_by_id: dict | None = None) -> set[str]:
    charts = charts_by_id or {}
    return set(claim.evidence_ids) | {eid for cid in claim.chart_ids if cid in charts for eid in charts[cid].source_refs}


def saved_claim_source_ids(claim: Claim, result: RunResult) -> set[str]:
    charts = {stable_id("chart", chart.model_dump(mode="json")): chart for chart in result.charts}
    return claim_source_ids(claim, charts)


@dataclass(frozen=True)
class SignalBatchPlan:
    batches: list[list[tuple[int, SignalCard]]]
    oversized: list[tuple[int, SignalCard]]


def plan_signal_batches(claims: list[Claim], store: Store, mode: str, model: str, *, charts_by_id: dict | None = None, cached_fingerprints: set[str] | None = None) -> SignalBatchPlan:
    pending = [(index, signal_card(claim, store, mode, charts_by_id=charts_by_id)) for index, claim in enumerate(claims) if claim.kind == 'fact']
    unique = {}
    for index, card in pending:
        if card.fingerprint not in (cached_fingerprints or set()):
            unique.setdefault(card.fingerprint, (index, card))
    pending = list(unique.values())
    batches, oversized = [], []
    while pending:
        batch = pending[:3]
        while batch:
            try:
                signal_payload([card for _, card in batch], model)
                break
            except ProviderError:
                batch = batch[:-1]
        if batch:
            batches.append(batch)
            del pending[:len(batch)]
        else:
            oversized.append(pending.pop(0))
    return SignalBatchPlan(batches, oversized)


def signal_card(claim: Claim, store: Store, mode: str, *, charts_by_id: dict | None = None) -> SignalCard:
    evidence_ids = claim_source_ids(claim, charts_by_id)
    qualifications = set()
    measurements = defaultdict(set)
    sources = defaultdict(set)
    for eid in evidence_ids:
        item = store.get_evidence(eid, mode)
        sources[(item.published_at, item.event_date, item.validation_status)].add(item.native_geography)
        for qualification in reviewed_source_qualifications(item):
            qualifications.add(qualification.support_span if isinstance(qualification, SourceReportedChange) else qualification.risk_span)
        if item.validation_reason:
            qualifications.update(part for part in re.split(r" Source (?:qualifications|date qualification)\. ", item.validation_reason)[1:] if part)
    for (publication, event, status), geographies in sources.items():
        qualifications.add(f"Source geography: {'; '.join(sorted(geographies))}. Publication: {publication or 'unknown'}. Event: {event or 'unknown'}. Status: {status}.")
    for item in store.list_observations(mode):
        if set(item.evidence_ids).intersection(evidence_ids):
            measurements[(item.metric, item.native_geography, item.definition)].add((item.period_start, item.period_end))
    shared_periods = defaultdict(list)
    for (metric, geography, definition), periods in sorted(measurements.items()):
        shared_periods[(geography, tuple(sorted(periods)))].append((metric, definition))
    for (geography, periods), definitions in shared_periods.items():
        dates = "; ".join(f"{start} to {end}" for start, end in periods)
        measures = "; ".join(f"{metric}: {definition}" for metric, definition in definitions)
        qualifications.add(f"Measurement: {geography}; {measures}. Reference periods: {dates}.")
    fingerprint = stable_id("card", claim.kind, claim.text, claim.headline, sorted(evidence_ids), sorted(claim.chart_ids))
    text = f"{claim.headline}\n{claim.text}" if claim.headline else claim.text
    return SignalCard(fingerprint=fingerprint, text=text, qualifications=sorted(qualifications))


def weekly_summary(store: Store, mode: str, run_id: str, context: WeeklyContext, claims: list[Claim], cited_ids: list[str], now: datetime) -> WeeklySummary:
    context = context.model_copy(update={"cutoff_at": now})
    zone = ZoneInfo(context.timezone)
    start = datetime.combine(context.window_start, time.min, tzinfo=zone)
    end = min(now, datetime.combine(context.window_end + timedelta(days=1), time.min, tzinfo=zone))
    def in_window(day: date | None) -> bool | None:
        return context.window_start <= day <= context.window_end if day is not None else None

    observations = store.list_observations(mode)
    updates = []
    sources = defaultdict(lambda: {"documents": set(), "evidence": set()})
    for eid in sorted(set(cited_ids)):
        item = store.get_evidence(eid, mode)
        if item is None:
            continue
        document = store.get_document(item.document_id, mode)
        domain = urlsplit(str(item.source_url)).hostname or "Unknown"
        contract = next((source for source in MONITORED_SOURCES.values() if source.url.rstrip("/") == str(item.source_url).rstrip("/")), None)
        publisher = contract.publisher if contract else document.publisher
        sources[(publisher, domain)]["documents"].add(str(item.source_url).rstrip("/"))
        sources[(publisher, domain)]["evidence"].add(item.id)
        updates.append(WeeklyEvidenceUpdate(evidence_id=item.id, document_id=item.document_id, published_at=item.published_at, event_date=item.event_date, retrieved_at=item.retrieved_at,
                                           first_ingested_at=item.first_ingested_at, previously_cited=item.id in context.prior_evidence_ids if context.prior_report_id else None,
                                           publication_in_window=in_window(item.published_at), event_in_window=in_window(item.event_date),
                                           first_ingested_in_window=start <= item.first_ingested_at <= end if item.first_ingested_at else None,
                                           reference_periods=[WeeklyReferencePeriod(observation_id=obs.id, period_start=obs.period_start, period_end=obs.period_end, revision_of=obs.revision_of) for obs in observations if item.id in obs.evidence_ids]))
    relevant_runs = [record for record in store.list_runs(mode, limit=100) if record.id == run_id or (record.task.entry_skill == "collect_market_evidence" and start <= record.started_at <= end)]
    attempts, searches, indicators = [], [], []
    for record in relevant_runs:
        for attempt in store.list_source_attempts(record.id):
            if start <= attempt.at <= end:
                attempts.append(WeeklySourceAttempt(run_id=record.id, **attempt.model_dump()))
        for index, event in enumerate(store.list_traces(record.id)):
            if event.kind != "tool" or not start <= event.at <= end:
                continue
            output = event.data.get("output", {})
            if event.name == "search":
                searches.append(WeeklySearch(run_id=record.id, query=event.data.get("arguments", {}).get("query", ""), at=event.at, outcome=event.outcome, start_date=output.get("start_date"), end_date=output.get("end_date")))
            if record.id == run_id and event.name == "compare" and event.outcome == "success" and output.get("week_over_week") and set(output.get("evidence_ids", [])).issubset(cited_ids):
                observation = store.get_observation(output["to_observation_id"], mode)
                fields = {key: output[key] for key in ("from_observation_id", "to_observation_id", "baseline_period_start", "baseline_period_end", "endpoint_period_start", "endpoint_period_end", "absolute_change", "unit", "relative_growth_percent", "evidence_ids", "weekly_basis")}
                indicators.append(WeeklyIndicator(**fields, publisher=observation.publisher, definition=observation.definition, native_geography=observation.native_geography, mapped_geography=observation.mapped_geography, metric=observation.metric, comparison_trace_index=index))
    gaps = [] if indicators else ["No executed compatible source-defined weekly comparison supports this briefing. Monthly and quarterly measures remain dated context; missing weekly evidence does not establish no market movement."]
    if not attempts and not searches:
        gaps.append("No current-window source checks are recorded; retained evidence is historical context.")
    concentration = [WeeklySourceCount(publisher=publisher, domain=domain, publication_count=len(values["documents"]), excerpt_count=len(values["evidence"])) for (publisher, domain), values in sorted(sources.items())]
    return WeeklySummary(context=context, evidence_updates=updates, source_concentration=concentration, cited_publication_count=len({str(store.get_evidence(item.evidence_id, mode).source_url).rstrip("/") for item in updates}), cited_excerpt_count=len(updates), cited_domain_count=len({item.domain for item in concentration}), source_attempts=sorted(attempts, key=lambda item: item.at), fresh_searches=sorted(searches, key=lambda item: item.at), week_over_week=indicators, indicator_gaps=gaps, unassessed_fact_count=sum(claim.kind == "fact" and claim.signal is None for claim in claims))
