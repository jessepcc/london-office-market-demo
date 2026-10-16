"""Synthetic examples; they demonstrate behavior and are not market evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date, datetime, timezone

from .domain import COVERAGE_REQUIREMENTS, CoverageGap, Document, Evidence, Observation, SUBMARKETS
from .store import Store

FIXTURE_LABEL = "Demo mode uses synthetic sample data and scripted responses. This is not live market research."


def scripted_weekly_tags(payload: dict) -> dict:
    claims = []
    for claim in payload.get("claims", []):
        if not isinstance(claim, dict) or "analysis" in claim:
            claims.append(claim)
            continue
        text = claim.get("text", "").lower()
        tag = "vacancy" if "vacan" in text else "availability" if "availability" in text else "rents" if "rent" in text else "supply" if any(word in text for word in ("supply", "construction", "development")) else "demand"
        claims.append({"topic_tags": [tag], **claim})
    return {**payload, "claims": claims}


def scripted_coverage(claims: list[dict], observations: list[dict], chart_ids: list[str], *, evidence: Sequence[dict] = (), chart_payloads: Sequence[dict] = ()) -> dict:
    from .coverage import lane_support
    selected_ids = {eid for claim in claims if claim.get("kind") != "watchpoint" for eid in claim.get("evidence_ids", [])} | {eid for chart in chart_payloads for eid in chart.get("source_refs", [])}
    selected_ids |= {eid for claim in claims for eid in claim.get("analysis", {}).values() if isinstance(eid, str) and eid.startswith("ev_")}
    evidence_by_id = {item["id"]: Evidence.model_validate(item) for item in evidence}
    normalized = [Observation.model_validate(item) for item in observations]
    gaps = {}
    for requirement in COVERAGE_REQUIREMENTS:
        sources, linked = lane_support(requirement, selected_ids, evidence_by_id, normalized)
        measured = any((item.mapped_geography == requirement.geography if requirement.dimension == "area" else item.metric in requirement.metrics) and set(item.evidence_ids).intersection(selected_ids) for item in normalized)
        available, _ = lane_support(requirement, set(evidence_by_id), evidence_by_id, normalized)
        available_measurement = any(item.mapped_geography == requirement.geography if requirement.dimension == "area" else item.metric in requirement.metrics for item in normalized)
        if not sources and not measured and (available or available_measurement):
            gaps[requirement.id] = ["Inspected evidence has no separate supported finding suitable for this bounded scripted demonstration; further evidence appraisal is required."]
    return {"scope": "full_market", "gaps": gaps}


def fixture_batches() -> list[tuple[Document, list[Evidence], list[Observation]]]:
    retrieved_at = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)
    specifications = [
        ("city-q1", "City", date(2026, 1, 1), date(2026, 3, 31), date(2026, 9, 30), 8.0, None),
        ("city-q2", "City", date(2026, 4, 1), date(2026, 6, 30), date(2026, 7, 15), 7.5, 9.2),
        ("west-end-q2", "West End", date(2026, 4, 1), date(2026, 6, 30), date(2026, 7, 15), 4.5, None),
    ]
    batches = []
    for slug, geography, start, end, published, vacancy, availability in specifications:
        text = f"Synthetic {geography} vacancy rate {vacancy}% for {start} to {end}."
        if availability is not None:
            text += f" Synthetic availability rate {availability}%; different definition."
        document = Document(mode="fixture", synthetic=True, url=f"https://fixtures.example.invalid/{slug}", title=f"Synthetic {geography} office example {end}", publisher="Synthetic Market Publisher", content_hash=hashlib.sha256(text.encode()).hexdigest(), published_at=published, retrieved_at=retrieved_at)
        evidence = []
        observations = []
        for metric, value, definition, category in [
            ("vacancy_rate", vacancy, "Unoccupied office space as a share of office stock", "vacancy"),
            ("availability_rate", availability, "Marketed office space including occupied space offered for future letting", "availability"),
        ]:
            if value is None:
                continue
            item = Evidence(document_id=document.id, mode="fixture", excerpt=f"Synthetic {geography}: {metric} {value}% during {start}–{end}.", event_date=end if slug == "city-q1" else None, published_at=published, retrieved_at=retrieved_at, source_url=document.url, native_geography=geography, mapped_geography=geography, category=category, validation_status="validated", validation_reason="Controlled synthetic monitoring fixture; no live-market authority")
            evidence.append(item)
            observations.append(Observation(mode="fixture", metric=metric, value=value, unit="%", publisher=document.publisher, definition=definition, category=category, native_geography=geography, mapped_geography=geography, period_start=start, period_end=end, evidence_ids=[item.id]))
        batches.append((document, evidence, observations))
    from .sources import fixture_analysis_sources
    return batches + fixture_analysis_sources()


def load_fixtures(store: Store) -> dict[str, int]:
    totals = {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
    for document, evidence, observations in fixture_batches():
        for name, count in store.ingest(document, evidence, observations).items():
            totals[name] += count
    return totals


def coverage_gaps(store: Store, mode: str) -> list[CoverageGap]:
    covered = {item.mapped_geography for item in store.list_observations(mode)}
    return [CoverageGap(geography=geography, reason="No validated submarket observations; aggregate Central London evidence cannot establish coverage here") for geography in SUBMARKETS if geography not in covered]
