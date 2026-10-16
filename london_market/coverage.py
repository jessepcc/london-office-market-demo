"""Saved report coverage, constrained by inspected evidence and native measurements."""
from __future__ import annotations

import re

from .claim_audit import note_figures
from .domain import COVERAGE_REQUIREMENTS, SUBMARKETS, CoverageEntry, CoverageIntent, CoverageRequirement, ChartPayload, Claim, Evidence, Observation, ReportCoverage, RunResult, compatibility, stable_id
from .sources import MONITORED_SOURCES, Passage, reviewed_nonfact_passages


def evidence_facets(item: Evidence) -> set[str]:
    """Reviewed exact passages, never source-level topics or model-authored hashtags."""
    facets = set()
    for source in MONITORED_SOURCES.values():
        if str(item.source_url).rstrip('/') != source.url.rstrip('/'):
            continue
        for passage in source.qualitative_patterns:
            if isinstance(passage, tuple):
                passage = Passage(*passage, 'Central London', 'Central London')
            if (item.category, item.native_geography, item.mapped_geography) == (passage.category, passage.native_geography, passage.geography) and re.fullmatch(passage.pattern, item.excerpt):
                facets.update(passage.facets)
                facets.update(requirement.id for requirement in COVERAGE_REQUIREMENTS if passage.category in requirement.metrics or passage.category == requirement.id)
        for measurement in source.measurements:
            if (item.category, item.native_geography, item.mapped_geography) == (measurement.category, measurement.native_geography or measurement.geography, measurement.geography) and re.fullmatch(measurement.pattern, item.excerpt, re.IGNORECASE):
                facets.update(requirement.id for requirement in COVERAGE_REQUIREMENTS if measurement.metric in requirement.metrics)
    return facets


def reviewed_commentary(item: Evidence) -> bool:
    return any(source.source_kind == "primary_research_commentary" for source, _, _ in reviewed_nonfact_passages(item))


def reviewed_nonfact_segments(item: Evidence) -> list[str]:
    return [span for _, _, span in reviewed_nonfact_passages(item)]


def reviewed_repair_evidence_ids(sources: list[Evidence], evidence: list[Evidence]) -> list[str]:
    urls = {str(item.source_url).rstrip('/') for item in sources}
    return sorted({item.id for item in evidence if item.validation_status == 'validated' and str(item.source_url).rstrip('/') in urls for source in MONITORED_SOURCES.values() if str(item.source_url).rstrip('/') == source.url.rstrip('/') and item.published_at == source.published_at for passage in source.qualitative_patterns if isinstance(passage, Passage) and passage.kind is not None and (item.category, item.native_geography, item.mapped_geography) == (passage.category, passage.native_geography, passage.geography) and re.fullmatch(passage.pattern, item.excerpt)})


def coverage_gap_issues(gaps: dict[str, list[str]]) -> list[str]:
    return [f'{lane}: coverage gaps must be qualitative; cited statistics belong in claims' for lane, notes in gaps.items() if isinstance(notes, list) for note in notes if isinstance(note, str) and any(not 1900 <= value <= 2100 for value in note_figures(note))]

def local_exhibit_candidates(observations: list[Observation]) -> list[dict]:
    values = {}
    for item in observations:
        values.setdefault(item.semantic_key, set()).add(item.value)
    local = [item for item in observations if item.mapped_geography in SUBMARKETS and len(values[item.semantic_key]) == 1]
    groups = []
    for item in sorted(local, key=lambda item: (-item.period_end.toordinal(), item.metric, item.publisher, item.id)):
        group = next((group for group in groups if not compatibility(group[0], item, same_period=True, same_geography=False)), None)
        if group is None:
            groups.append([item])
        elif item.native_geography not in {value.native_geography for value in group}:
            group.append(item)
    return [dict(preferred='submarkets', metric=group[0].metric, publisher=group[0].publisher, definition=group[0].definition, unit=group[0].unit, period_start=str(group[0].period_start), period_end=str(group[0].period_end), native_geographies=[item.native_geography for item in group], observation_ids=[item.id for item in group], caveat='Same reviewed measurement and period across native local areas. Historical context; selected districts do not represent wider London.') for group in groups if len(group) >= 2]


class CoverageValidationError(ValueError):
    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__('; '.join(issues))


def missing_local_exhibit(observations: list[Observation], charts: list[ChartPayload]) -> list[dict]:
    candidates = local_exhibit_candidates(observations)
    by_id = {item.id: item for item in observations}
    attainable = {by_id[oid].mapped_geography for candidate in candidates for oid in candidate['observation_ids']}
    covered = set()
    for chart in charts:
        if chart.chart_type not in ('bar', 'multi_line'):
            continue
        for candidate in candidates:
            selected = set(candidate['observation_ids']).intersection(chart.observation_ids)
            if len(selected) >= 2:
                covered.update(by_id[oid].mapped_geography for oid in selected)
    missing = attainable - covered
    required = [{**candidate, 'missing_area_groups': sorted({by_id[oid].mapped_geography for oid in candidate['observation_ids']} & missing)} for candidate in candidates]
    return sorted((candidate for candidate in required if candidate['missing_area_groups']), key=lambda candidate: (-len(candidate['missing_area_groups']), -len(candidate['observation_ids']), -by_id[candidate['observation_ids'][0]].period_end.toordinal()))



def lane_support(requirement: CoverageRequirement, evidence_ids: set[str], evidence_by_id: dict[str, Evidence], observations: list[Observation]) -> tuple[list[Evidence], list[Observation]]:
    """Eligibility for one lane, not semantic entailment of arbitrary cited prose."""
    sources = [item for eid, item in evidence_by_id.items() if eid in evidence_ids and item.validation_status != 'rejected']
    linked = [item for item in observations if set(item.evidence_ids).intersection({source.id for source in sources}) and (item.mapped_geography == requirement.geography if requirement.dimension == 'area' else item.metric in requirement.metrics)]
    measured_ids = {eid for item in linked for eid in item.evidence_ids}
    matching = [item for item in sources if (item.mapped_geography == requirement.geography if requirement.dimension == 'area' else item.id in measured_ids or requirement.id in evidence_facets(item) or (item.validation_status == 'candidate' and item.category == requirement.id))]
    matching_ids = {item.id for item in matching}
    return matching, [item for item in linked if set(item.evidence_ids).intersection(matching_ids)]


def retained_coverage_index(evidence_by_id: dict[str, Evidence], observations: list[Observation]) -> list[dict]:
    """Unpaginated bounded pointers; inspect exact excerpts before using or excluding."""
    rows = []
    all_evidence = list(evidence_by_id.values())
    for requirement in COVERAGE_REQUIREMENTS:
        sources, linked = lane_support(requirement, set(evidence_by_id), evidence_by_id, observations)
        measured_ids = {eid for item in linked for eid in item.evidence_ids}
        ordered = sorted(sources, key=lambda item: (item.validation_status == 'candidate', item.id not in measured_ids, -(item.published_at.toordinal() if item.published_at else 0), item.id))
        selected = ordered[:2]
        ids = {item.id for item in selected}
        related = {}
        for evidence_id in reviewed_repair_evidence_ids(selected, all_evidence):
            item = evidence_by_id[evidence_id]
            key = (str(item.source_url), item.category, item.native_geography, item.mapped_geography, item.statement_kind, item.excerpt)
            previous = related.get(key)
            if previous is None or (item.retrieved_at, item.id) > (previous.retrieved_at, previous.id):
                related[key] = item
        rows.append(dict(requirement_id=requirement.id, related_evidence_ids=sorted(item.id for item in related.values())[:4], evidence_ids=[item.id for item in selected], observation_ids=[item.id for item in linked if set(item.evidence_ids).intersection(ids)], native_geographies=sorted({item.native_geography for item in selected}), publication_dates=sorted({str(item.published_at) for item in selected if item.published_at}), reference_periods=sorted({f'{item.period_start} to {item.period_end}' for item in linked if set(item.evidence_ids).intersection(ids)}), candidate=any(item.validation_status == 'candidate' for item in selected)))
    return rows



def hybrid_subject_missing(text: str) -> bool:
    """Clear absence of the attendance subject; presence does not prove entailment."""
    subject = r"\b(?:hybrid|attendance|homeworking|remote[ -]work(?:ing)?|work(?:ing)? from home|return[ -]to[ -]office|RTO|back to (?:their )?(?:desks|offices))\b"
    gap = r"\b(?:no|missing|unknown|unavailable|lack(?:ing)?|does not establish)\b.{0,65}\b(?:data|measurements?|evidence|indicators?)\b"
    return not any(re.search(subject, sentence, re.I) and not re.search(gap, sentence, re.I) for sentence in re.split(r'[.!?]', text))


def bind_report_coverage(intent: CoverageIntent, claims: list[Claim], charts_by_id: dict[str, ChartPayload], evidence_by_id: dict[str, Evidence], observations: list[Observation], *, require_full_market: bool, inspected_evidence_ids: set[str] | None = None) -> ReportCoverage:
    issues, entries = coverage_gap_issues(intent.gaps), []
    inspected_evidence_ids = inspected_evidence_ids or set()
    if require_full_market and intent.scope != 'full_market':
        issues.append('Reporting skill requires full_market coverage; focused scope cannot bypass required lanes')
    retained_index = {row["requirement_id"]: row for row in retained_coverage_index(evidence_by_id, observations)}
    for requirement in COVERAGE_REQUIREMENTS:
        indexes, chart_ids, sources_by_id, measurements, absent_hybrid = [], [], {}, {}, set()
        for index, claim in enumerate(claims):
            if claim.kind == 'watchpoint':
                continue
            sources, linked = lane_support(requirement, set(claim.evidence_ids), evidence_by_id, observations)
            if requirement.id == "hybrid" and sources and hybrid_subject_missing(claim.text):
                absent_hybrid.update(item.id for item in sources if item.mapped_geography != "Unmapped")
                sources = [item for item in sources if item.mapped_geography == "Unmapped"]
                linked = [item for item in linked if item.mapped_geography == "Unmapped"]
            if sources:
                indexes.append(index)
                sources_by_id.update((item.id, item) for item in sources)
                measurements.update((item.id, item) for item in linked)
        for cid, chart in charts_by_id.items():
            sources, linked = lane_support(requirement, set(chart.source_refs), evidence_by_id, [item for item in observations if item.id in chart.observation_ids])
            if sources:
                chart_ids.append(cid)
                sources_by_id.update((item.id, item) for item in sources)
                measurements.update((item.id, item) for item in linked)
        sources = list(sources_by_id.values())
        gaps = list(intent.gaps.get(requirement.id, []))
        if absent_hybrid and not any(item.mapped_geography != "Unmapped" for item in sources) and not gaps:
            issues.append("hybrid: selected local leaf evidence_ids=" + ",".join(sorted(absent_hybrid)) + " has no substantive attendance, hybrid-working or return-to-office finding. Write a narrowly supported local attendance finding or retain an explicit context-only London gap; footprint, talent and best-space prose alone cannot close hybrid.")
        if not sources:
            if intent.scope == 'focused' and requirement.id not in intent.gaps:
                continue
            available, _ = lane_support(requirement, set(evidence_by_id), evidence_by_id, observations)
            if intent.scope == 'full_market' and available and (not gaps or not inspected_evidence_ids.intersection(item.id for item in available)):
                pointers = retained_index[requirement.id]
                issues.append(f"{requirement.id}: retained support exists; inspect exact evidence_ids={','.join(pointers['evidence_ids'])}, then include a substantive dated finding or give an explicit evidence-applicability exclusion in coverage.gaps. Watchpoints alone do not close a lane.")
            entries.append(CoverageEntry(requirement_id=requirement.id, basis='gap', gaps=gaps or ['No eligible retained source supports a substantive finding for this lane.']))
            continue
        context = all(item.mapped_geography == 'Unmapped' for item in sources)
        local_measurements = [item for item in measurements.values() if item.mapped_geography != 'Unmapped']
        basis = 'context' if context else 'measurement' if local_measurements else 'qualitative'
        broader = sorted({item.native_geography for item in measurements.values() if item.mapped_geography == 'Unmapped'})
        if broader:
            gaps.append('Measurements for ' + ', '.join(broader) + ' are broader context, not London office measurements.')
        if any(item.validation_status == 'candidate' for item in sources):
            gaps.append('Candidate evidence awaits monitoring validation; category eligibility is not semantic or measurement verification.')
        if requirement.id == 'midtown_fringe' and not any('Midtown' in item.native_geography for item in sources):
            gaps.append('City Fringe evidence does not establish Midtown measurements.')
        if requirement.id == 'wider_london':
            gaps.append('Selected wider London districts do not represent the whole area.')
        if requirement.dimension == 'area' and not local_measurements:
            gaps.append('Local qualitative events do not establish area-wide market measurements.')
        entries.append(CoverageEntry(requirement_id=requirement.id, basis=basis, claim_indexes=indexes, chart_ids=chart_ids, gaps=list(dict.fromkeys(gaps))))
    if intent.scope == 'full_market':
        candidates = missing_local_exhibit(observations, list(charts_by_id.values()))
        if candidates:
            issues.append('Full-market reports must prepare a compatible local exhibit through choose_visualization then prepare_visualization_data; candidate observation_ids=' + ','.join(candidates[0]['observation_ids']) + '; missing area groups=' + ','.join(candidates[0]['missing_area_groups']))
        attached = {cid for entry in entries for cid in entry.chart_ids}
        if set(charts_by_id) - attached:
            issues.append('Selected exhibit has no eligible required topic or area support')
    if issues:
        raise CoverageValidationError(issues)
    return ReportCoverage(scope=intent.scope, entries=entries)


def report_coverage_rows(result: RunResult, evidence: list[Evidence], observations: list[Observation]) -> list[dict]:
    if result.coverage is None:
        return []
    entries = {entry.requirement_id: entry for entry in result.coverage.entries}
    charts = {stable_id('chart', chart.model_dump(mode='json')): chart for chart in result.charts}
    evidence_by_id = {item.id: item for item in evidence}
    rows = []
    for requirement in COVERAGE_REQUIREMENTS:
        entry = entries.get(requirement.id)
        if entry is None:
            continue
        selected_claims = [result.claims[index] for index in entry.claim_indexes if index < len(result.claims)]
        selected_charts = [charts[cid] for cid in entry.chart_ids if cid in charts]
        eids = {eid for claim in selected_claims for eid in claim.evidence_ids} | {eid for chart in selected_charts for eid in chart.source_refs}
        sources, linked = lane_support(requirement, eids, evidence_by_id, observations)
        periods = sorted({f'{item.period_start} to {item.period_end}' for item in linked})
        dates = sorted({str(item.published_at) for item in sources if item.published_at})
        events = sorted({str(item.event_date) for item in sources if item.event_date})
        findings = [claim.text for claim in selected_claims]
        if not findings:
            findings = [f'{point.native_geography or point.geography}: {point.value:g} {point.unit}' for chart in selected_charts for point in chart.points if (not requirement.geography or point.geography == requirement.geography) and (not requirement.metrics or point.metric in requirement.metrics)]
        rows.append(dict(id=requirement.id, label=requirement.label, dimension=requirement.dimension, basis=entry.basis, candidate=any(item.validation_status == 'candidate' for item in sources), findings=findings, claim_indexes=entry.claim_indexes, chart_ids=entry.chart_ids, evidence_ids=sorted(item.id for item in sources), native_geographies=sorted({item.native_geography for item in sources}), periods=periods, publication_dates=dates, event_dates=events, gaps=entry.gaps, source_urls=sorted({str(item.source_url) for item in sources}), qualifications=sorted({item.validation_reason for item in sources if item.validation_reason})))
    return rows
