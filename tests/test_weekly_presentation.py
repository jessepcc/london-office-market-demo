from datetime import date, datetime, timezone
from pathlib import Path

from streamlit.testing.v1 import AppTest

from london_market.domain import (
    ChartPayload, ChartPoint, Claim, Document, Evidence, ReportSection, RunRecord, RunResult, Task, WeeklyContext,
    WeeklyEvidenceUpdate, WeeklyIndicator, WeeklyReferencePeriod,
    WeeklySourceAttempt, WeeklySourceCount, WeeklySummary, utc_now,
)
from london_market.fixtures import fixture_batches
from london_market.presentation import (
    claim_metadata, claim_timing, market_evidence_groups,
    partition_market_evidence, weekly_indicator_groups, weekly_summary_copy,
)
from london_market.report import chart_reference_id, export_report_html, report_groups
from london_market.store import Store


def weekly_example():
    document, sources, observations = fixture_batches()[0]
    source = sources[0]
    older = source.model_copy(update={"id": "older", "document_id": "older-document", "published_at": date(2026, 7, 10)})
    recent = source.model_copy(update={"id": "recent", "document_id": "recent-document", "published_at": date(2026, 10, 9)})
    at = datetime(2026, 10, 10, 12, tzinfo=timezone.utc)
    period = WeeklyReferencePeriod(observation_id=observations[0].id, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    updates = [WeeklyEvidenceUpdate(
        evidence_id=item.id, document_id=item.document_id, published_at=item.published_at,
        event_date=None, retrieved_at=at, first_ingested_at=at,
        previously_cited=None, publication_in_window=item.id == "recent",
        event_in_window=False, first_ingested_in_window=item.id == "recent",
        reference_periods=[period],
    ) for item in [older, recent]]
    summary = WeeklySummary(
        context=WeeklyContext(briefing_date=date(2026, 10, 10), report_week_start=date(2026, 10, 5),
                              window_start=date(2026, 10, 4), window_end=date(2026, 10, 10),
                              cutoff_at=at, timezone="Europe/London"),
        evidence_updates=updates,
        source_concentration=[WeeklySourceCount(publisher="Test publisher", domain="example.com", publication_count=2, excerpt_count=2)],
        cited_publication_count=2, cited_excerpt_count=2, cited_domain_count=1,
        indicator_gaps=["Quarterly vacancy data do not support a weekly comparison."],
        unassessed_fact_count=1,
    )
    claims = [
        Claim(kind="fact", text="Retained supported fact.", evidence_ids=[older.id], signal="bullish", topic_tags=["demand", "rents"]),
        Claim(kind="fact", text="New publication describes an earlier quarter.", evidence_ids=[recent.id], signal="bullish", topic_tags=["availability"]),
        Claim(kind="fact", text="Available space increases competition.", evidence_ids=[older.id], signal="bearish", topic_tags=["supply"]),
        Claim(kind="fact", text="Undirected market fact.", evidence_ids=[older.id], signal="neutral", topic_tags=["macro"]),
        Claim(kind="fact", text="Legacy fact has no assessment.", evidence_ids=[older.id]),
        Claim(kind="interpretation", text="Consider the market context.", evidence_ids=[older.id], topic_tags=["rents"]),
        Claim(kind="forecast", text="Source expectation remains uncertain.", evidence_ids=[older.id], topic_tags=["supply"]),
        Claim(kind="watchpoint", text="Inspect the next dated release.", evidence_ids=[older.id], topic_tags=["availability"]),
    ]
    result = RunResult(status="partial", answer="Saved example", reason="Test", briefing_date=date(2026, 10, 10),
                       claims=claims, weekly=summary, evidence_ids=[older.id, recent.id],
                       sections=[ReportSection(kind="executive_findings", claim_indexes=list(range(len(claims))))])
    return document, [older, recent], observations, result


def test_weekly_groups_all_facts_once_with_recent_publications_first_and_unknown_separate():
    _, _, _, result = weekly_example()
    assert market_evidence_groups(result) == [
        ("bullish", "Bullish", [1, 0]), ("bearish", "Bearish", [2]),
        ("neutral", "Neutral", [3]), (None, "Signal unavailable", [4]),
    ]
    groups = list(report_groups(result))
    assert groups == [
        ("market_facts", "Market evidence", [0, 1, 2, 3, 4], []),
        ("source_forecasts", "Source forecasts", [6], []),
        ("interpretations", "Interpretation", [5], []),
        ("watchpoints", "What to watch", [7], []),
    ]
    failed = result.model_copy(update={"status": "failed"})
    assert market_evidence_groups(failed)[-1][2] == [1, 0, 2, 3, 4]
    assert market_evidence_groups(result, {"recent"})[-1][2] == [1, 4]


def test_new_publication_keeps_old_measurement_period_and_saved_timing_authority():
    _, sources, observations, result = weekly_example()
    changed_source = sources[1].model_copy(update={"published_at": date(2027, 1, 1), "event_date": date(2027, 1, 1)})
    assert claim_timing(result.claims[1], result) == "New publication this week"
    assert claim_metadata(result.claims[1], result, [changed_source], observations) == [
        "New publication this week", "Published 9 Oct 2026", "Reference period 1 Apr 2026 to 30 Jun 2026",
    ]
    assert weekly_summary_copy(result)["window"] == "4 Oct 2026 to 10 Oct 2026"
    assert weekly_summary_copy(result)["coverage"] == "2 cited publications · 2 excerpts · 1 domain"
    event_update = result.weekly.evidence_updates[1].model_copy(update={"event_in_window": True, "event_date": date(2026, 10, 8)})
    event = result.model_copy(update={"weekly": result.weekly.model_copy(update={"evidence_updates": [event_update]})})
    assert claim_timing(event.claims[1], event) == "Event this week"


def test_recent_ingestion_does_not_claim_earlier_publication_for_unknown_or_future_dates():
    _, sources, observations, result = weekly_example()
    for published_at, publication_label in ((None, "Publication date unknown"), (date(2027, 1, 1), "Published 1 Jan 2027")):
        update = result.weekly.evidence_updates[1].model_copy(update={"publication_in_window": False, "published_at": published_at})
        updated = result.model_copy(update={"weekly": result.weekly.model_copy(update={"evidence_updates": [update]})})
        assert claim_timing(updated.claims[1], updated) == "Evidence added this week"
        assert claim_metadata(updated.claims[1], updated, sources, observations) == [
            "Evidence added this week", publication_label, "Reference period 1 Apr 2026 to 30 Jun 2026",
        ]
        assert partition_market_evidence(updated, [1, 0]) == ([], [1, 0])


def test_current_publications_and_events_stay_open_while_ingested_historical_evidence_is_context():
    _, _, _, result = weekly_example()
    assert partition_market_evidence(result, [1, 0]) == ([1], [0])
    for publication_date in (date(2026, 7, 10), None):
        update = result.weekly.evidence_updates[0].model_copy(update={"published_at": publication_date, "first_ingested_in_window": True})
        ingested = result.model_copy(update={"weekly": result.weekly.model_copy(update={"evidence_updates": [update]})})
        assert partition_market_evidence(ingested, [0]) == ([], [0])
    event_update = update.model_copy(update={"event_date": date(2026, 10, 8), "event_in_window": True})
    event = result.model_copy(update={"weekly": result.weekly.model_copy(update={"evidence_updates": [event_update]})})
    assert partition_market_evidence(event, [0]) == ([0], [])
    legacy = result.model_copy(update={"weekly": None})
    assert partition_market_evidence(legacy, [1, 0]) == ([1, 0], [])


def test_export_has_one_perspective_three_assessments_tags_and_honest_weekly_gaps():
    _, sources, observations, result = weekly_example()
    artifact = export_report_html(result, sources, observations)
    assert artifact.count("From a landlord’s leasing perspective") == 1
    assert '<h3 class="signal-heading signal-bullish">Bullish · 2</h3>' in artifact
    assert '<h3 class="signal-heading signal-bearish">Bearish · 1</h3>' in artifact
    assert '<h3 class="signal-heading signal-neutral">Neutral · 1</h3>' in artifact
    assert 'Signal unavailable · 1' in artifact
    assert artifact.index("New publication describes an earlier quarter.") < artifact.index("Retained supported fact.")
    assert artifact.index("New publication describes an earlier quarter.") < artifact.index('<details class="retained-context">')
    assert artifact.count('<details class="retained-context"><summary>Retained context · 1</summary>') == 4
    assert '<details class="retained-context" open' not in artifact
    for claim in result.claims:
        assert artifact.count(claim.text) == 1
    assert "#demand #rents" in artifact
    assert "New publication this week · Published 9 Oct 2026 · Reference period 1 Apr 2026 to 30 Jun 2026" in artifact
    assert "No comparable weekly measurements are available." in artifact
    assert "Quarterly vacancy data do not support a weekly comparison." in artifact
    assert "Several excerpts from one publication are not independent sources." in artifact
    assert 'href="#evidence-recent"' in artifact
    assert "Source forecasts" in artifact and "Interpretation" in artifact and "What to watch" in artifact


def test_compact_weekly_status_keeps_failures_and_missing_measurements_before_evidence():
    _, sources, observations, result = weekly_example()
    attempts = [WeeklySourceAttempt(run_id="check", url="https://example.com/failed", status="failed", at=utc_now(), error="Exact source failure")]
    reviewed = result.model_copy(update={"reason": "Editor-reviewed: documented correction", "weekly": result.weekly.model_copy(update={"source_attempts": attempts})})
    copy = weekly_summary_copy(reviewed)
    assert copy["status"] == "Completed with gaps · 1 source check, 1 failed or blocked."
    assert copy["measurements"] == "No comparable weekly measurements are available."
    artifact = export_report_html(reviewed, sources, observations)
    before_evidence = artifact.split('<section id="market_facts">')[0]
    assert copy["window"] in before_evidence and copy["cutoff"] in before_evidence
    assert copy["status"] in before_evidence and copy["measurements"] in before_evidence
    assert "Quarterly vacancy data do not support a weekly comparison." not in before_evidence
    assert "Exact source failure" not in before_evidence and "Exact source failure" in artifact
    assert "Editor-reviewed briefing. Run history retains the original draft and documented corrections." in artifact
    assert "pre-let example was added" not in artifact


def test_weekly_indicators_show_exact_endpoints_values_units_and_separate_uk_context():
    _, sources, observations, result = weekly_example()
    first = observations[0].model_copy(update={"id": "week-1", "value": 8, "period_start": date(2026, 9, 27), "period_end": date(2026, 10, 3)})
    second = first.model_copy(update={"id": "week-2", "value": 7.5, "period_start": date(2026, 10, 4), "period_end": date(2026, 10, 10)})
    indicator = WeeklyIndicator(from_observation_id=first.id, to_observation_id=second.id,
                                baseline_period_start=first.period_start, baseline_period_end=first.period_end,
                                endpoint_period_start=second.period_start, endpoint_period_end=second.period_end,
                                absolute_change=-0.5, unit="percentage points", publisher=first.publisher,
                                definition=first.definition, native_geography="City", mapped_geography="City",
                                metric="vacancy_rate", weekly_basis="period_total", evidence_ids=[sources[0].id], comparison_trace_index=1)
    uk = indicator.model_copy(update={"metric": "bank_rate", "native_geography": "United Kingdom", "mapped_geography": "Unmapped"})
    result = result.model_copy(update={"weekly": result.weekly.model_copy(update={"week_over_week": [indicator, uk]})})
    groups = weekly_indicator_groups(result, [first, second])
    assert [heading for heading, _ in groups] == ["London office indicators", "UK economic context"]
    row = groups[0][1][0]
    assert row["Previous period"] == "27 Sep 2026 to 3 Oct 2026"
    assert row["Latest period"] == "4 Oct 2026 to 10 Oct 2026"
    assert row["Previous value"] == "8 %" and row["Latest value"] == "7.5 %"
    assert row["Change"] == "-0.5 percentage points"
    artifact = export_report_html(result, sources, [first, second])
    assert "UK economic context" in artifact and "27 Sep 2026 to 3 Oct 2026" in artifact
    assert "No comparable weekly measurements are available." not in artifact


def test_saved_weekly_streamlit_has_grouped_cards_and_collapsed_context(tmp_path, monkeypatch):
    document, sources, observations, result = weekly_example()
    documents = [Document(**(document.model_dump(exclude={"id"}) | {"published_at": source.published_at, "content_hash": str(index + 1) * 64})) for index, source in enumerate(sources)]
    stored_sources = [Evidence(**(source.model_dump(exclude={"id", "document_id"}) | {"document_id": item.id})) for source, item in zip(sources, documents, strict=True)]
    id_map = {source.id: saved.id for source, saved in zip(sources, stored_sources, strict=True)}
    claims = [claim.model_copy(update={"evidence_ids": [id_map[eid] for eid in claim.evidence_ids]}) for claim in result.claims]
    updates = [update.model_copy(update={"evidence_id": id_map[update.evidence_id], "document_id": item.id}) for update, item in zip(result.weekly.evidence_updates, documents, strict=True)]
    result = result.model_copy(update={"claims": claims, "evidence_ids": list(id_map.values()), "weekly": result.weekly.model_copy(update={"evidence_updates": updates})})
    observation = observations[0].model_copy(update={"evidence_ids": [stored_sources[0].id]})
    result = result.model_copy(update={"observation_ids": [observation.id]})
    database = tmp_path / "weekly.sqlite3"
    run = RunRecord(task=Task(objective="Weekly example", entry_skill="prepare_weekly_briefing"),
                    status="partial", main_model="fixture", router_model="fixture", result=result, finished_at=utc_now())
    with Store(database) as store:
        for index, (source_document, source) in enumerate(zip(documents, stored_sources, strict=True)):
            store.ingest(source_document, [source], [observation] if index == 0 else [])
        store.save_run(run)
        store.save_report(run.id, result.weekly.context.report_week_start, result)
    monkeypatch.setenv("LONDON_MARKET_DB", str(database))
    monkeypatch.setenv("LONDON_MARKET_MODE", "fixture")
    from ui_support import app_test
    app = app_test(database)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception
    assert {"This week", "Market evidence", "Source forecasts", "Interpretation", "What to watch"}.issubset({item.value for item in app.subheader})
    cards = [item.value for item in app.markdown if 'class="market-claim"' in item.value]
    assert len(cards) == 8
    assert any("#demand #rents" in card for card in cards)
    assert sum(item.value == "From a landlord’s leasing perspective" for item in app.caption) == 1
    status = next(item.value for item in app.markdown if item.value.startswith("**4 Oct 2026 to 10 Oct 2026**"))
    assert "Research cutoff" in status and "Completed with gaps" in status
    assert "No source checks recorded." in status and "No comparable weekly measurements are available." in status
    assert status.count("  \n") == 3
    labels = {item.label for item in app.expander}
    assert {"Market context", "Evidence coverage and source concentration", "Inspect cited sources"}.issubset(labels)
    context = next(item for item in app.expander if item.label == "Market context")
    assert context.proto.expanded is False
    retained = [item for item in app.expander if item.label == "Retained context · 1"]
    assert len(retained) == 4 and all(item.proto.expanded is False for item in retained)
    retained_cards = [card.value for item in retained for card in item.markdown if 'class="market-claim"' in card.value]
    assert len(retained_cards) == 4
    assert all("New publication describes an earlier quarter." not in card for card in retained_cards)
    assert any("Retained supported fact." in card for card in retained_cards)
    assert len(app.get("download_button")) == 1


def test_chart_only_fact_uses_saved_chart_sources_for_chronology_and_availability():
    _, sources, observations, result = weekly_example()
    observation = observations[0]
    chart = ChartPayload(chart_type="table", title="Earlier quarter", purpose="Source-native context",
                         source_refs=["recent"], observation_ids=[observation.id],
                         points=[ChartPoint(label="City", value=observation.value, observation_id=observation.id,
                                            evidence_ids=["recent"], period_start=date(2026, 4, 1), period_end=date(2026, 6, 30),
                                            geography=observation.mapped_geography, metric=observation.metric,
                                            unit=observation.unit, publisher=observation.publisher,
                                            definition=observation.definition, category=observation.category)])
    claim = Claim(kind="fact", text="A dated source-native exhibit.", evidence_ids=[],
                  chart_ids=[chart_reference_id(chart)], signal="bullish")
    result = result.model_copy(update={"claims": [claim], "charts": [chart], "sections": []})
    assert claim_timing(claim, result) == "New publication this week"
    assert claim_metadata(claim, result, sources, []) == [
        "New publication this week", "Published 9 Oct 2026", "Reference period 1 Apr 2026 to 30 Jun 2026",
    ]
    assert partition_market_evidence(result, [0]) == ([0], [])
    assert market_evidence_groups(result, {"recent"})[-1] == (None, "Signal unavailable", [0])
    artifact = export_report_html(result, sources, [])
    assert "New publication this week · Published 9 Oct 2026 · Reference period 1 Apr 2026 to 30 Jun 2026" in artifact
    assert 'retained-context' not in artifact.split(claim.text)[0].rsplit('<section id="market_facts">', 1)[-1]
    missing = export_report_html(result, [], [])
    assert "Unsupported fact" in missing
    assert claim.evidence_ids == []


def test_actual_saved_chart_period_and_publication_do_not_require_direct_citations():
    import json
    packet = json.loads((Path(__file__).parent / "fixtures/completion/attempt8-signal-plan.json").read_text())
    chart = ChartPayload.model_validate(next(c for c in packet["charts"] if c["chart_type"] == "bar"))
    evidence = [Evidence.model_validate(e) for e in packet["evidence"]]
    claim = Claim(kind="fact", text="Source-native vacancy comparison.", chart_ids=[chart_reference_id(chart)])
    result = RunResult(status="completed", answer="Context", reason="Saved exhibit", claims=[claim], charts=[chart])
    assert claim_metadata(claim, result, evidence, []) == [
        "Published 30 Jan 2026", "Reference period 1 Oct 2025 to 31 Dec 2025",
    ]
    assert claim.evidence_ids == []
