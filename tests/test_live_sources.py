"""Reviewed source excerpts exercise live bindings without paid services."""

import hashlib
from datetime import date
from pathlib import Path

import httpx
import pytest

from london_market.config import Settings
from london_market.domain import Document, RunRecord, Task
from london_market.providers import FetchedPage, ProviderError, WebProvider
from london_market.sources import MONITORED_SOURCES, parse_monitored_source
from london_market.store import Store
from london_market.tools import (
    CollectMonitoredSourceArgs, CompareArgs, MonitoringSourcesArgs, PrepareArgs, ReadDocumentArgs,
    ToolContext, VisualizationArgs, choose_visualization, collect_monitored_source,
    compare, monitoring_sources, prepare_visualization_data, read_document,
)


@pytest.mark.parametrize("characters,provider_truncated", [(2000, False), (5000, False), (2000, True), (5000, True)])
def test_retrieval_preview_and_provider_retention_are_independent(tmp_path, characters, provider_truncated):
    from london_market.tools import RetrieveArgs, retrieve
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    page = page_for(source.id, text="x" * characters).model_copy(update={"truncated": provider_truncated})
    with Store(tmp_path / "db") as store:
        ctx = context(store, {source.url: page})
        result = retrieve(ctx, RetrieveArgs(url=source.url))
        assert result.preview_truncated is (characters > 3500)
        assert result.provider_truncated is provider_truncated
        assert result.retained_characters == characters
        assert result.preview_limit_characters == 3500
        assert result.document_limit_characters == ctx.settings.max_document_chars
        assert "truncated" not in result.model_dump()
        if characters > 3500:
            continuation = read_document(ctx, ReadDocumentArgs(retrieved_id=result.retrieved_id, offset=result.next_offset))
            assert len(result.text + continuation.text) == characters
            assert continuation.provider_truncated is provider_truncated
        else:
            assert result.next_offset is None


SOURCE_IDS = ["savills-central-london-q1-2026", "savills-central-london-q2-2026", "savills-central-london-june-2026"]
EXCERPTS = Path(__file__).parent / "fixtures" / "live_sources"
SECONDARY_PAGES = {"semafor-canary-wharf-october-2026": 1, "cyprus-mail-uk-defence-october-2026": 2}


def secondary_page_for(source_id):
    path = Path(__file__).parents[1] / 'evidence/acceptance/requirements-coverage' / f'build-20261011-source-{SECONDARY_PAGES[source_id]}.json'
    return FetchedPage.model_validate_json(path.read_text())


def text_for(source_id):
    key = dict(zip(SOURCE_IDS, ("q1", "q2", "june")))[source_id]
    return (EXCERPTS / f"{key}.txt").read_text()


def page_for(source_id, text=None, **updates):
    source = MONITORED_SOURCES[source_id]
    text = text_for(source_id) if text is None else text
    document = Document(mode="live", url=source.url, publisher="www.savills.com", title="Savills reviewed article", content_hash=hashlib.sha256(text.encode()).hexdigest())
    if updates:
        document = Document.model_validate({**document.model_dump(exclude={"id"}), **updates})
    return FetchedPage(document=document, text=text)


def context(store, pages=None):
    pages = pages or {MONITORED_SOURCES[sid].url: page_for(sid) for sid in SOURCE_IDS}
    class ReviewedExcerptWeb:
        def retrieve(self, url, timeout):
            value = pages[url]
            if isinstance(value, Exception):
                raise value
            return value
    run = RunRecord(task=Task(objective="Collect reviewed sources", entry_skill="collect_market_evidence", mode="live"), main_model="unpaid-test", router_model="unpaid-test")
    store.save_run(run)
    return ToolContext(Settings(), store, ReviewedExcerptWeb(), "live", run.id, 3)


@pytest.mark.parametrize("source_id,expected_values", [
    (SOURCE_IDS[0], [2.2, 19.1, 7.4, 7.0, 7.8, 1.4, 130.8, 165.0]),
    (SOURCE_IDS[1], [2.4, 18.7, 7.2, 7.0, 1.4, 16.1, 175.1, 76.21, 106.17]),
    (SOURCE_IDS[2], [600830, 135, 183.38]),
])
def test_reviewed_primary_excerpts_preserve_observed_values_and_provenance(source_id, expected_values):
    page = page_for(source_id)
    parsed = parse_monitored_source(source_id, page)
    assert [item.value for item in parsed.observations] == expected_values
    assert parsed.document.published_at == MONITORED_SOURCES[source_id].published_at
    assert all(item.excerpt in page.text and item.validation_status == "validated" for item in parsed.evidence)
    assert all(item.source_url == page.document.url and item.event_date is None for item in parsed.evidence)
    assert all(item.publisher == "Savills" and item.evidence_ids for item in parsed.observations)
    assert parsed.caveats


def test_actual_web_provider_extracts_article_and_retains_raw_digest(tmp_path):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    raw = f'<html><title>Savills</title><main><time class="sv-content-header__date">01 May 2026</time><article>{text_for(source.id)}</article></main></html>'.encode()
    web = WebProvider(Settings(), transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw, headers={"content-type": "text/html; charset=utf-8"})), resolver=lambda host: ["93.184.216.34"])
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        ctx.web = web
        result = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert result.document.raw_content_hash == hashlib.sha256(raw).hexdigest()
        assert result.counts == {"documents": 1, "evidence": 11, "observations": 8, "revisions": 0}
        assert store.list_source_attempts(ctx.run_id)[0].status == "success"
        assert store.get_observation(result.observation_ids[0]).value == 2.2


def test_ingestion_repeats_without_duplicate_rows_and_charts_use_real_values(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        first = [collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=sid)) for sid in SOURCE_IDS]
        assert store.counts("live")["observations"] == 20
        for result in first:
            repeated = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=result.source_id))
            assert repeated.counts == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
        observations = [item for item in store.list_observations("live") if item.metric == "vacancy_rate" and item.mapped_geography == "Central London"]
        compared = compare(ctx, CompareArgs(observation_ids=[item.id for item in observations]))
        assert compared.compatible and compared.absolute_change == -0.2 and compared.unit == "percentage points"
        plan = choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in observations], title="Central London vacancy", purpose="Compare reported quarters"))
        payload = prepare_visualization_data(ctx, PrepareArgs(plan_id=plan.plan_id)).payload
        assert payload.chart_type == "line" and [point.value for point in payload.points] == [7.4, 7.2]
        assert all(store.get_evidence(eid).validation_status == "validated" for eid in payload.source_refs)
        assert [attempt.status for attempt in store.list_source_attempts(ctx.run_id)] == ["success"] * 3 + ["unchanged"] * 3


def test_period_ambiguities_do_not_create_rent_trend_or_false_june_activity(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        for sid in SOURCE_IDS:
            collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=sid))
        observations = store.list_observations("live")
        west_prime = [item for item in observations if item.metric == "prime_rent" and item.mapped_geography == "West End" and item.period_end in (date(2026, 3, 31), date(2026, 6, 30))]
        compared = compare(ctx, CompareArgs(observation_ids=[item.id for item in west_prime]))
        assert not compared.compatible and "cadence" in compared.reason
        assert not any(item.metric == "prime_rent" and item.mapped_geography == "City" and item.period_end == date(2026, 6, 30) for item in observations)
        may = next(item for item in observations if item.metric == "take_up" and item.unit == "sq ft")
        assert (may.period_start, may.period_end) == (date(2026, 5, 1), date(2026, 5, 31))
        assert store.get_evidence(may.evidence_ids[0]).published_at == date(2026, 6, 30)
        h1_grade_a = [item for item in observations if item.metric == "grade_a_rent"]
        assert len(h1_grade_a) == 2 and all(item.period_start == date(2026, 1, 1) for item in h1_grade_a)
        assert compare(ctx, CompareArgs(observation_ids=[item.id for item in h1_grade_a], comparison="submarkets")).compatible


@pytest.mark.parametrize("mutation", [
    lambda s: s.replace("2.2m sq ft", "2.2m square metres"),
    lambda s: s.replace("This equates to a vacancy rate", "This equates to an availability rate"),
    lambda s: s.replace("01 May 2026 Article", "01 May 2025 Article"),
    lambda s: s.replace("Leasing across Central London reached", "Leasing across Central London is forecast to reach"),
    lambda s: s + " This equates to a vacancy rate of 9.1%",
    lambda s: s.replace("vacancy rate of 7.4%", "vacancy rate of 104%"),
])
def test_source_contract_mismatch_fails_closed_before_storage(tmp_path, mutation):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    with Store(tmp_path / "db") as store:
        ctx = context(store, {source.url: page_for(source.id, mutation(text_for(source.id)))})
        with pytest.raises(ValueError):
            collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert store.counts("live")["documents"] == 0 and store.list_observations("live") == []
        assert store.list_source_attempts(ctx.run_id)[0].status == "failed"


def test_changed_supported_value_is_visible_conflict_not_silent_overwrite(tmp_path):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    pages = {source.url: page_for(source.id)}
    with Store(tmp_path / "db") as store:
        ctx = context(store, pages)
        first = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        pages[source.url] = page_for(source.id, text_for(source.id).replace("2.2m sq ft", "2.3m sq ft"))
        second = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert second.counts["observations"] == 1 and second.counts["revisions"] == 1
        assert sorted(item.value for item in store.list_observations("live") if item.metric == "take_up") == [2.2, 2.3]
        assert not compare(ctx, CompareArgs(observation_ids=[first.observation_ids[0], second.observation_ids[0]])).compatible


def test_provider_failure_is_inspectable_and_fixture_cannot_network(tmp_path):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    with Store(tmp_path / "db") as store:
        ctx = context(store, {source.url: ProviderError("source unavailable")})
        with pytest.raises(ProviderError, match="unavailable"):
            collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert store.list_source_attempts(ctx.run_id)[0].error == "source unavailable"
        ctx.mode = "fixture"
        with pytest.raises(ValueError, match="live mode"):
            collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert len(store.list_source_attempts(ctx.run_id)) == 1


def test_unknown_source_identity_and_bounded_document_reads(tmp_path):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    with pytest.raises(ValueError, match="identity"):
        parse_monitored_source(source.id, page_for(source.id, publisher="unreviewed.example"))
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        assert len(monitoring_sources(ctx, MonitoringSourcesArgs(scope="reviewed_articles")).sources) == len(MONITORED_SOURCES)
        with pytest.raises(ValueError, match="unknown monitored"):
            collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id="unapproved"))
        result = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        chunk = read_document(ctx, ReadDocumentArgs(retrieved_id=result.document.id, offset=3, limit=15))
        assert chunk.text == text_for(source.id)[3:18] and chunk.next_offset == 18
        with pytest.raises(ValueError, match="offset"):
            read_document(ctx, ReadDocumentArgs(retrieved_id=result.document.id, offset=99999))
        with pytest.raises(ValueError, match="during this run"):
            read_document(ctx, ReadDocumentArgs(retrieved_id="unknown"))


def test_monitoring_catalogue_requires_explicit_reviewed_article_scope(tmp_path):
    from london_market.tools import bindings
    binding = bindings()["monitoring_sources"]
    assert binding.schema()["function"]["parameters"]["required"] == ["scope"]
    with pytest.raises(ValueError):
        binding.input_schema.model_validate({"reasoning": "Retry with empty args"})
    with pytest.raises(ValueError):
        binding.input_schema.model_validate({"scope": "all_london_current_sources"})
    with Store(tmp_path / "db") as store:
        result = binding.handler(context(store), binding.input_schema.model_validate({"scope": "reviewed_articles"}))
        assert [item.source_id for item in result.sources] == list(MONITORED_SOURCES)
        assert store.list_observations("live") == []


def test_repeat_collection_refreshes_source_qualifications_without_new_rows(tmp_path):
    source = MONITORED_SOURCES[SOURCE_IDS[0]]
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        first = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        counts = store.counts("live")
        evidence_id = first.evidence_ids[1]
        store.set_evidence_validation(evidence_id, "validated", "Old source contract without qualifications")
        repeated = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source.id))
        assert repeated.evidence_ids == first.evidence_ids
        assert all(value == 0 for value in repeated.counts.values())
        assert store.counts("live") == counts
        assert "Source qualifications" in store.get_evidence(evidence_id, "live").validation_reason


def test_source_qualifications_survive_storage_and_evidence_query(tmp_path):
    from london_market.tools import QueryArgs, query_evidence
    source_id = "savills-central-london-q2-2026"
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id=source_id))
        selected = [item.id for item in store.list_evidence("live") if "City prime rent stood at £116.73" in item.excerpt or item.excerpt.startswith("Supply across Central London fell")]
        returned = query_evidence(ctx, QueryArgs(evidence_ids=selected))
        q2 = [item for item in returned.evidence if str(item.source_url) == MONITORED_SOURCES[source_id].url]
        city_prime = next(item for item in q2 if "City prime rent stood at £116.73" in item.excerpt)
        assert "lacks an explicit averaging window" in city_prime.validation_reason
        assert "qualitative only" in city_prime.validation_reason
        supply = next(item for item in q2 if item.excerpt.startswith("Supply across Central London fell"))
        assert supply.excerpt == "Supply across Central London fell to 18.7 million sq ft during the quarter, down 0.5 million sq ft on Q1 2026."
        assert "19.1m and 18.7m imply a 0.4m change" in supply.validation_reason
        assert "Q2 narrative says 0.5m" in supply.validation_reason
        assert "no precise equivalence to marketed availability" in supply.validation_reason
        assert all(store.get_evidence(item.id).validation_reason == item.validation_reason for item in q2)


def test_q2_pipeline_excerpts_retain_h1_starts_and_forecast_denominator(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id="savills-central-london-q2-2026"))
        evidence = store.list_evidence("live")
        starts = next(item for item in evidence if "Delta at Paddington" in item.excerpt)
        assert "development starts fell sharply in H1 2026" in starts.excerpt
        assert "1.5 million sq ft commencing" in starts.excerpt
        assert "remaining 14 starts were refurbishments" in starts.excerpt
        future_supply = next(item for item in evidence if "28% of this future supply" in item.excerpt)
        assert "expected to reach 22 million sq ft by the end of 2029" in future_supply.excerpt
        assert "148 schemes scheduled for completion" in future_supply.excerpt
        assert "source forecasts, not current availability" in future_supply.validation_reason
        assert future_supply.statement_kind == "forecast"
        assert starts.statement_kind is None


def test_stored_qualifications_are_specific_to_the_measurement(tmp_path):
    with Store(tmp_path / "db") as store:
        ctx = context(store)
        result = collect_monitored_source(ctx, CollectMonitoredSourceArgs(source_id="savills-central-london-q2-2026"))
        take_up = store.get_evidence(result.evidence_ids[0])
        assert "Source qualifications" not in take_up.validation_reason
        assert "0.4m" not in take_up.validation_reason
        h1_prime = next(item for item in store.list_evidence("live") if "West End average prime rent at" in item.excerpt)
        assert "H1 January-June average" in h1_prime.validation_reason
        assert "City prime rent lacks" not in h1_prime.validation_reason
        starts = next(item for item in store.list_evidence("live") if "Delta at Paddington" in item.excerpt)
        assert "source forecasts" not in starts.validation_reason






@pytest.mark.parametrize('metadata_date', [None, date(2026, 9, 30)])
def test_existing_dated_body_header_cannot_be_replaced_by_stale_document_title(metadata_date):
    from tests.test_weekly_sources import source_page
    source_id = 'savills-central-london-september-2026'
    original = source_page(source_id, published_at=metadata_date)
    assert parse_monitored_source(source_id, original).observations[0].value == 557416
    changed = original.model_copy(update={'text': original.text.replace('30 September 2026', '01 October 2026')})
    with pytest.raises(ValueError, match='publication date'):
        parse_monitored_source(source_id, changed)
