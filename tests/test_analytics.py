import hashlib
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import ChartPayload, Document, Evidence, Observation, ObservationPair, stable_id
from london_market.fixtures import load_fixtures
from london_market.store import Store
from london_market.providers import FetchedPage
from london_market.sources import MONITORED_SOURCES, parse_monitored_source
from london_market.tools import MarketInventoryArgs, MonitoringSourcesArgs, PrepareArgs, QueryArgs, ToolContext, VisualizationArgs, choose_visualization, visual_purposes, market_inventory, monitoring_sources, prepare_visualization_data, query_evidence


@pytest.fixture
def context(tmp_path):
    with Store(tmp_path / "market.sqlite3") as store:
        load_fixtures(store)
        yield ToolContext(Settings(), store, None, "fixture", "analytics-test", 5)


def measurement(ctx, **changes):
    base = ctx.store.list_observations("fixture")[0]
    fields = {**base.model_dump(exclude={"id", "revision_of"}), "metric": "analytic_take_up", "unit": "sq ft", "definition": "Office space leased during the reference quarter", "category": "take-up", "native_geography": "Central London", "mapped_geography": "Central London", "period_start": date(2026, 1, 1), "period_end": date(2026, 3, 31), **changes}
    item = Observation(**fields)
    source = ctx.store.get_evidence(item.evidence_ids[0], "fixture")
    ctx.store.ingest(ctx.store.get_document(source.document_id, "fixture"), [], [item])
    return ctx.store.get_observation(item.id, "fixture")


def exhibit(ctx, items=(), preferred="total", **kwargs):
    plan = choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in items], title="Market evidence", purpose="Inspect reported market measurements", preferred=preferred, **kwargs))
    return prepare_visualization_data(ctx, PrepareArgs(plan_id=plan.plan_id)).payload


def test_existing_observation_id_is_preserved_and_components_are_distinct(context):
    old = context.store.list_observations("fixture")[0]
    expected_key = stable_id("obskey", old.mode, old.metric, old.publisher, old.definition, old.unit, old.category, old.native_geography, old.mapped_geography, old.period_start, old.period_end)
    assert old.semantic_key == expected_key
    assert old.id == stable_id("obs", expected_key, old.value)
    first = measurement(context, value=23.5, dimension="occupier_sector", segment="Media & Tech", partition="reviewed-sector-shares")
    second = measurement(context, value=23.5, dimension="occupier_sector", segment="Services", partition="reviewed-sector-shares")
    assert first.id != second.id and second.revision_of is None
    with pytest.raises(ValidationError, match="together"):
        Observation(**{**old.model_dump(exclude={"id"}), "dimension": "occupier_sector"})


def test_inventory_reads_persisted_values_conflicts_and_bounded_continuations(context):
    q1 = measurement(context, value=100)
    q2 = measurement(context, value=120, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    conflict = measurement(context, value=130, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    result = market_inventory(context, MarketInventoryArgs(metric="analytic_take_up", points_per_series=1))
    assert result.series_total == 1 and result.observation_total == 3 and result.conflict_total == 1
    series = result.series[0]
    assert series.value_total == 3 and series.next_point_offset == 1 and result.truncated
    assert set(series.conflicts[0].observation_ids) == {q2.id, conflict.id}
    page = market_inventory(context, MarketInventoryArgs(metric="analytic_take_up", points_per_series=1, point_offset=2))
    assert page.series[0].values[0].observation_id == q1.id and page.series[0].next_point_offset is None
    for value in series.values + page.series[0].values:
        saved = context.store.get_observation(value.observation_id, "fixture")
        assert value.value == saved.value and value.evidence_ids == saved.evidence_ids
    assert {row.geography: row.observation_count for row in result.coverage} == {geography: sum(item.mapped_geography == geography for item in context.store.list_observations("fixture")) for geography in ("City", "West End", "Canary Wharf", "Midtown and Fringe", "Wider London")}
    assert result.coverage == market_inventory(context, MarketInventoryArgs()).coverage


def test_inventory_qualitative_samples_remain_inspectable_and_separate(context):
    base = context.store.list_evidence("fixture")[0]
    source = Evidence(**{**base.model_dump(exclude={"id"}), "excerpt": "A long source-reported occupier demand finding. " * 10, "category": "hybrid_working", "validation_status": "candidate", "validation_reason": None})
    context.store.ingest(context.store.get_document(source.document_id, "fixture"), [source], [])
    result = market_inventory(context, MarketInventoryArgs(metric="hybrid_working"))
    assert result.observation_total == 0
    sample = result.qualitative[0].samples[0]
    assert sample.evidence_id == source.id and sample.validation_status == "candidate"
    assert len(sample.excerpt) == 240 and context.store.get_evidence(sample.evidence_id).excerpt.startswith(sample.excerpt)
    assert sample.published_at == source.published_at
    detail = query_evidence(context, QueryArgs(evidence_ids=[sample.evidence_id]))
    assert [item.id for item in detail.evidence] == [source.id]
    assert detail.evidence[0].excerpt == source.excerpt
    assert detail.observations == []


def test_targeted_evidence_lookup_retains_observation_links_and_unresolved_conflicts(context):
    original = measurement(context, value=100)
    source = context.store.get_evidence(original.evidence_ids[0])
    other_source = Evidence(**{**source.model_dump(exclude={"id"}), "excerpt": "The publisher reports a conflicting later value."})
    context.store.ingest(context.store.get_document(source.document_id), [other_source], [])
    disputed = measurement(context, value=120, evidence_ids=[other_source.id])
    detail = query_evidence(context, QueryArgs(metric="analytic_take_up", evidence_ids=[source.id]))
    assert [item.id for item in detail.evidence] == [source.id]
    assert [item.id for item in detail.observations] == [original.id]
    assert set(detail.conflicts[0].observation_ids) == {original.id, disputed.id}
    assert query_evidence(context, QueryArgs(evidence_ids=["unknown-source"])).evidence == []


def test_inventory_paginates_series_without_loading_full_excerpts(context):
    for index in range(3):
        measurement(context, value=index + 1, segment=f"Sector {index}", dimension="occupier_sector")
    first = market_inventory(context, MarketInventoryArgs(metric="analytic_take_up", limit=2))
    second = market_inventory(context, MarketInventoryArgs(metric="analytic_take_up", limit=2, offset=first.next_offset))
    assert len(first.series) == 2 and len(second.series) == 1
    assert second.next_offset is None and first.series_total == 3
    assert {item.series_id for item in first.series}.isdisjoint(item.series_id for item in second.series)


def test_total_is_reported_and_refuses_mixed_units_aggregate_overlap_and_conflicts(context):
    city = measurement(context, value=100, native_geography="City", mapped_geography="City")
    west = measurement(context, value=120, native_geography="West End", mapped_geography="West End")
    result = exhibit(context, [city, west])
    assert result.chart_type == "total" and [point.value for point in result.points] == [100, 120]
    assert set(result.observation_ids) == {city.id, west.id}
    mixed = measurement(context, value=1, unit="million sq ft", native_geography="West End", mapped_geography="West End")
    assert exhibit(context, [city, mixed]).chart_type == "table"
    aggregate = measurement(context, value=1000)
    assert exhibit(context, [city, aggregate]).chart_type == "table"
    measurement(context, value=999, native_geography="City", mapped_geography="City")
    assert exhibit(context, [city, west]).chart_type == "insufficient_data"


def test_dated_totals_preserve_reported_values_and_refuse_overlap_or_mixed_measurements(context):
    q1 = measurement(context, metric="reported_supply", value=19.1, unit="million sq ft", definition="Reported office supply at quarter end", category="availability")
    q2 = measurement(context, metric="reported_supply", value=18.7, unit="million sq ft", definition=q1.definition, category=q1.category, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    result = exhibit(context, [q2, q1], "total")
    assert result.chart_type == "total"
    assert sorted(point.value for point in result.points) == [18.7, 19.1]
    assert {(point.period_start, point.period_end) for point in result.points} == {(q1.period_start, q1.period_end), (q2.period_start, q2.period_end)}
    assert result.unit == "million sq ft" and len(result.points) == 2
    assert all(context.store.get_observation(point.observation_id).evidence_ids == point.evidence_ids for point in result.points)
    from london_market.report import displayed_flow_total
    assert displayed_flow_total(result) is None
    assert any("net stock change" in caveat for caveat in result.caveats)
    overlap = measurement(context, metric=q1.metric, value=19.0, unit=q1.unit, definition=q1.definition, category=q1.category, period_start=date(2026, 3, 1), period_end=date(2026, 5, 31))
    declined = exhibit(context, [q1, overlap], "total")
    assert declined.chart_type == "table" and any("overlap" in caveat.lower() for caveat in declined.caveats)
    unrelated = measurement(context, metric="under_construction", value=16.1, unit=q1.unit, definition="Space under construction", category="pipeline", period_start=q2.period_start, period_end=q2.period_end)
    assert exhibit(context, [q1, unrelated], "total").chart_type == "table"
    west = measurement(context, metric=q1.metric, value=5, unit=q1.unit, definition=q1.definition, category=q1.category, native_geography="West End", mapped_geography="West End", period_start=q2.period_start, period_end=q2.period_end)
    city = measurement(context, metric=q1.metric, value=6, unit=q1.unit, definition=q1.definition, category=q1.category, native_geography="City", mapped_geography="City")
    geographic = exhibit(context, [city, west], "total")
    assert geographic.chart_type == "table" and any("shared period" in caveat for caveat in geographic.caveats)
    q3 = measurement(context, metric=q1.metric, value=18.5, unit=q1.unit, definition=q1.definition, category=q1.category, period_start=date(2026, 7, 1), period_end=date(2026, 9, 30))
    assert exhibit(context, [q1, q3], "total").chart_type == "total"
    assert exhibit(context, [q1, q3], "trend").chart_type == "table"


def test_chart_fallback_offers_bounded_measurement_ids_and_specific_recovery(context):
    vacancy = measurement(context, metric="vacancy_rate", value=7.4, unit="%", definition="Vacancy at quarter end", category="vacancy")
    supply = measurement(context, metric="reported_supply", value=19.1, unit="million sq ft", definition="Reported supply at quarter end", category="availability")
    result = exhibit(context, [vacancy, supply], "trend")
    assert result.chart_type == "table"
    recovery = " ".join(result.caveats)
    assert "Recovery" in recovery and "one metric" in recovery
    assert vacancy.id in recovery and supply.id in recovery
    relation = exhibit(context, [vacancy, supply], "relation")
    assert relation.chart_type == "table" and any("pairs" in caveat for caveat in relation.caveats)


def test_partial_reported_shares_preserve_source_segments_without_inventing_other(context):
    sectors = [measurement(context, metric="sector_share", value=value, unit="%", definition="Source-reported share of Central London quarterly take-up by occupier sector", dimension="occupier_sector", segment=segment, partition="reviewed-sector-share") for segment, value in (("Media & Tech", 23.5), ("Services", 19.8), ("Banking & Finance", 17))]
    result = exhibit(context, sectors, "proportion")
    assert result.chart_type == "proportion" and len(result.points) == 3
    assert [point.value for point in result.points] == [23.5, 19.8, 17]
    assert [point.segment for point in result.points] == ["Media & Tech", "Services", "Banking & Finance"]
    assert all(point.geography == "Central London" for point in result.points)
    assert result.denominator is None and "remainder" in result.caveats[0]
    assert all(point.observation_id in result.observation_ids for point in result.points)
    assert set(result.source_refs) == {eid for item in sectors for eid in item.evidence_ids}
    unreviewed = [measurement(context, metric="unreviewed_share", value=10, unit="%", dimension="occupier_sector", segment=name) for name in ("A", "B")]
    assert exhibit(context, unreviewed, "proportion").chart_type == "table"
    single = measurement(context, metric="grade_share", value=78, unit="%", dimension="office_grade", segment="Grade A")
    assert exhibit(context, [single], "proportion").chart_type == "proportion"


def test_amount_proportions_remain_tables_without_a_reviewed_subset_contract(context):
    sectors = [measurement(context, value=value, dimension="occupier_sector", segment=name, partition="reviewed-sector-amounts") for name, value in (("A", 20), ("B", 30))]
    assert exhibit(context, sectors, "proportion").chart_type == "table"
    total = measurement(context, metric="reported_take_up_total", value=100)
    result = exhibit(context, sectors, "proportion", denominator_observation_id=total.id)
    assert result.chart_type == "table" and result.denominator.observation_id == total.id
    assert "unreviewed denominator contract" in result.caveats[0].lower()
    assert [point.value for point in result.points] == [20, 30]
    assert total.id in result.observation_ids
    small = measurement(context, metric="reported_take_up_total", value=40, publisher="Other")
    assert exhibit(context, sectors, "proportion", denominator_observation_id=small.id).chart_type == "table"
    too_small = measurement(context, metric="smaller_total", value=40)
    assert exhibit(context, sectors, "proportion", denominator_observation_id=too_small.id).chart_type == "table"


def test_completion_flow_is_not_a_proportion_of_construction_stock_even_in_the_same_source(context):
    flow = measurement(context, metric="new_build_completions", value=2, unit="million sq ft", definition="New-build space completed during the quarter", category="pipeline", dimension="completion_type", segment="New builds")
    stock = measurement(context, metric="under_construction", value=16.1, unit="million sq ft", definition="Space under construction at quarter end, including prelets", category="pipeline")
    assert flow.evidence_ids == stock.evidence_ids
    result = exhibit(context, [flow], "proportion", denominator_observation_id=stock.id)
    assert result.chart_type == "table"
    assert "unreviewed denominator contract" in result.caveats[0].lower()
    assert result.points[0].value == 2 and result.denominator.value == 16.1
    assert set(result.observation_ids) == {flow.id, stock.id}


def test_relationship_has_typed_aligned_pairs_independent_axes_and_source_links(context):
    pairs = []
    for month, end, stock, flow in ((1, 31, 100, 20), (2, 28, 110, 25), (3, 31, 90, 28)):
        period = {"period_start": date(2026, month, 1), "period_end": date(2026, month, end)}
        x = measurement(context, metric="reported_vacancy_rate", value=stock / 10, unit="%", definition="Vacancy at month end", category="vacancy", **period)
        y = measurement(context, value=flow, **period)
        pairs.append(ObservationPair(x_observation_id=x.id, y_observation_id=y.id))
    result = exhibit(context, preferred="relation", pairs=pairs)
    assert result.chart_type == "relation" and len(result.pairs) == 3
    assert result.x_unit == "%" and result.y_unit == "sq ft"
    assert result.x_label == "reported_vacancy_rate" and result.y_label == "analytic_take_up"
    assert "causation" in result.caveats[0] and "Stocks" in result.caveats[1]
    for pair in result.pairs:
        assert pair.x.period_start == pair.y.period_start
        assert pair.x.evidence_ids == context.store.get_observation(pair.x.observation_id).evidence_ids
        assert pair.y.evidence_ids == context.store.get_observation(pair.y.observation_id).evidence_ids
    assert result.source_refs and len(result.observation_ids) == 6
    assert exhibit(context, preferred="relation", pairs=pairs[:2]).chart_type == "table"
    wrong = measurement(context, value=50, period_start=date(2026, 4, 1), period_end=date(2026, 4, 30))
    mismatch = pairs[:-1] + [ObservationPair(x_observation_id=pairs[-1].x_observation_id, y_observation_id=wrong.id)]
    assert exhibit(context, preferred="relation", pairs=mismatch).chart_type == "table"


def test_monitoring_catalogue_exposes_source_metadata(context):
    result = monitoring_sources(context, MonitoringSourcesArgs(scope="reviewed_articles"))
    assert result.sources
    assert all(source.publisher and source.source_kind and source.topics and source.geographies for source in result.sources)




def test_inventory_does_not_hide_older_secondary_reporting(tmp_path):
    from tests.test_source_catalogue import page_for
    with Store(tmp_path / 'sources.db') as store:
        for sid in ('deutsche-bank-canary-wharf-2026', 'lendlease-stratford-cross-2026', 'cityam-london-polarisation-2026'):
            parsed = parse_monitored_source(sid, page_for(sid))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        ctx = ToolContext(Settings(),store,None,'live','inventory',5)
        result = market_inventory(ctx,MarketInventoryArgs(metric='news'))
        samples = next(category.samples for category in result.qualitative if category.category=='news')
        assert any(item.publisher=='City AM' and item.source_kind=='secondary_news' for item in samples)
        assert len(samples) <= 3
        assert all(item.qualification for item in samples)


def test_exact_evidence_lookup_returns_requested_excerpts_and_names_missing(tmp_path):
    from tests.test_source_catalogue import page_for
    from london_market.tools import query_evidence, QueryArgs
    with Store(tmp_path / "sources.db") as store:
        for sid in ("savills-central-london-q2-2026", "boe-september-2026"):
            parsed = parse_monitored_source(sid, page_for(sid, (Path("tests/fixtures/live_sources/q2.txt").read_text() if sid == "savills-central-london-q2-2026" else None)))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        ids = [item.id for item in store.list_evidence("live")][:11]
        result = query_evidence(ToolContext(Settings(),store,None,"live","exact",5),QueryArgs(evidence_ids=ids+["ev_missing"],limit=1))
        assert {item.id for item in result.evidence} == set(ids)
        assert result.missing_evidence_ids == ["ev_missing"]
        assert len(result.observations) <= 1


def test_inventory_relationship_candidate_prepares_actual_aligned_points(tmp_path):
    from tests.test_source_catalogue import page_for
    with Store(tmp_path / "sources.db") as store:
        for sid in ("savills-central-london-q3-2025", "savills-central-london-q4-2025", "savills-central-london-q1-2026", "savills-central-london-q2-2026"):
            parsed = parse_monitored_source(sid, page_for(sid, Path(f"tests/fixtures/live_sources/{sid.split('-')[-2]}.txt").read_text() if sid.endswith("2026") else None))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        ctx = ToolContext(Settings(),store,None,"live","relation",5)
        candidate = market_inventory(ctx,MarketInventoryArgs()).relationship_candidates[0]
        plan = choose_visualization(ctx,VisualizationArgs(title="Vacancy and take-up",purpose="Descriptive association",preferred="relation",pairs=candidate.pairs))
        chart = prepare_visualization_data(ctx,PrepareArgs(plan_id=plan.plan_id)).payload
        assert chart.chart_type == "relation"
        assert [(point.x.value,point.y.value) for point in chart.pairs] == [(7.8,1.9),(7.4,2.63),(7.4,2.2),(7.2,2.4)]





@pytest.mark.parametrize('changes', [
    {'period_start': date(2026, 7, 1), 'period_end': date(2026, 9, 30)},
    {'publisher': 'Different publisher'},
    {'definition': 'Different measurement'},
    {'period_start': date(2026, 3, 1), 'period_end': date(2026, 5, 31)},
])
def test_purpose_history_requires_actual_compatible_contiguous_window(context, changes):
    first = measurement(context, metric='vacancy_rate', unit='%', value=8)
    second = measurement(context, metric=first.metric, unit=first.unit, value=7.5, **{'period_start': date(2026, 4, 1), 'period_end': date(2026, 6, 30), **changes})
    result = visual_purposes([first, second])
    assert result['trend']['status'] == 'unsupported'
    assert all(entry['status'] == 'unsupported' for entry in result.values())


def test_purpose_conflicts_stock_and_valid_subwindow_remain_honest(context):
    first = measurement(context, metric='available_space', value=100)
    second = measurement(context, metric='available_space', value=120, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    assert visual_purposes([first, second])['sum']['status'] == 'unsupported'
    conflict = measurement(context, metric=first.metric, value=130, period_start=second.period_start, period_end=second.period_end)
    assert visual_purposes([first, second, conflict])['trend']['status'] == 'unsupported'
    later = measurement(context, metric=first.metric, value=140, period_start=date(2026, 10, 1), period_end=date(2026, 12, 31))
    assert visual_purposes([first, second, later])['trend']['candidate']['observation_ids'] == [first.id, second.id]


def test_purpose_relationship_and_partition_controls(context):
    from datetime import timedelta
    vacancies, take_up = [], []
    periods = [(date(2026, 1, 1), date(2026, 3, 31)), (date(2026, 4, 1), date(2026, 6, 30)), (date(2026, 7, 1), date(2026, 9, 30))]
    for index, (start, end) in enumerate(periods):
        vacancies.append(measurement(context, metric='vacancy_rate', value=8-index, unit='%', period_start=start, period_end=end))
        take_up.append(measurement(context, metric='take_up', value=index+1, period_start=start, period_end=end))
    assert visual_purposes(vacancies+take_up)['relationship']['status'] == 'missing'
    wrong = take_up[-1].model_copy(update={'period_start': take_up[-1].period_start+timedelta(days=1)})
    assert visual_purposes(vacancies+take_up[:-1]+[wrong])['relationship']['status'] == 'unsupported'
    repeats = [item.model_copy(update={'value': 1}) for item in vacancies+take_up]
    assert visual_purposes(repeats)['relationship']['status'] == 'unsupported'
    shares = [measurement(context, metric='occupier_take_up_share', value=value, unit='%', dimension='occupier_sector', segment=segment, partition='reviewed-sector-shares') for value, segment in [(20, 'Finance'), (30, 'Tech')]]
    assert visual_purposes(shares)['proportion']['status'] == 'missing'
    overlapping = [shares[0], shares[1].model_copy(update={'segment': shares[0].segment})]
    assert visual_purposes(overlapping)['proportion']['status'] == 'unsupported'
    assert visual_purposes([item.model_copy(update={'partition': None}) for item in shares])['proportion']['status'] == 'unsupported'



def test_optional_completion_exhibits_cannot_discharge_uncovered_required_purposes(context):
    from london_market.report import displayed_flow_total
    first = measurement(context, metric='development_completions', category='pipeline', definition='Office development and major refurbishment completions during the reference period', unit='million sq ft', value=1.4)
    second = measurement(context, metric=first.metric, category=first.category, definition=first.definition, unit=first.unit, value=1.4, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    chart = exhibit(context, [first, second], preferred='total')
    assert chart.chart_type == 'total'
    assert displayed_flow_total(chart) == 'Calculated sum of displayed periods: 2.8 million sq ft'
    line = exhibit(context, [first, second], preferred='trend')
    assert line.chart_type == 'line'
    assert [point.value for point in line.points] == [1.4, 1.4]
    trend = visual_purposes([first, second], {'optional-completion-line': line})['trend']
    assert trend['status'] == 'unsupported' and trend['chart_ids'] == []
    assert trend['reason'] == 'No compatible contiguous history supports a trend graph for the report’s covered topics or local areas.'
    assert trend['candidate'] is None

    expected = 'No compatible nonoverlapping take-up history supports the required displayed-period leasing sum.'
    available = visual_purposes([first, second])['sum']
    assert available['status'] == 'unsupported'
    assert available['reason'] == expected and available['candidate'] is None
    selected = visual_purposes([first, second], {'optional-completion-chart': chart})['sum']
    assert selected['status'] == 'unsupported' and selected['chart_ids'] == []



def test_required_trend_can_cover_a_local_area_without_assigning_mixed_completions_to_a_topic(context):
    first = measurement(context, metric='development_completions', value=1.4, category='pipeline', native_geography='City', mapped_geography='City')
    second = measurement(context, metric=first.metric, value=1.4, category=first.category, native_geography=first.native_geography, mapped_geography=first.mapped_geography, period_start=date(2026, 4, 1), period_end=date(2026, 6, 30))
    observations = [first, second]
    assert visual_purposes(observations)['trend']['status'] == 'missing'
    line = exhibit(context, observations, preferred='trend')
    assert line.chart_type == 'line'
    assert [point.value for point in line.points] == [1.4, 1.4]
    result = visual_purposes(observations, {'prepared-local-line': line})
    assert result['trend']['status'] == 'included'
    assert result['sum']['status'] == 'unsupported'
