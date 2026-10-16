import hashlib
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from london_market.domain import Document, Evidence, compatibility, stable_id
from london_market.providers import FetchedPage
from london_market.sources import MONITORED_SOURCES, parse_monitored_source
from london_market.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "source_catalogue"
EXPECTED = {
    "savills-central-london-q3-2025": [1.9, 7.8, 7.4, 8.3, 8.3],
    "savills-central-london-q4-2025": [2.63, 7.4, 7.0, 7.9, 8.2, 26, 18],
    "bnp-central-london-q1-2026": [2.5, 1.0, 0.8, 8.4, 6.8, 9.8, 23.5, 19.8, 17, 78, 73],
    "boe-september-2026": [3.75],
    "ons-hybrid-work-2025": [28],
    "gla-london-economy": [],
    "lendlease-stratford-cross-2026": [],
    "lendlease-turing-stratford-2025": [],
    "cityam-london-polarisation-2026": [],
    "deutsche-bank-canary-wharf-2026": [],
}


def page_for(source_id, text=None, published=None):
    source = MONITORED_SOURCES[source_id]
    text = text or (FIXTURES / f"{source_id}.txt").read_text()
    document = Document(mode="live", url=source.url, title=source.header,
                        publisher=urlsplit(source.url).hostname,
                        content_hash=hashlib.sha256(text.encode()).hexdigest(), published_at=published)
    return FetchedPage(document=document, text=text)


@pytest.mark.parametrize("source_id", EXPECTED)
def test_reviewed_excerpts_yield_exact_observations(source_id):
    page = page_for(source_id)
    parsed = parse_monitored_source(source_id, page)
    assert [item.value for item in parsed.observations] == EXPECTED[source_id]
    assert parsed.document.published_at == MONITORED_SOURCES[source_id].published_at
    assert all(item.excerpt in page.text and item.validation_status == "validated" for item in parsed.evidence)
    assert all(item.publisher == MONITORED_SOURCES[source_id].publisher for item in parsed.observations)
    assert all(item.event_date is None for item in parsed.evidence)


def test_new_sources_repeat_without_duplicates_and_preserve_readback(tmp_path):
    with Store(tmp_path / "sources.sqlite3") as store:
        for source_id in EXPECTED:
            parsed = parse_monitored_source(source_id, page_for(source_id))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        assert store.counts("live") == {"documents": 10, "evidence": 40, "observations": 25, "reports": 0, "runs": 0}
        for source_id in EXPECTED:
            parsed = parse_monitored_source(source_id, page_for(source_id))
            assert store.ingest(parsed.document, parsed.evidence, parsed.observations) == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
        wider = [item for item in store.list_observations("live") if item.mapped_geography == "Wider London"]
        assert {(item.native_geography, item.value) for item in wider} == {("Hammersmith", 26), ("VNEB", 18)}
        sectors = [item for item in store.list_observations("live") if item.dimension == "occupier_sector"]
        assert {item.segment for item in sectors} == {"Media & Tech", "Services", "Banking & Finance"}
        assert len({item.id for item in sectors}) == 3


@pytest.mark.parametrize("replacement", ["Total take-up is forecast to reach 2.5m sq ft", "Total take-up reached 2.5m square metres"])
def test_changed_measurement_contract_rejects_entire_source(tmp_path, replacement):
    sid = "bnp-central-london-q1-2026"
    text = page_for(sid).text.replace("Total take-up reached 2.5m sq ft", replacement)
    with Store(tmp_path / "rejected.sqlite3") as store:
        with pytest.raises(ValueError, match="measurement phrase"):
            parsed = parse_monitored_source(sid, page_for(sid, text))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        assert store.counts("live")["documents"] == 0


def test_independent_publisher_numbers_cannot_be_spliced():
    bnp = parse_monitored_source("bnp-central-london-q1-2026", page_for("bnp-central-london-q1-2026")).observations
    savills = parse_monitored_source("savills-central-london-q4-2025", page_for("savills-central-london-q4-2025")).observations
    first = next(item for item in savills if item.metric == "vacancy_rate" and item.native_geography == "Central London")
    second = next(item for item in bnp if item.metric == "vacancy_rate" and item.native_geography == "Central London")
    assert compatibility(first, second) == ["publisher", "definition"]


def test_partial_sector_shares_and_different_grade_denominators_stay_distinct():
    parsed = parse_monitored_source("bnp-central-london-q1-2026", page_for("bnp-central-london-q1-2026"))
    sectors = [item for item in parsed.observations if item.dimension == "occupier_sector"]
    assert round(sum(item.value for item in sectors), 1) == 60.3
    assert {item.partition for item in sectors} == {"bnp-q1-2026-central-london-occupier-sector-share"}
    assert not any(item.segment == "Other" for item in parsed.observations)
    grades = [item for item in parsed.observations if item.dimension == "office_grade"]
    assert {item.metric for item in grades} == {"take_up_grade_share", "supply_grade_share"}
    assert "metric" in compatibility(*grades)


def test_macro_context_is_not_relabelled_as_london_or_new_evidence_period():
    ons = parse_monitored_source("ons-hybrid-work-2025", page_for("ons-hybrid-work-2025", published=date(2025, 6, 10)))
    observation = ons.observations[0]
    assert (observation.native_geography, observation.mapped_geography) == ("Great Britain", "Unmapped")
    assert (observation.period_start, observation.period_end) == (date(2025, 1, 8), date(2025, 3, 30))
    assert ons.document.published_at == date(2025, 6, 11)
    assert "metadata reports 10 June" in ons.evidence[0].validation_reason
    boe = parse_monitored_source("boe-september-2026", page_for("boe-september-2026"))
    assert boe.document.published_at == date(2026, 9, 17)
    assert boe.observations[0].period_start == boe.observations[0].period_end == date(2026, 9, 16)
    assert boe.observations[0].native_geography == "United Kingdom"
    assert boe.observations[0].mapped_geography == "Unmapped"


def test_qualitative_news_forecasts_and_unknown_publication_are_explicit():
    gla = parse_monitored_source("gla-london-economy", page_for("gla-london-economy"))
    assert gla.document.published_at is None and not gla.observations
    assert "2024 and 2025 are GLA estimates" in gla.evidence[2].validation_reason
    canary = parse_monitored_source("deutsche-bank-canary-wharf-2026", page_for("deutsche-bank-canary-wharf-2026"))
    assert canary.evidence[0].mapped_geography == "Canary Wharf"
    assert "planned occupation in early 2028" in canary.evidence[0].validation_reason
    assert "target sustainability performance" in canary.evidence[1].validation_reason
    stratford = parse_monitored_source("lendlease-stratford-cross-2026", page_for("lendlease-stratford-cross-2026"))
    assert "not consented development or completed office supply" in stratford.evidence[0].validation_reason
    secondary = parse_monitored_source("cityam-london-polarisation-2026", page_for("cityam-london-polarisation-2026"))
    assert len(secondary.evidence[0].excerpt.split()) <= 25
    assert "Secondary City AM reporting attributed to CoStar" in secondary.evidence[0].validation_reason
    assert not secondary.observations


def test_catalogue_accounts_for_all_remit_topics_with_discovery_hints():
    topics = {topic for source in MONITORED_SOURCES.values() for topic in source.topics}
    assert topics >= {"rents", "vacancy_availability", "take_up", "supply_pipeline", "submarket_dynamics", "macro", "occupier_demand", "news"}
    geographies = {geography for source in MONITORED_SOURCES.values() for geography in source.geographies}
    assert geographies >= {"City", "West End", "Canary Wharf", "Midtown and Fringe", "Wider London"}
    assert {source.source_kind for source in MONITORED_SOURCES.values()} >= {"primary_research", "primary_news", "secondary_news", "official_statistics", "official_analysis"}
    assert all(source.discovery_query and source.discovery_urls for source in MONITORED_SOURCES.values())


def test_dynamic_html_changes_do_not_create_new_reviewed_evidence(tmp_path):
    sid = "bnp-central-london-q1-2026"
    first_page = page_for(sid)
    second_page = page_for(sid, first_page.text + " Rotating advertisement and request-specific token 98765.")
    first = parse_monitored_source(sid, first_page)
    second = parse_monitored_source(sid, second_page)
    assert first.document.raw_content_hash != second.document.raw_content_hash
    assert first.document.id == second.document.id
    assert [item.id for item in first.evidence] == [item.id for item in second.evidence]
    with Store(tmp_path / "stable.sqlite3") as store:
        store.ingest(first.document, first.evidence, first.observations)
        assert store.ingest(second.document, second.evidence, second.observations) == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
        revised = parse_monitored_source(sid, page_for(sid, first_page.text.replace("Total take-up reached 2.5m sq ft", "Total take-up reached 2.6m sq ft")))
        assert revised.document.id != first.document.id
        counts = store.ingest(revised.document, revised.evidence, revised.observations)
        assert counts == {"documents": 1, "evidence": 11, "observations": 1, "revisions": 1}
        assert sorted(item.value for item in store.list_observations("live") if item.metric == "take_up" and item.native_geography == "Central London") == [2.5, 2.6]


def test_boe_actual_cpi_and_future_outlook_have_separate_reviewed_types(tmp_path):
    sid = "boe-september-2026"
    parsed = parse_monitored_source(sid, page_for(sid))
    actual = next(item for item in parsed.evidence if item.statement_kind == "fact")
    forecast = next(item for item in parsed.evidence if item.statement_kind == "forecast")
    assert actual.excerpt == "UK CPI inflation increased to 3.1% in August"
    assert forecast.excerpt == "and is likely to rise further over coming quarters."
    assert "antecedent is UK CPI inflation" in forecast.validation_reason
    with Store(tmp_path / "statement-kinds.sqlite3") as store:
        store.ingest(parsed.document, parsed.evidence, parsed.observations)
        assert store.get_evidence(actual.id).statement_kind == "fact"
        assert store.get_evidence(forecast.id).statement_kind == "forecast"
        repeated = parse_monitored_source(sid, page_for(sid, page_for(sid).text + " Changing unapproved footer."))
        assert store.ingest(repeated.document, repeated.evidence, repeated.observations) == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}


def test_announced_db_occupation_plan_retains_exact_headline_and_field():
    sid = "deutsche-bank-canary-wharf-2026"
    parsed = parse_monitored_source(sid, page_for(sid))
    planned = [item for item in parsed.evidence if item.statement_kind == "fact"]
    assert [item.excerpt for item in planned] == ["Deutsche Bank to take space at YY London, Canary Wharf, from early 2028", "Planned occupation: Early 2028"]
    assert all("Fact of" in item.validation_reason for item in planned)
    assert all("plan" in item.validation_reason for item in planned)
    assert all(item.event_date is None for item in planned)
    assert not parsed.observations


def test_legacy_evidence_ids_validate_without_new_statement_kind():
    source = parse_monitored_source("boe-september-2026", page_for("boe-september-2026")).evidence[0]
    legacy_id = stable_id("ev", source.mode, source.document_id, source.excerpt, source.event_date, source.native_geography, source.mapped_geography, source.category)
    legacy = Evidence.model_validate({**source.model_dump(exclude={"id", "statement_kind"}), "id": legacy_id})
    assert legacy.statement_kind is None and legacy.id == legacy_id
    classified = Evidence.model_validate({**legacy.model_dump(exclude={"id"}), "statement_kind": "forecast"})
    assert classified.id != legacy.id


def test_only_affected_source_contracts_change_version():
    affected = {'savills-central-london-q1-2026', "savills-central-london-q2-2026", "boe-september-2026", "deutsche-bank-canary-wharf-2026", "bnp-central-london-q1-2026", "gla-london-economy", "ons-regional-labour-september-2026"}
    added_contracts = {"savills-central-london-august-2026", "savills-central-london-september-2026", "london-fire-lambeth-hq-october-2026", "knight-frank-london-commentary-october-2026", "semafor-canary-wharf-october-2026", "cyprus-mail-uk-defence-october-2026"}
    earlier_contracts = set(MONITORED_SOURCES) - added_contracts
    assert {source.id for source in MONITORED_SOURCES.values() if source.id in earlier_contracts and source.version != "2"} == affected
    assert MONITORED_SOURCES["savills-central-london-q2-2026"].version == "6"
    assert all(source.version == "2" for source in MONITORED_SOURCES.values() if source.id in earlier_contracts and source.id not in affected)


def test_ons_london_resident_levels_preserve_period_scope_and_metadata_discrepancy():
    sid = 'ons-regional-labour-september-2026'
    parsed = parse_monitored_source(sid, page_for(sid, published=date(2026, 9, 14)))
    assert [(item.metric, item.value) for item in parsed.observations] == [('employment_rate', 73.9), ('unemployment_rate', 6.8), ('inactivity_rate', 20.5)]
    assert parsed.document.published_at == date(2026, 9, 15)
    assert {(item.native_geography, item.mapped_geography, item.period_start, item.period_end) for item in parsed.observations} == {('London', 'Whole London', date(2026, 5, 1), date(2026, 7, 31))}
    assert all('residents' in item.definition for item in parsed.observations)
    assert all('Residence-based' in item.validation_reason and 'Sampling volatility' in item.validation_reason for item in parsed.evidence)
    assert MONITORED_SOURCES[sid].weekly_basis is None
    assert all(item.weekly_basis is None for item in MONITORED_SOURCES[sid].measurements)


@pytest.mark.parametrize('old,new', [
    ('seasonally adjusted, May to July 2026', 'not seasonally adjusted, April to June 2026'),
    ('Employment rate (%) aged 16 to 64 years', 'Employment rate (%) aged 16 years and over'),
    ('based on the region where someone lives', 'based on the region where someone works'),
    ('Caution should be taken when drawing conclusions from short-term changes', 'Short-term changes are reliable'),
    ('London 73.9 -0.4 6.8 0.1 20.5 0.2', 'London 73.9 -0.4 6.8 0.1 20.5 0.2 London 73.9 -0.4 6.8 0.1 20.5 0.2'),
])
def test_ons_changed_table_or_missing_scope_is_rejected(old, new):
    sid = 'ons-regional-labour-september-2026'
    original = page_for(sid)
    assert old in original.text
    with pytest.raises(ValueError, match='scope|phrase'):
        parse_monitored_source(sid, original.model_copy(update={'text': original.text.replace(old, new)}))
