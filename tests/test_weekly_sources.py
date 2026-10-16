import hashlib
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from london_market.domain import Document, compatibility
from london_market.providers import FetchedPage
from london_market.sources import MONITORED_SOURCES, parse_monitored_source
from london_market.store import Store

FIXTURES = Path(__file__).parent / "fixtures" / "source_catalogue"
JULY = "savills-central-london-august-2026"
AUGUST = "savills-central-london-september-2026"
FIRE = "london-fire-lambeth-hq-october-2026"
KNIGHT_FRANK = "knight-frank-london-commentary-october-2026"
FILES = {JULY: "savills-central-london-august-2026.txt", AUGUST: "savills-central-london-september-2026.txt", FIRE: "london-fire-lambeth-hq-october-2026.txt", KNIGHT_FRANK: "knight-frank-london-commentary-october-2026.txt"}


def source_page(source_id, *, text=None, url=None, published_at=None):
    source = MONITORED_SOURCES[source_id]
    text = text if text is not None else (FIXTURES / FILES[source_id]).read_text()
    return FetchedPage(document=Document(mode="live", url=url or source.url, title=source.header,
                       publisher=urlsplit(url or source.url).hostname,
                       content_hash=hashlib.sha256(text.encode()).hexdigest(), published_at=published_at), text=text)


def test_monthly_acquisitions_keep_periods_publications_and_definition_separate():
    july = parse_monitored_source(JULY, source_page(JULY))
    august = parse_monitored_source(AUGUST, source_page(AUGUST))
    assert (july.document.published_at, august.document.published_at) == (date(2026, 8, 28), date(2026, 9, 30))
    a, b = july.observations[0], august.observations[0]
    assert (a.period_start, a.period_end) == (date(2026, 7, 1), date(2026, 7, 31))
    assert (b.period_start, b.period_end) == (date(2026, 8, 1), date(2026, 8, 31))
    assert (a.value, b.value, a.unit, b.unit) == (719623, 557416, "sq ft", "sq ft")
    assert a.metric == b.metric == "space_acquired"
    assert a.native_geography == b.native_geography == "Central London"
    assert a.mapped_geography == b.mapped_geography == "Central London"
    assert compatibility(a, b) == []
    assert "owner-occupier" in a.definition
    assert MONITORED_SOURCES[JULY].weekly_basis is None
    assert MONITORED_SOURCES[AUGUST].measurements[0].weekly_basis is None
    assert "needs reconciliation" in august.evidence[0].validation_reason
    assert "reported August 35%" in august.evidence[0].validation_reason
    assert "22.54" not in august.evidence[0].validation_reason
    assert round((b.value / a.value - 1) * 100, 2) == -22.54
    assert all(item.event_date is None for parsed in (july, august) for item in parsed.evidence)


@pytest.mark.parametrize("source_id,old,new,error", [
    (JULY, "sq ft", "square metres", "measurement phrase"),
    (JULY, "leasing activity moderated in July", "leasing activity moderated in August", "scope, reference period"),
    (AUGUST, "City and West End office occupational markets", "whole London office occupational markets", "scope, reference period"),
    (AUGUST, "30 September 2026", "01 October 2026", "publication date"),
    (AUGUST, "was acquired", "is forecast to be acquired", "measurement phrase"),
    (FIRE, "is set to return", "has returned", "qualitative passage"),
    (FIRE, "receiving planning permission", "submitting a planning application", "qualitative passage"),
    (KNIGHT_FRANK, "Shabab Qadar", "Another guest", "attribution"),
    (KNIGHT_FRANK, "07 October 2026", "08 October 2026", "attribution"),
])
def test_changed_reviewed_contract_fails_before_persistence(tmp_path, source_id, old, new, error):
    page = source_page(source_id)
    changed = page.text.replace(old, new)
    assert changed != page.text
    with Store(tmp_path / "rejected.sqlite3") as store:
        with pytest.raises(ValueError, match=error):
            parsed = parse_monitored_source(source_id, source_page(source_id, text=changed))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        assert store.counts("live")["documents"] == 0


@pytest.mark.parametrize("source_id", FILES)
def test_source_identity_metadata_and_duplicate_matches_are_enforced(source_id):
    source = MONITORED_SOURCES[source_id]
    with pytest.raises(ValueError, match="identity"):
        parse_monitored_source(source_id, source_page(source_id, url=source.url + "?unreviewed=1"))
    with pytest.raises(ValueError, match="publication date"):
        parse_monitored_source(source_id, source_page(source_id, published_at=date(2026, 10, 10)))
    text = source_page(source_id).text
    with pytest.raises(ValueError, match="ambiguous"):
        parse_monitored_source(source_id, source_page(source_id, text=text + text))


def test_october_developments_do_not_create_completed_occupation_or_series():
    fire = parse_monitored_source(FIRE, source_page(FIRE))
    assert fire.document.published_at == date(2026, 10, 6)
    assert fire.observations == []
    assert all(item.event_date is None for item in fire.evidence)
    assert {(item.native_geography, item.mapped_geography) for item in fire.evidence} == {("Lambeth", "Wider London")}
    assert all(item.statement_kind == "fact" for item in fire.evidence)
    assert "future plan" in fire.evidence[0].validation_reason
    assert "Not a completed letting" in fire.evidence[0].validation_reason
    commentary = parse_monitored_source(KNIGHT_FRANK, source_page(KNIGHT_FRANK))
    assert commentary.document.published_at == date(2026, 10, 7)
    assert commentary.observations == []
    assert commentary.evidence[0].statement_kind is None
    assert "Interpretation, not a quantified" in commentary.evidence[0].validation_reason
    assert "Knight Frank" in commentary.evidence[0].validation_reason
    assert "William Matthews" in commentary.evidence[0].validation_reason


def test_new_reviewed_sources_repeat_without_duplicates_and_preserve_evidence(tmp_path):
    with Store(tmp_path / "weekly-sources.sqlite3") as store:
        for source_id in FILES:
            parsed = parse_monitored_source(source_id, source_page(source_id))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
        assert store.counts("live") == {"documents": 4, "evidence": 4, "observations": 2, "reports": 0, "runs": 0}
        for source_id in FILES:
            parsed = parse_monitored_source(source_id, source_page(source_id))
            assert store.ingest(parsed.document, parsed.evidence, parsed.observations) == {"documents": 0, "evidence": 0, "observations": 0, "revisions": 0}
        assert {item.metric for item in store.list_observations("live")} == {"space_acquired"}
        assert all(item.mapped_geography == "Central London" for item in store.list_observations("live"))
