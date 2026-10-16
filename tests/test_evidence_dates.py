"""Calendar-date provenance must come from the exact cached source excerpt."""

import hashlib
from datetime import date

import pytest

from london_market.config import Settings
from london_market.domain import Document
from london_market.providers import FetchedPage, FixtureWebProvider
from london_market.store import Store
from london_market.tools import CacheArgs, ToolContext, cache_evidence


def cache_dated_excerpt(store, excerpt, event_date):
    document = Document(mode="live", url="https://www.savills.com/research_articles/255800/393427-0", publisher="www.savills.com", title="Calendar date validation test", published_at=date(2026, 8, 6), content_hash=hashlib.sha256(excerpt.encode()).hexdigest())
    ctx = ToolContext(Settings(), store, FixtureWebProvider(), "live", "date-test", 2)
    ctx.fetched[document.id] = FetchedPage(document=document, text=excerpt)
    return cache_evidence(ctx, CacheArgs(retrieved_id=document.id, excerpt=excerpt, native_geography="Central London", mapped_geography="Central London", category="occupier_demand", event_date=event_date))


@pytest.mark.parametrize("excerpt,event_date", [
    ("AI occupiers acquired 645,000 sq ft during H1 2026.", date(2026, 6, 30)),
    ("Named Q2 transactions include Anthropic at 1 Triton Square.", date(2026, 8, 6)),
    ("Report published on 6 August 2026, covering AI demand in H1.", date(2026, 8, 6)),
    ("The reference period ended on 30 June 2026; take-up reached 645,000 sq ft.", date(2026, 6, 30)),
    ("H1 ending 2026-06-30 recorded 645,000 sq ft of take-up.", date(2026, 6, 30)),
    ("Demand as of June 30, 2026 was elevated.", date(2026, 6, 30)),
    ("The lease closed on 16 June 2026.", date(2026, 6, 30)),
])
def test_cache_rejects_inferred_period_or_publication_event_dates(tmp_path, excerpt, event_date):
    with Store(tmp_path / "db") as store:
        with pytest.raises(ValueError, match="event_date.*null"):
            cache_dated_excerpt(store, excerpt, event_date)
        assert store.counts("live")["documents"] == 0
        assert store.list_evidence("live") == []


@pytest.mark.parametrize("calendar_text", ["2026-06-16", "16 June 2026", "June 16, 2026", "16th June 2026", "16 Jun 2026", "Jun. 16, 2026"])
def test_cache_preserves_explicit_event_dates(tmp_path, calendar_text):
    with Store(tmp_path / "db") as store:
        result = cache_dated_excerpt(store, f"The lease closed on {calendar_text} at 1 Triton Square.", date(2026, 6, 16))
        assert result.evidence.event_date == date(2026, 6, 16)
        assert store.get_evidence(result.evidence.id).event_date == date(2026, 6, 16)
        assert result.evidence.published_at == date(2026, 8, 6)


def test_cache_keeps_unknown_event_date_null_and_distinct_from_publication(tmp_path):
    with Store(tmp_path / "db") as store:
        result = cache_dated_excerpt(store, "AI occupiers acquired 645,000 sq ft during H1 2026.", None)
        assert result.evidence.event_date is None
        assert result.evidence.published_at == date(2026, 8, 6)


def test_explicit_event_on_publication_day_is_accepted_when_separately_supported(tmp_path):
    with Store(tmp_path / "db") as store:
        result = cache_dated_excerpt(store, "Report published on 6 August 2026. The lease closed on 6 August 2026.", date(2026, 8, 6))
        assert result.evidence.event_date == date(2026, 8, 6)
