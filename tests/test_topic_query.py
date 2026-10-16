import hashlib
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import Document, Evidence, Observation
from london_market.store import Store
from london_market.tools import QueryArgs, ToolContext, query_evidence


@pytest.fixture
def context(tmp_path):
    with Store(tmp_path / "topics.sqlite3") as store:
        yield ToolContext(Settings(), store, None, "fixture", "topic-query", 5)


def seed(ctx, label, *, category="macro", metric=None, geography="City", month=1, published=date(2026, 10, 1)):
    document = Document(mode="fixture", synthetic=True, url=f"https://topics.example.invalid/{label}", title="Synthetic topic source", publisher="Synthetic Publisher", content_hash=hashlib.sha256(label.encode()).hexdigest(), published_at=published, retrieved_at=datetime(2026, 10, 10, tzinfo=timezone.utc))
    evidence = Evidence(document_id=document.id, mode="fixture", excerpt=f"Synthetic {label} evidence.", source_url=document.url, native_geography=geography, mapped_geography=geography, category=category, published_at=published, retrieved_at=document.retrieved_at, validation_status="validated" if metric else "candidate", validation_reason="Synthetic controlled measurement" if metric else None)
    observation = Observation(mode="fixture", metric=metric, value=month, unit="%", publisher=document.publisher, definition="Synthetic definition", category=category, native_geography=geography, mapped_geography=geography, period_start=date(2026, month, 1), period_end=date(2026, month, 28), evidence_ids=[evidence.id]) if metric else None
    ctx.store.ingest(document, [evidence], [observation] if observation else [])
    return evidence, observation


def test_category_returns_normalized_and_qualitative_topic_evidence(context):
    measured, observation = seed(context, "rate", metric="bank_rate")
    qualitative, _ = seed(context, "employment-commentary")
    seed(context, "vacancy", category="vacancy", metric="vacancy_rate")
    seed(context, "news", category="news")
    result = query_evidence(context, QueryArgs(category="macro", limit=12))
    assert {item.id for item in result.evidence} == {measured.id, qualitative.id}
    assert [item.id for item in result.observations] == [observation.id]
    assert result.evidence_total == 2 and result.observation_total == 1
    assert result.supporting_evidence_ids == [measured.id]


def test_category_metric_geography_and_exact_ids_are_combined(context):
    matching, observation = seed(context, "matching", metric="bank_rate")
    wrong_metric, _ = seed(context, "wrong-metric", metric="employment_rate")
    wrong_category, _ = seed(context, "wrong-category", metric="bank_rate", category="news")
    wrong_geography, _ = seed(context, "wrong-geography", metric="bank_rate", geography="West End")
    query = QueryArgs(category="macro", metric="bank_rate", geography="City", evidence_ids=[matching.id, wrong_metric.id, wrong_category.id, wrong_geography.id, "ev_missing"], limit=1)
    result = query_evidence(context, query)
    assert [item.id for item in result.observations] == [observation.id]
    assert [item.id for item in result.evidence] == [matching.id]
    assert result.missing_evidence_ids == [wrong_metric.id, wrong_category.id, wrong_geography.id, "ev_missing"]
    assert result.evidence_total == result.observation_total == 1
    assert result.next_offset is None and not result.truncated


def test_category_pages_keep_support_outside_publication_order(context):
    recent, recent_observation = seed(context, "recent-period", metric="bank_rate", month=6, published=date(2026, 7, 1))
    older, older_observation = seed(context, "older-period", metric="bank_rate", month=1, published=date(2026, 10, 9))
    first = query_evidence(context, QueryArgs(category="macro", limit=1))
    assert [item.id for item in first.observations] == [recent_observation.id]
    assert [item.id for item in first.evidence] == [older.id, recent.id]
    assert first.supporting_evidence_ids == [recent.id]
    assert first.next_offset == 1 and first.truncated
    last = query_evidence(context, QueryArgs(category="macro", limit=1, offset=1))
    assert [item.id for item in last.observations] == [older_observation.id]
    assert {item.id for item in last.evidence} == {older.id, recent.id}
    assert last.next_offset is None and not last.truncated
    beyond = query_evidence(context, QueryArgs(category="macro", offset=2))
    assert beyond.evidence == beyond.observations == [] and not beyond.truncated


def test_exact_topic_ids_return_all_excerpts_while_observations_paginate(context):
    a, _ = seed(context, "first", metric="bank_rate", month=1)
    b, _ = seed(context, "second", metric="bank_rate", month=2)
    qualitative, _ = seed(context, "qualitative")
    requested = [a.id, b.id, qualitative.id]
    first = query_evidence(context, QueryArgs(category="macro", evidence_ids=requested, limit=1))
    last = query_evidence(context, QueryArgs(category="macro", evidence_ids=requested, limit=1, offset=1))
    assert {item.id for item in first.evidence} == set(requested)
    assert {item.id for item in last.evidence} == set(requested)
    assert len(first.observations) == len(last.observations) == 1
    assert first.observations[0].id != last.observations[0].id
    assert first.next_offset == 1 and last.next_offset is None
    assert first.missing_evidence_ids == last.missing_evidence_ids == []


def test_category_is_optional_bounded_and_matches_actual_categories(context):
    assert QueryArgs().category is None
    assert QueryArgs(category="x" * 80).category == "x" * 80
    with pytest.raises(ValidationError):
        QueryArgs(category="x" * 81)
    seed(context, "occupier", category="occupier_demand")
    assert query_evidence(context, QueryArgs(category="news")).evidence == []
    assert query_evidence(context, QueryArgs(category="occupier_demand")).evidence_total == 1


def test_category_preserves_metric_candidate_fallback_without_topic_aliases(context):
    candidate, _ = seed(context, "candidate-vacancy", category="vacancy")
    measured, observation = seed(context, "normalized-vacancy", category="vacancy", metric="vacancy_rate")
    seed(context, "availability", category="availability")
    matching = query_evidence(context, QueryArgs(category="vacancy", metric="vacancy_rate"))
    assert {item.id for item in matching.evidence} == {candidate.id, measured.id}
    assert [item.id for item in matching.observations] == [observation.id]
    mismatch = query_evidence(context, QueryArgs(category="availability", metric="vacancy_rate"))
    assert mismatch.evidence == mismatch.observations == []
    assert query_evidence(context, QueryArgs(category="vacancy_rate")).evidence == []
