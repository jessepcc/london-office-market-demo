"""Validated boundary contracts and stable evidence identities."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Annotated, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

Mode = Literal["fixture", "live"]
Signal = Literal["bullish", "bearish", "neutral"]
REPORT_CLAIM_LIMIT = 40
FOCUSED_CLAIM_LIMIT = 24
TopicTag = Literal["supply", "demand", "rents", "vacancy", "availability", "news", "regulations", "macro", "employment", "ESG", "refurbishment", "prelets", "hybrid-work"]
Geography = Literal["City", "West End", "Canary Wharf", "Midtown and Fringe", "Wider London", "Central London", "Whole London", "Unmapped"]
SUBMARKETS: tuple[Geography, ...] = ("City", "West End", "Canary Wharf", "Midtown and Fringe", "Wider London")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def stable_id(prefix: str, *parts: Any) -> str:
    body = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(body.encode()).hexdigest()[:24]}"


def validate_mapping(native: str, mapped: Geography) -> None:
    if native.strip().lower() in {"central london", "greater london", "whole london", "london"} and mapped in SUBMARKETS:
        raise ValueError("aggregate source geography cannot establish submarket coverage")


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CoverageRequirement(Model):
    id: str
    label: str
    dimension: Literal["topic", "area"] = "topic"
    geography: Geography | None = None
    metrics: tuple[str, ...] = ()


COVERAGE_REQUIREMENTS = (
    CoverageRequirement(id="prime_rents", label="Prime rents", metrics=("prime_rent", "prime_asking_rent")),
    CoverageRequirement(id="grade_a_rents", label="Grade A rents", metrics=("grade_a_rent",)),
    CoverageRequirement(id="vacancy", label="Vacancy", metrics=("vacancy_rate",)),
    CoverageRequirement(id="availability", label="Availability", metrics=("availability_rate", "available_space")),
    CoverageRequirement(id="leasing", label="Take-up and leasing activity", metrics=("take_up", "occupier_take_up_share")),
    CoverageRequirement(id="new_development", label="New development", metrics=("new_development",)),
    CoverageRequirement(id="refurbishment", label="Refurbishment", metrics=("refurbishment",)),
    CoverageRequirement(id="prelets", label="Pre-lets", metrics=("prelet_area",)),
    CoverageRequirement(id="rates", label="Interest rates", metrics=("bank_rate",)),
    CoverageRequirement(id="economy", label="Economic activity", metrics=("gva_growth",)),
    CoverageRequirement(id="employment", label="Employment", metrics=("employment_rate", "unemployment_rate", "inactivity_rate")),
    CoverageRequirement(id="flight_to_quality", label="Flight to quality", metrics=("take_up_grade_share",)),
    CoverageRequirement(id="esg", label="ESG and sustainability"),
    CoverageRequirement(id="hybrid", label="Hybrid working", metrics=("hybrid_work_share",)),
    CoverageRequirement(id="news", label="Material news and events"),
    CoverageRequirement(id="city", label="City", dimension="area", geography="City"),
    CoverageRequirement(id="west_end", label="West End", dimension="area", geography="West End"),
    CoverageRequirement(id="canary_wharf", label="Canary Wharf", dimension="area", geography="Canary Wharf"),
    CoverageRequirement(id="midtown_fringe", label="Midtown and Fringe", dimension="area", geography="Midtown and Fringe"),
    CoverageRequirement(id="wider_london", label="Wider London", dimension="area", geography="Wider London"),
)


class CoverageIntent(Model):
    scope: Literal["full_market", "focused"]
    gaps: dict[str, Annotated[list[str], Field(min_length=1, max_length=3)]] = Field(default_factory=dict, json_schema_extra={"propertyNames": {"enum": [item.id for item in COVERAGE_REQUIREMENTS]}})

    @field_validator("gaps")
    @classmethod
    def known_gap_requirements(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        unknown = set(value) - {item.id for item in COVERAGE_REQUIREMENTS}
        if unknown:
            raise ValueError("unknown coverage requirements: " + ", ".join(sorted(unknown)))
        if any(any(not note.strip() for note in notes) for notes in value.values()):
            raise ValueError("each coverage gap needs one to three concise qualitative limitations")
        return value


class CoverageEntry(Model):
    requirement_id: str = Field(json_schema_extra={"enum": [item.id for item in COVERAGE_REQUIREMENTS]})
    basis: Literal["measurement", "qualitative", "context", "gap"]
    claim_indexes: list[int] = Field(default_factory=list, max_length=REPORT_CLAIM_LIMIT)
    chart_ids: list[str] = Field(default_factory=list, max_length=12)
    gaps: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("requirement_id")
    @classmethod
    def known_requirement(cls, value: str) -> str:
        if value not in {item.id for item in COVERAGE_REQUIREMENTS}:
            raise ValueError("unknown coverage requirement")
        return value

    @field_validator("claim_indexes", "chart_ids")
    @classmethod
    def unique_references(cls, values: list) -> list:
        if len(set(values)) != len(values) or any(isinstance(value, int) and value < 0 for value in values):
            raise ValueError("coverage references must be unique and nonnegative")
        return values


class ReportCoverage(Model):
    scope: Literal["full_market", "focused"]
    entries: list[CoverageEntry] = Field(max_length=20)

    @model_validator(mode="after")
    def complete_manifest(self) -> ReportCoverage:
        keys = [item.requirement_id for item in self.entries]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate coverage requirement")
        if self.scope == "full_market":
            missing = sorted({item.id for item in COVERAGE_REQUIREMENTS} - set(keys))
            if missing:
                raise ValueError("missing required coverage lanes: " + ", ".join(missing))
        return self


class Document(Model):
    id: str = ""
    mode: Mode
    url: HttpUrl
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    raw_content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$", description="Original fetched-byte digest when content_hash identifies a reviewed semantic snapshot")
    published_at: date | None = None
    retrieved_at: datetime = Field(default_factory=utc_now)
    synthetic: bool = False

    @field_validator("retrieved_at")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("retrieval timestamp must have a timezone")
        return value

    @model_validator(mode="after")
    def identity(self) -> Document:
        expected = stable_id("doc", self.mode, str(self.url), self.content_hash)
        if self.id and self.id != expected:
            raise ValueError("document ID must identify mode, URL and content hash")
        object.__setattr__(self, "id", expected)
        if (self.mode == "fixture") != self.synthetic:
            raise ValueError("fixture documents must be labeled synthetic; live documents cannot be synthetic")
        return self


class Evidence(Model):
    id: str = ""
    document_id: str
    mode: Mode
    excerpt: str = Field(min_length=1)
    event_date: date | None = None
    published_at: date | None = None
    retrieved_at: datetime
    first_ingested_at: datetime | None = None
    source_url: HttpUrl
    native_geography: str = Field(min_length=1)
    mapped_geography: Geography
    category: str = Field(min_length=1)
    statement_kind: Literal["fact", "forecast", "interpretation"] | None = None
    validation_status: Literal["candidate", "validated", "rejected"] = "candidate"
    validation_reason: str | None = None

    @field_validator("retrieved_at", "first_ingested_at")
    @classmethod
    def timezone_required(cls, value: datetime | None) -> datetime | None:
        return Document.timezone_required(value) if value is not None else None

    @model_validator(mode="after")
    def identity(self) -> Evidence:
        validate_mapping(self.native_geography, self.mapped_geography)
        if self.validation_status == "validated" and not self.validation_reason:
            raise ValueError("validated evidence requires a monitoring validation reason")
        parts = [self.mode, self.document_id, self.excerpt, self.event_date, self.native_geography, self.mapped_geography, self.category]
        if self.statement_kind is not None:
            parts.append(self.statement_kind)
        expected = stable_id("ev", *parts)
        if self.id and self.id != expected:
            raise ValueError("evidence ID does not match content")
        object.__setattr__(self, "id", expected)
        return self


class Quantity(Model):
    value: float
    unit: str


class SourceReportedChange(Model):
    kind: Literal["source_reported_change"] = "source_reported_change"
    evidence_id: str
    source_contract_id: str
    source_contract_version: str
    support_span: str
    metric: str
    native_geography: str
    baseline_period_start: date
    baseline_period_end: date
    endpoint_period_start: date
    endpoint_period_end: date
    change: Quantity
    basis: Literal["absolute", "relative_percent"] = "absolute"
    requires_comparison: bool = False

    @model_validator(mode="after")
    def change_unit(self):
        if self.basis == "relative_percent" and self.change.unit != "%":
            raise ValueError("relative reported change requires percent, not percentage points")
        return self


class ForecastDeliveryRisk(Model):
    kind: Literal["forecast_delivery_risk"] = "forecast_delivery_risk"
    evidence_id: str
    source_contract_id: str
    source_contract_version: str
    forecast_span: str
    risk_span: str
    forecast_quantity: Quantity
    scheme_count: int | None
    horizon_end: date
    unstarted_share: Quantity
    share_basis: Literal["future_supply"] = "future_supply"
    construction_status: Literal["not_started"] = "not_started"
    risk: Literal["delay_or_withdrawal"] = "delay_or_withdrawal"


class PreletCommitment(Model):
    evidence_id: str
    source_contract_id: str
    source_contract_version: str
    publisher: str
    support_span: str
    native_geography: str
    occupier: str
    building: str
    term_years: int = Field(gt=0)
    period_start: date
    period_end: date
    status: Literal["transaction_completed"] = "transaction_completed"


SourceQualification = SourceReportedChange | ForecastDeliveryRisk


class ComparisonDiscrepancy(Model):
    kind: Literal["publisher_endpoint_discrepancy"] = "publisher_endpoint_discrepancy"
    reported: SourceReportedChange
    calculated_change: Quantity
    from_observation_id: str
    to_observation_id: str
    evidence_ids: list[str]


class Observation(Model):
    id: str = ""
    mode: Mode
    metric: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    unit: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    category: str = Field(min_length=1)
    native_geography: str = Field(min_length=1)
    mapped_geography: Geography
    period_start: date
    period_end: date
    evidence_ids: list[str] = Field(min_length=1)
    revision_of: str | None = None
    dimension: str | None = Field(default=None, min_length=1)
    segment: str | None = Field(default=None, min_length=1)
    partition: str | None = Field(default=None, min_length=1, description="Reviewed source-specific contract identifying mutually exclusive components of one denominator; does not imply exhaustive coverage")

    @property
    def semantic_key(self) -> str:
        parts = [self.mode, self.metric, self.publisher, self.definition, self.unit, self.category, self.native_geography, self.mapped_geography, self.period_start, self.period_end]
        if self.dimension is not None:
            parts += [self.dimension, self.segment, self.partition]
        return stable_id("obskey", *parts)

    @model_validator(mode="after")
    def identity(self) -> Observation:
        validate_mapping(self.native_geography, self.mapped_geography)
        if (self.dimension is None) != (self.segment is None) or (self.partition is not None and self.dimension is None):
            raise ValueError("dimension and segment must occur together; partition requires a dimension")
        if self.period_end < self.period_start:
            raise ValueError("reference period ends before it starts")
        expected = stable_id("obs", self.semantic_key, self.value)
        if self.id and self.id != expected:
            raise ValueError("observation ID does not match semantic identity and value")
        object.__setattr__(self, "id", expected)
        return self


def compatibility(a: Observation, b: Observation, *, same_period: bool = False, same_geography: bool = True) -> list[str]:
    fields = ["mode", "metric", "publisher", "definition", "unit", "category", "dimension", "segment", "partition"]
    if same_geography:
        fields += ["native_geography", "mapped_geography"]
    if same_period:
        fields += ["period_start", "period_end"]
    return [name for name in fields if getattr(a, name) != getattr(b, name)]


class Limits(Model):
    max_steps: int = Field(default=192, ge=1, le=192)
    max_seconds: float = Field(default=240, gt=0, le=3600, allow_inf_nan=False)
    max_tokens: int = Field(default=500000, ge=1)
    max_spend_usd: float = Field(default=0.50, gt=0, allow_inf_nan=False)


class Usage(Model):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    spend_usd: float = Field(default=0, ge=0, allow_inf_nan=False)
    cost_known: bool = True


class ChatMessage(Model):
    role: Literal["user", "assistant"]
    content: str


class Task(Model):
    objective: str = Field(min_length=1)
    entry_skill: str = Field(min_length=1)
    mode: Mode = "fixture"
    trigger: Literal["chat", "manual", "fixture"] = "manual"
    history: list[ChatMessage] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    limits: Limits = Field(default_factory=Limits)
    briefing_date: date | None = None
    report_week_start: date | None = None


class Claim(Model):
    kind: Literal["fact", "interpretation", "forecast", "watchpoint"]
    interpretation_role: Literal["source_view", "landlord_analysis"] | None = None
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    headline: str = Field(default="", max_length=100)
    chart_ids: list[str] = Field(default_factory=list, max_length=4, description="Optional prepared charts supporting this claim. Their provenance is resolved for audit and classification; direct evidence_ids remain model-selected narrative citations.")
    signal: Signal | None = None
    topic_tags: list[TopicTag] = Field(default_factory=list, max_length=3)

    @field_validator("topic_tags")
    @classmethod
    def unique_tags(cls, tags: list[TopicTag]) -> list[TopicTag]:
        if len(set(tags)) != len(tags):
            raise ValueError("topic tags must be unique")
        return tags

    @model_validator(mode="after")
    def fact_requires_evidence(self) -> Claim:
        if self.interpretation_role is not None and self.kind != "interpretation":
            raise ValueError("interpretation_role is allowed only for interpretation claims")
        if self.signal is not None and self.kind != "fact":
            raise ValueError("landlord signals annotate observed facts only")
        if self.kind == "forecast" and not self.evidence_ids:
            raise ValueError("source forecasts require original evidence IDs")
        if self.kind == "fact" and not (self.evidence_ids or self.chart_ids):
            raise ValueError("facts and source forecasts require evidence IDs")
        return self


ChartType = Literal["line", "multi_line", "bar", "total", "proportion", "relation", "table", "insufficient_data"]


class ObservationPair(Model):
    x_observation_id: str
    y_observation_id: str


class ChartPlan(Model):
    chart_type: ChartType
    title: str
    purpose: str
    observation_ids: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    pairs: list[ObservationPair] = Field(default_factory=list)
    denominator_observation_id: str | None = None


class ChartPoint(Model):
    label: str
    value: float = Field(allow_inf_nan=False)
    observation_id: str
    evidence_ids: list[str] = Field(min_length=1)
    period_start: date
    period_end: date
    geography: Geography
    metric: str
    unit: str
    publisher: str
    definition: str
    category: str
    dimension: str | None = None
    segment: str | None = None
    native_geography: str = ""


class ChartPair(Model):
    x: ChartPoint
    y: ChartPoint


class ChartPayload(Model):
    chart_type: ChartType
    title: str
    purpose: str
    unit: str = ""
    x_label: str = "Reference period"
    y_label: str = "Value"
    points: list[ChartPoint] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    pairs: list[ChartPair] = Field(default_factory=list)
    denominator: ChartPoint | None = None
    x_unit: str = ""
    y_unit: str = ""


class CoverageGap(Model):
    geography: Geography
    reason: str


SECTION_TITLES = {
    "executive_findings": "Executive findings",
    "key_trends": "Key trends",
    "submarket_dynamics": "Submarket dynamics",
    "concrete_examples": "Sourced concrete examples",
    "landlord_implications": "Landlord implications",
    "outlook": "Outlook and watchpoints",
    "sources_and_gaps": "Sources, freshness and gaps",
}


class ReportSection(Model):
    kind: Literal["executive_findings", "key_trends", "submarket_dynamics", "concrete_examples", "landlord_implications", "outlook", "sources_and_gaps"]
    claim_indexes: list[int] = Field(default_factory=list)
    chart_indexes: list[int] = Field(default_factory=list)

    @field_validator("claim_indexes", "chart_indexes")
    @classmethod
    def valid_indexes(cls, indexes: list[int]) -> list[int]:
        if any(index < 0 for index in indexes) or len(set(indexes)) != len(indexes):
            raise ValueError("section references must be unique nonnegative indexes")
        return indexes


def validate_sections(sections: list[ReportSection], claims: list[Claim], charts: list[Any]) -> None:
    if len({section.kind for section in sections}) != len(sections):
        raise ValueError("report section kinds must be unique")
    for section in sections:
        if any(index >= len(claims) for index in section.claim_indexes) or any(index >= len(charts) for index in section.chart_indexes):
            raise ValueError("report section contains a dangling claim or chart reference")


class WeeklyContext(Model):
    briefing_date: date
    report_week_start: date
    window_start: date
    window_end: date
    cutoff_at: datetime
    timezone: str
    prior_report_id: str | None = None
    prior_run_id: str | None = None
    prior_briefing_date: date | None = None
    prior_evidence_ids: list[str] = Field(default_factory=list)


class WeeklyReferencePeriod(Model):
    observation_id: str
    period_start: date
    period_end: date
    revision_of: str | None = None


class WeeklyEvidenceUpdate(Model):
    evidence_id: str
    document_id: str
    published_at: date | None
    event_date: date | None
    retrieved_at: datetime
    first_ingested_at: datetime | None
    previously_cited: bool | None
    publication_in_window: bool | None
    event_in_window: bool | None
    first_ingested_in_window: bool | None
    reference_periods: list[WeeklyReferencePeriod] = Field(default_factory=list)


class WeeklySourceCount(Model):
    publisher: str
    domain: str
    publication_count: int = Field(ge=0)
    excerpt_count: int = Field(ge=0)


class WeeklySourceAttempt(Model):
    run_id: str
    url: HttpUrl
    status: Literal["success", "failed", "unchanged", "blocked"]
    at: datetime
    document_id: str | None = None
    error: str | None = None


class WeeklySearch(Model):
    run_id: str
    query: str
    at: datetime
    outcome: str
    start_date: date | None = None
    end_date: date | None = None


class WeeklyIndicator(Model):
    from_observation_id: str
    to_observation_id: str
    baseline_period_start: date
    baseline_period_end: date
    endpoint_period_start: date
    endpoint_period_end: date
    absolute_change: float
    unit: str
    relative_growth_percent: float | None = None
    publisher: str
    definition: str
    native_geography: str
    mapped_geography: Geography
    metric: str
    weekly_basis: Literal["period_total", "snapshot"]
    evidence_ids: list[str]
    comparison_trace_index: int


class WeeklySummary(Model):
    context: WeeklyContext
    evidence_updates: list[WeeklyEvidenceUpdate] = Field(default_factory=list)
    source_concentration: list[WeeklySourceCount] = Field(default_factory=list)
    cited_publication_count: int = Field(default=0, ge=0)
    cited_excerpt_count: int = Field(default=0, ge=0)
    cited_domain_count: int = Field(default=0, ge=0)
    source_attempts: list[WeeklySourceAttempt] = Field(default_factory=list)
    fresh_searches: list[WeeklySearch] = Field(default_factory=list)
    week_over_week: list[WeeklyIndicator] = Field(default_factory=list)
    indicator_gaps: list[str] = Field(default_factory=list)
    unassessed_fact_count: int = Field(default=0, ge=0)


class RunResult(Model):
    coverage: ReportCoverage | None = None
    briefing_date: date | None = None
    status: Literal["completed", "partial", "failed", "stopped"]
    answer: str
    claims: list[Claim] = Field(default_factory=list)
    charts: list[ChartPayload] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    coverage_gaps: list[CoverageGap] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    topic_gaps: list[str] = Field(default_factory=list)
    collection_counts: dict[str, int] = Field(default_factory=dict)
    reason: str = Field(min_length=1)
    usage: Usage = Field(default_factory=Usage)
    weekly: WeeklySummary | None = None

    @model_validator(mode="after")
    def section_references(self) -> RunResult:
        validate_sections(self.sections, self.claims, self.charts)
        return self


class TraceEvent(Model):
    at: datetime = Field(default_factory=utc_now)
    kind: str
    name: str
    outcome: str
    data: dict[str, Any] = Field(default_factory=dict)


class SourceAttempt(Model):
    url: HttpUrl
    status: Literal["success", "failed", "unchanged", "blocked"]
    at: datetime = Field(default_factory=utc_now)
    document_id: str | None = None
    error: str | None = None


class RunRecord(Model):
    id: str = Field(default_factory=lambda: uuid4().hex)
    task: Task
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: Literal["running", "completed", "partial", "failed", "stopped"] = "running"
    main_model: str
    router_model: str
    config_version: str = "1"
    prompt_version: str = "1"
    result: RunResult | None = None
    usage: Usage = Field(default_factory=Usage)
