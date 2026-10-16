"""Bounded source-grounded authoring; saved Claims remain the display contract."""

from dataclasses import dataclass
import re
from typing import Annotated, Literal, get_args

from pydantic import BeforeValidator, Discriminator, Field, Tag, TypeAdapter

from .domain import Claim, ComparisonDiscrepancy, Evidence, ForecastDeliveryRisk, Model, Observation, TopicTag, stable_id
from .sources import Measurement, comparison_discrepancies, reviewed_prelet_commitment, reviewed_source_leaves, reviewed_source_qualifications, reviewed_nonfact_passages

AUTHORING_VERSION = "3"


class DeliveryUncertainty(Model):
    kind: Literal["delivery_uncertainty"]
    forecast_evidence_id: str


class LeaseCommitment(Model):
    kind: Literal["lease_commitment"]
    transaction_evidence_id: str


class LandlordAnalysisDraft(Model):
    kind: Literal["interpretation"] = "interpretation"
    interpretation_role: Literal["landlord_analysis"] = "landlord_analysis"
    analysis: Annotated[DeliveryUncertainty | LeaseCommitment, Field(discriminator="kind")]


class PublisherEndpointDiscrepancy(Model):
    kind: Literal["publisher_endpoint_discrepancy"]
    from_observation_id: str = Field(min_length=1)
    to_observation_id: str = Field(min_length=1)


class DiscrepancyDraft(Model):
    kind: Literal["fact"] = "fact"
    comparison: PublisherEndpointDiscrepancy


class SourceViewIntent(Model):
    evidence_id: str = Field(min_length=1)


class SourceViewDraft(Model):
    kind: Literal["interpretation"] = "interpretation"
    interpretation_role: Literal["source_view"] = "source_view"
    source_view: SourceViewIntent


def parse_draft(value):
    value = value.model_dump() if isinstance(value, Claim) else value
    if isinstance(value, dict) and value.get("kind") == "interpretation" and not ("analysis" in value or "source_view" in value):
        raise ValueError("Interpretations require a source-bound analysis or source_view intent; source_view selects one exact registered commentary evidence_id without prose overrides, or omit the optional card")
    return value


ClaimDraft = Annotated[
    Annotated[Claim, Tag("claim")] | Annotated[LandlordAnalysisDraft, Tag("analysis")] | Annotated[DiscrepancyDraft, Tag("comparison")] | Annotated[SourceViewDraft, Tag("source_view")],
    Discriminator(lambda value: "source_view" if isinstance(value, SourceViewDraft) or isinstance(value, dict) and "source_view" in value else "analysis" if isinstance(value, LandlordAnalysisDraft) or isinstance(value, dict) and "analysis" in value else "comparison" if isinstance(value, DiscrepancyDraft) or isinstance(value, dict) and "comparison" in value else "claim"),
    BeforeValidator(parse_draft),
]
CLAIM_DRAFT_ADAPTER = TypeAdapter(ClaimDraft)


@dataclass(frozen=True)
class AuthoredClaim:
    claim: Claim
    identity: str
    provenance: dict


def author_source_view(draft: SourceViewDraft, evidence: Evidence) -> AuthoredClaim:
    matches = [(source, passage, span) for source, passage, span in reviewed_nonfact_passages(evidence) if passage.kind == "interpretation" or source.source_kind == "primary_research_commentary"]
    if len(matches) != 1:
        raise ValueError("source_view requires one exact registered commentary passage; measurements, forecasts and self-labelled candidates are not publisher commentary. Inspect narrower eligible support or omit the optional card")
    source, passage, span = matches[0]
    prefix = "SYNTHETIC: " if evidence.mode == "fixture" else ""
    text = f'{prefix}{source.publisher} commentary on {passage.native_geography}: “{span}”'
    if passage.qualification:
        text += " " + passage.qualification
    tags = sorted({facet for facet in passage.facets if facet in get_args(TopicTag)} | ({passage.category} if passage.category in get_args(TopicTag) else {"demand"} if passage.category == "occupier_demand" else {"news"}))[:3]
    identity = stable_id("source_view", source.id, source.version, source.url, evidence.published_at, passage.native_geography, passage.geography, passage.category, span, passage.qualification)
    claim = Claim(kind="interpretation", interpretation_role="source_view", text=text, headline=f"{prefix}{source.publisher} commentary", evidence_ids=[evidence.id], topic_tags=tags)
    return AuthoredClaim(claim, identity, {"constructor": "source_view", "authoring_version": AUTHORING_VERSION, "support_identity": identity, "evidence_id": evidence.id, "source_contract_id": source.id, "source_contract_version": source.version})


def author_landlord_analysis(draft: LandlordAnalysisDraft, evidence: Evidence) -> AuthoredClaim:
    intent = draft.analysis
    leaves = reviewed_source_leaves(evidence)
    if len(leaves) != 1:
        raise ValueError("analysis requires one exact validated source leaf in its mode; candidate, changed or ambiguous support is insufficient")
    source, _ = leaves[0]
    prefix = "SYNTHETIC: " if evidence.mode == "fixture" else ""
    if isinstance(intent, DeliveryUncertainty):
        risks = [q for q in reviewed_source_qualifications(evidence) if isinstance(q, ForecastDeliveryRisk)]
        if len(risks) != 1:
            raise ValueError("delivery_uncertainty requires the exact reviewed forecast delivery-risk contract, not aggregate vacancy, starts or estate rents")
        q = risks[0]
        possessive = source.publisher + ("'" if source.publisher.endswith("s") else "'s")
        text = (f"{prefix}{possessive} {evidence.published_at:%B %Y} forecast puts {evidence.native_geography} completions at {q.forecast_quantity.value:g} {q.forecast_quantity.unit} by end-{q.horizon_end.year}. "
                f"It says {q.unstarted_share.value:g}% of future supply has not yet started construction and remains vulnerable to delay or withdrawal. "
                "Prospective leasing competition depends on delivery and on space being available to occupiers. The source does not establish the amount that will be available for new lettings.")
        headline = "Forecast delivery risk limits certainty about future competition"
        tags = ["supply"]
    else:
        q = reviewed_prelet_commitment(evidence)
        if q is None:
            raise ValueError("lease_commitment requires the exact reviewed completed pre-let, named occupier/building, lease term and report period")
        quarter = (q.period_start.month - 1) // 3 + 1
        text = (f"{prefix}{q.publisher} reports that {q.occupier}'s pre-let at {q.building} completed in Q{quarter} {q.period_start.year} for a {q.term_years}-year term. "
                f"That establishes an occupier commitment to this {q.native_geography} building at the transaction date. It should be distinguished from an uncommitted letting prospect. "
                "The source does not establish building completion, occupation or demand across a wider segment.")
        headline = "The completed pre-let establishes a building-specific commitment"
        tags = ["prelets", "demand"]
    identity = stable_id("analysis", intent.kind, source.id, source.version, source.url, evidence.published_at, evidence.native_geography, evidence.excerpt)
    claim = Claim(kind="interpretation", interpretation_role="landlord_analysis", text=text, headline=prefix + headline, evidence_ids=[evidence.id], topic_tags=tags)
    return AuthoredClaim(claim, identity, {"constructor": intent.kind, "authoring_version": AUTHORING_VERSION, "support_identity": identity, "evidence_id": evidence.id, "source_contract_id": source.id, "source_contract_version": source.version})


def author_discrepancy(discrepancy: ComparisonDiscrepancy, baseline: Observation, endpoint: Observation, evidence: list[Evidence]) -> AuthoredClaim:
    """Render one exact reviewed source/calculation relationship, not receipt execution."""
    closure = sorted(set(baseline.evidence_ids + endpoint.evidence_ids))
    if set(discrepancy.evidence_ids) != set(closure) or {item.id for item in evidence} != set(closure):
        raise ValueError("discrepancy requires the complete executed pair evidence closure")
    for observation in (baseline, endpoint):
        for item in evidence:
            if item.id not in observation.evidence_ids:
                continue
            leaves = reviewed_source_leaves(item)
            if item.mode != observation.mode or len(leaves) != 1:
                raise ValueError("discrepancy requires exact validated source measurements in its mode")
            source, leaf = leaves[0]
            if not isinstance(leaf, Measurement):
                raise ValueError("discrepancy requires exact reviewed measurement leaves")
            actual = (observation.metric, observation.definition, observation.unit, observation.publisher,
                      observation.native_geography, observation.mapped_geography, observation.category,
                      observation.period_start, observation.period_end, observation.value,
                      observation.dimension, observation.segment, observation.partition)
            expected = (leaf.metric, leaf.definition, leaf.unit, source.publisher,
                        leaf.native_geography or leaf.geography, leaf.geography, leaf.category,
                        *(leaf.period or source.period), float(re.fullmatch(leaf.pattern, item.excerpt, re.I)['value'].replace(',', '')),
                        leaf.dimension, leaf.segment, leaf.partition)
            if actual != expected:
                raise ValueError("discrepancy observation differs from its exact reviewed measurement")
    expected = comparison_discrepancies(baseline, endpoint, evidence, endpoint.value - baseline.value)
    if discrepancy not in expected or baseline.period_end >= endpoint.period_start:
        raise ValueError("discrepancy differs from the current reviewed ordered-pair relationship")
    reported = discrepancy.reported
    if (reported.metric, reported.basis) not in (("reported_supply", "absolute"), ("space_acquired", "relative_percent")):
        raise ValueError("discrepancy authoring supports reviewed absolute supply and relative acquired-space changes only")
    def period(item):
        return f"Q{(item.period_start.month - 1) // 3 + 1} {item.period_start.year}"
    def change(quantity):
        return f"{'fall' if quantity.value < 0 else 'rise'} of {abs(quantity.value):g} {quantity.unit}"
    first, last = period(baseline), period(endpoint)
    if reported.basis == "relative_percent":
        first, last = baseline.period_start.strftime("%B %Y"), endpoint.period_start.strftime("%B %Y")
        text = (f"{endpoint.publisher} reports {endpoint.native_geography} space acquired of {baseline.value:,.0f} {baseline.unit} in {first} and {endpoint.value:,.0f} in {last}. "
                f"The published {'fall' if reported.change.value < 0 else 'rise'} is {abs(reported.change.value):g}%; the calculated {'fall' if discrepancy.calculated_change.value < 0 else 'rise'} is {abs(discrepancy.calculated_change.value):.2f}%, rounded to two decimal places. "
                "The cause is unknown. The measure can include owner-occupier acquisitions and is not pure leased space.")
        headline, tags = "Acquired space", ["demand"]
    else:
        text = (f"{endpoint.publisher} reports {endpoint.native_geography} supply of {endpoint.value:g} {endpoint.unit} in {last} and a {change(reported.change)} from {first}. "
            f"The reported endpoint levels, {baseline.value:g} {baseline.unit} in {first} and {endpoint.value:g} {endpoint.unit} in {last}, give a calculated {change(discrepancy.calculated_change)}. "
            "The reported and calculated changes differ. Both are retained; the source does not establish the reason for the difference. "
            "Equivalence between reported supply and marketed availability has not been established.")
        headline, tags = f"{endpoint.native_geography} supply: reported and calculated changes differ", ["availability", "supply"]
    relationship = reported.model_dump(mode="json", exclude={"evidence_id"})
    identity = stable_id("comparison", relationship, baseline.id, endpoint.id, discrepancy.calculated_change.model_dump(mode="json"))
    claim = Claim(kind="fact", headline=headline, text=text, evidence_ids=closure, topic_tags=tags)
    return AuthoredClaim(claim, identity, {"constructor": "publisher_endpoint_discrepancy", "authoring_version": AUTHORING_VERSION, "support_identity": identity, "from_observation_id": baseline.id, "to_observation_id": endpoint.id, "source_contract_id": reported.source_contract_id, "source_contract_version": reported.source_contract_version, "basis": reported.basis, "calculated_change": discrepancy.calculated_change.model_dump(mode="json"), "evidence_ids": closure})
