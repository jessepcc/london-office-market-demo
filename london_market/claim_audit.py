from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

from .domain import Claim, Evidence

NUMBER = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?!\d|\.\d)")


def figures(text: str) -> set[float]:
    return {float(match.group().replace(',', '')) for match in NUMBER.finditer(text)}


def note_figures(text: str) -> set[float]:
    text = re.sub(r"\b\d+ (?:[A-Z][\w'-]+ ){1,4}(?:Street|Road|Lane|Square|Place)\b", "", text)
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", text):
        try:
            date.fromisoformat(match.group())
        except ValueError:
            continue
        text = text.replace(match.group(), "")
    for match in re.finditer(r"\b(?:(?P<start>\d{1,2})\s*[-–—]\s*)?(?P<end>\d{1,2}) (?P<month>[A-Za-z]{3,9})(?P<year> \d{4})?\b", text):
        suffix = match['month'] + (match['year'] or "")
        for pattern in ("%d %B %Y", "%d %b %Y", "%d %B", "%d %b"):
            try:
                end = datetime.strptime(match['end'] + " " + suffix, pattern)
                start = datetime.strptime((match['start'] or match['end']) + " " + suffix, pattern)
            except ValueError:
                continue
            if start <= end:
                text = text.replace(match.group(), "")
            break
    return figures(text)


def unsupported_figures(claim: Claim, sources: list[Evidence], derived: Iterable[float] = (), reference_dates: Iterable[date] = (), reference_definitions: Iterable[str] = ()) -> list[float]:
    text = claim.text + ' ' + claim.headline
    reference_dates = list(reference_dates)
    dates = [source.retrieved_at.date() for source in sources] + reference_dates + [timestamp for source in sources for timestamp in (source.published_at, source.event_date) if timestamp]
    for timestamp in dates:
        for form in (timestamp.isoformat(), timestamp.strftime('%d %B %Y'), timestamp.strftime('%d %b %Y'), f'{timestamp.day} {timestamp:%B %Y}', f'{timestamp.day} {timestamp:%b %Y}', f'{timestamp.day} {timestamp:%B}', f'{timestamp.day} {timestamp:%b}'):
            text = text.replace(form, '')
    supported = set(derived)
    supported.update(value for definition in reference_definitions for value in figures(definition))
    supported.update(float(timestamp.year) for timestamp in reference_dates)
    for source in sources:
        supported.update(figures(source.excerpt))
        if source.published_at:
            supported.add(float(source.published_at.year))
    source_quantities = [item for source in sources for item in quantity_occurrences(source.excerpt)]
    for quantity in quantity_occurrences(text):
        if any(item.unit == quantity.unit and abs(item.value - quantity.value) <= max(1e-8, abs(item.value) * 1e-6) for item in source_quantities):
            supported.add(quantity.raw_value)
    missing = []
    for value in figures(text):
        if not any(abs(value - other) <= max(1e-8, abs(other) * 1e-6) for other in supported):
            missing.append(value)
    return sorted(missing)


# Quantities are matched as occurrences, never as a global allowed-number bag.


@dataclass(frozen=True)
class QuantityOccurrence:
    value: float
    unit: str
    operator: str
    raw_value: float
    phrase: str
    referents: frozenset[str]
    periods: frozenset[str]


_QUANTITY = re.compile(r'(?P<currency>[£$€])?\s*(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>billion|million|[mk])?\s*(?P<unit>percentage points?|per cent|percent|%|sq\.?\s*ft|square feet|sqft|sq\.?\s*m|square metres?|sqm|psf|per sq\.?\s*ft|per square foot|GBP per (?:sq\.?\s*ft|square foot))?', re.I)
_STOP = set('the a an of for in on to by at and or with from this that these those its their is are was were be has have had will would could may as it not yet than per cent percent percentage point million billion sq ft square feet metre metres space source reported reports report says stated amount allocation share total london office'.split())
_BOUND = re.compile(r'(up to|at most|no more than|less than|under|at least|no less than|more than|over|above|about|approximately|roughly|around)\s*$', re.I)
_OPERATORS = {'up to': 'upper_inclusive', 'at most': 'upper_inclusive', 'no more than': 'upper_inclusive', 'less than': 'upper_strict', 'under': 'upper_strict', 'at least': 'lower_inclusive', 'no less than': 'lower_inclusive', 'more than': 'lower_strict', 'over': 'lower_strict', 'above': 'lower_strict', 'about': 'approximate', 'approximately': 'approximate', 'roughly': 'approximate', 'around': 'approximate'}


def _words(text: str) -> frozenset[str]:
    words = []
    for word in re.findall(r'[a-z]+', text.lower()):
        if word in _STOP or len(word) < 4:
            continue
        if word.endswith('ies'): word = word[:-3] + 'y'
        elif word.endswith('s') and len(word) > 4: word = word[:-1]
        words.append(word)
    return frozenset(words)


def quantity_occurrences(text: str) -> list[QuantityOccurrence]:
    result = []
    matches = [match for match in _QUANTITY.finditer(text) if match['currency'] or match['unit']]
    for index, match in enumerate(matches):
        currency, unit = match['currency'], (match['unit'] or '').lower()
        if not currency and not unit:
            continue
        scale = (match['scale'] or '').lower()
        value = float(match['number'].replace(',', '')) * {'million': 1e6, 'm': 1e6, 'billion': 1e9, 'k': 1e3}.get(scale, 1)
        normalized = {'£': 'GBP', '$': 'USD', '€': 'EUR'}.get(currency, '')
        if currency:
            if unit in ('psf', 'per sq ft', 'per sq. ft', 'per square foot'): normalized += '/sq_ft'
        elif unit.startswith('gbp per'): normalized = 'GBP/sq_ft'
        elif unit in ('%', 'per cent', 'percent'): normalized = '%'
        elif unit.startswith('percentage point'): normalized = 'percentage_points'
        elif unit in ('sqm', 'square metre', 'square metres', 'sq m', 'sq. m'): normalized = 'sq_m'
        else: normalized = 'sq_ft'
        before = text[max(0, match.start() - 75):match.start()]
        bound = _BOUND.search(before)
        operator = _OPERATORS[bound.group().strip().lower()] if bound else 'exact'
        end = match.end()
        if text[end:end + 5].startswith('-plus') or text[end:end + 1] == '+':
            operator = 'lower_strict'
            end += 5 if text[end:end + 5] == '-plus' else 1
        left_start = max(0, match.start() - 100, matches[index - 1].end() if index else 0)
        right_end = min(len(text), end + 130, matches[index + 1].start() if index + 1 < len(matches) else len(text))
        delimiter = r'[.;!?,]|\band\b|\bwhile\b'
        left = re.split(delimiter, text[left_start:match.start()])[-1]
        right = re.split(delimiter, text[end:right_end])[0]
        context = left + ' ' + right
        periods = frozenset(re.sub(r'\s+', '', item.upper()) for item in re.findall(r'\b(?:Q[1-4]|H[12])\s*20\d{2}\b|\b20\d{2}\b', context, re.I))
        result.append(QuantityOccurrence(value, normalized, operator, float(match['number'].replace(',', '')), text[match.start() - (len(bound.group()) if bound else 0):end].strip(), _words(context), periods))
    return result


def quantity_bound_issues(claim: Claim, sources: list[Evidence]) -> list[str]:
    issues = []
    source_occurrences = [(source, occurrence) for source in sources for occurrence in quantity_occurrences(source.excerpt)]
    for field, text in (('headline', claim.headline), ('body', claim.text)):
        for occurrence in quantity_occurrences(text):
            same = [(source, item) for source, item in source_occurrences if item.unit == occurrence.unit and abs(item.value - occurrence.value) <= max(1e-8, abs(item.value) * 1e-6)]
            qualified = [(source, item) for source, item in same if item.operator != 'exact']
            if not qualified:
                continue
            faithful = [(source, item) for source, item in qualified if not item.periods or not occurrence.periods or item.periods == occurrence.periods]
            if faithful and all(item.operator == occurrence.operator for _, item in faithful):
                continue  # Preserving a bound is not an entailment certificate.
            matches = [(source, item) for source, item in same if (item.referents & occurrence.referents) - _words(source.native_geography) and (not occurrence.periods or item.periods == occurrence.periods)]
            bounded = [(source, item) for source, item in matches if item.operator != 'exact']
            if not matches:
                issues.append(f'{field} quantity {occurrence.phrase!r} is ambiguous against qualified source occurrences; cite narrower same-referent support or preserve the bound: ' + '; '.join(f'{source.id}: {item.phrase!r}' for source, item in qualified))
                continue
            if not bounded:
                continue  # An exact occurrence for this referent, rather than an unrelated equal amount.
            if any(item.operator == occurrence.operator for _, item in bounded):
                continue
            if any(item.operator == 'exact' and (source.native_geography, source.mapped_geography) == (bound_source.native_geography, bound_source.mapped_geography) and item.periods == bound_item.periods and item.referents & bound_item.referents & occurrence.referents for source, item in matches for bound_source, bound_item in bounded):
                continue
            issues.append(f'{field} quantity {occurrence.phrase!r} uses {occurrence.operator}; preserve its same-referent source bound: ' + '; '.join(f'{source.id}: {item.phrase!r} ({item.operator})' for source, item in bounded))
    return issues


def qualification_issues(claim: Claim, qualifications: list, comparisons: list, *, authored_pair: tuple[str, str] | None = None) -> list[str]:
    """Finite reviewed forecast and comparison relationships, not prose entailment."""
    from .domain import ForecastDeliveryRisk, SourceReportedChange, SUBMARKETS
    issues = []
    text = claim.headline + '. ' + claim.text
    for qualification in qualifications:
        if not isinstance(qualification, ForecastDeliveryRisk):
            continue
        expected = quantity_occurrences(f'{qualification.forecast_quantity.value} {qualification.forecast_quantity.unit}')[0]
        magnitude = any(item.unit == expected.unit and abs(item.value - expected.value) < 1e-6 for item in quantity_occurrences(text))
        count = qualification.scheme_count is not None and re.search(rf'\b{qualification.scheme_count}\s+schemes\b', text, re.I)
        if not (magnitude or count):
            continue
        share = re.escape(f'{qualification.unstarted_share.value:g}') + r'(?:\.0+)?\s*(?:%|per cent|percent)'
        denominator = share + r'\s+of\s+(?:(?:this|the|that)\s+)?(?:future|forecast)\s+supply\b'
        not_started = r'(?:has|have)\s+(?:not\s+(?:yet\s+)?(?:started|commenced|begun)|yet\s+to\s+(?:commence|start))\s+construction'
        risk = r'(?:vulnerable\s+to|(?:at\s+)?risk\s+of|subject\s+to)\s+(?:further\s+)?delays?\s+or\s+(?:potential\s+)?withdrawal\b'
        construction_referent = re.search(r'\b(?:construction|completions?|break ground|schemes commence)\b', text, re.I)
        reduced = r'(?:not\s+(?:yet\s+)?(?:started|commenced|begun)|yet\s+to\s+(?:commence|start))(?:\s+construction)?'
        risk_link = r'\s+and\s+(?:therefore\s+)?(?:(?:remains?|is|are)\s+)?' + risk
        complete = any(
            re.search(denominator + r'\s+' + not_started + risk_link, clause, re.I)
            or construction_referent and re.search(denominator + r'\s+(?:(?:has|have)\s+)?' + reduced + risk_link, clause, re.I)
            for clause in re.split(r'[.!?](?:\s+|$)', claim.text)
        )
        if not complete or re.search(r'\b(?:guaranteed|already completed|already delivered|certain completion)\b', claim.headline, re.I):
            issues.append(f'{qualification.evidence_id}: asserted forecast total/count requires its same-card future-supply denominator, unstarted construction and delay/withdrawal risk, not a schemes percentage or generic availability disclaimer. Exact source: {qualification.risk_span}')
    def asserting(field, target, reported):
        metric = r'\bsupply\b' if reported.metric == "reported_supply" else r'\b(?:space\s+(?:was\s+|is\s+)?acquired|acquired\s+(?:office\s+)?space)\b'
        subjects = r'\b(?:supply|take[- ]?up|availability|vacancy|rents?|building|floor\s+area|space\s+(?:was\s+|is\s+)?acquired|acquired\s+(?:office\s+)?space)\b'
        for sentence in re.split(r'[.!?](?:\s+|$)', field):
            for clause in re.split(r'[;,]|\s+(?:and|while|but|differs?\s+from)\s+', sentence, flags=re.I):
                if not re.search(r'\b(?:decline|fell|fall|down|change|difference)\b', clause, re.I):
                    continue
                if not any(item.unit == target.unit and abs(item.value - target.value) < 1e-6 for item in quantity_occurrences(clause)):
                    continue
                # Reduced change clauses inherit only their sentence's metric/scope.
                subject = clause if re.search(subjects, clause, re.I) else sentence
                other_area = any(re.search(r'\b' + re.escape(area) + r'\b', subject, re.I) for area in SUBMARKETS if area != reported.native_geography)
                if re.search(subjects, subject, re.I) and not re.search(metric, subject, re.I) or other_area and not re.search(r'\b' + re.escape(reported.native_geography) + r'\b', subject, re.I):
                    continue
                return True
        return False

    def asserted(target, reported):
        return asserting(claim.headline, target, reported) or asserting(claim.text, target, reported)

    routed_pairs, inspected = set(), set()
    for comparison in comparisons:
        for discrepancy in comparison.discrepancies:
            reported = discrepancy.reported
            if not any(isinstance(item, SourceReportedChange) and item == reported for item in qualifications):
                continue
            inspected.add(reported.evidence_id)
            pair = (discrepancy.from_observation_id, discrepancy.to_observation_id)
            if pair == authored_pair or pair in routed_pairs:
                continue
            published = quantity_occurrences(f'{abs(reported.change.value)} {reported.change.unit}')[0]
            value = abs(discrepancy.calculated_change.value)
            values = [value] + ([round(value, precision) for precision in range(7)] if reported.basis == "relative_percent" else [])
            targets = [published] + [quantity_occurrences(f'{value} {discrepancy.calculated_change.unit}')[0] for value in values]
            if not any(asserted(target, reported) for target in targets):
                continue
            routed_pairs.add(pair)
            replacement = {"kind": "fact", "comparison": {"kind": "publisher_endpoint_discrepancy", "from_observation_id": pair[0], "to_observation_id": pair[1]}}
            label = "supply" if reported.metric == "reported_supply" else "acquired-space"
            issues.append(f'{reported.evidence_id}: this inspected reported/calculated {label} change requires the typed replacement {json.dumps(replacement, separators=(",", ":"))} or omission of the change assertion. Execute compare for this ordered pair if its receipt lacks this relationship. The shared author preserves reported {reported.change.value:g} and calculated {discrepancy.calculated_change.value} {reported.change.unit}; its complete evidence_ids={",".join(discrepancy.evidence_ids)}. Do not rewrite attribution prose. Exact source: {reported.support_span}')
    for reported in qualifications:
        if not isinstance(reported, SourceReportedChange) or not reported.requires_comparison or reported.evidence_id in inspected:
            continue
        published = quantity_occurrences(f'{abs(reported.change.value)} {reported.change.unit}')[0]
        if asserted(published, reported):
            issues.append(f'{reported.evidence_id}: this reviewed published change requires an actual compatible compare from {reported.baseline_period_start}–{reported.baseline_period_end} to {reported.endpoint_period_start}–{reported.endpoint_period_end} for {reported.native_geography} {reported.metric}, then the returned publisher_endpoint_discrepancy fact intent, or omission of the change assertion. Its source requires reconciliation; no revised value or cause is established. Exact source: {reported.support_span}')
    return issues
