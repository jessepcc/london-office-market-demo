from authoring_support import analysis_intents, seed_analyses
"""Whole report classification is planned before spending any classifier steps."""
import json
from pathlib import Path

from london_market.domain import Claim, ChartPayload, CoverageIntent, Document, Evidence, Observation, Task, Limits, Usage, stable_id
from london_market.coverage import bind_report_coverage
from london_market.weekly import claim_source_ids, plan_signal_batches, signal_card
from london_market.providers import signal_payload, ModelReply, ToolCall, FixtureWebProvider
from london_market.runner import Runner
from london_market.config import Settings
from london_market.tools import visual_purposes
from london_market.store import Store


def snapshot():
    return json.loads((Path(__file__).parent / 'fixtures/completion/attempt8-signal-plan.json').read_text())


class View:
    def __init__(self, data):
        self.evidence = {e['id']: Evidence.model_validate(e) for e in data['evidence']}
        self.observations = [Observation.model_validate(o) for o in data['observations']]
    def get_evidence(self, eid, mode): return self.evidence.get(eid)
    def list_observations(self, mode): return self.observations


def unpack(data):
    return [Claim.model_validate(c) for c in data['claims']], {stable_id('chart', c): ChartPayload.model_validate(c) for c in data['charts']}


def test_actual_attempt8_requires_seventeen_calls_before_any_execution():
    data = snapshot()
    claims, charts = unpack(data)
    before = [c.model_dump() for c in claims]
    plan = plan_signal_batches(claims, View(data), 'live', 'cloudflare/clef-flash', charts_by_id=charts)
    assert len([c for c in claims if c.kind == 'fact']) == 29
    assert len(plan.batches) == 17 and plan.oversized == []
    assert [[i for i, _ in b] for b in plan.batches] == [[0, 1, 2], [5, 6], [8], [9, 10], [11, 12], [13, 14], [15, 17], [18, 19, 20], [21], [22], [23, 24], [25, 26], [29], [30], [31], [32, 33], [34]]
    assert [c.model_dump() for c in claims] == before
    assert all(len(signal_payload([card for _, card in batch], 'cloudflare/clef-flash')['state'][0].encode()) <= 2000 for batch in plan.batches)
    cached = {card.fingerprint for _, card in plan.batches[0]}
    assert len(plan_signal_batches(claims, View(data), 'live', 'cloudflare/clef-flash', charts_by_id=charts, cached_fingerprints=cached).batches) == 16
    repeated = plan_signal_batches([claims[0], claims[0]], View(data), 'live', 'clef')
    assert len(repeated.batches) == 1 and len(repeated.batches[0]) == 1


def test_direct_narrative_and_chart_provenance_bind_separately():
    data = snapshot()
    claims, charts = unpack(data)
    view = View(data)
    claim = claims[30]
    assert len(claim.evidence_ids) == 2
    assert len(claim_source_ids(claim, charts)) > len(claim.evidence_ids)
    coverage = bind_report_coverage(CoverageIntent.model_validate(data['coverage']), claims, charts, view.evidence, view.observations, require_full_market=True)
    entries = {e.requirement_id: e for e in coverage.entries}
    assert 30 in entries['midtown_fringe'].claim_indexes
    assert 30 not in entries['city'].claim_indexes and 30 not in entries['wider_london'].claim_indexes
    assert claim.chart_ids[0] in entries['city'].chart_ids
    assert entries['hybrid'].basis == 'qualitative'
    assert 'Measurements for Great Britain are broader context, not London office measurements.' in entries['hybrid'].gaps
    card = signal_card(claim, view, 'live', charts_by_id=charts)
    for eid in claim_source_ids(claim, charts):
        assert view.evidence[eid].native_geography in ' '.join(card.qualifications)
    assert len(claim.evidence_ids) == 2
