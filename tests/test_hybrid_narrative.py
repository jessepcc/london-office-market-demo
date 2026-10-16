import json
from pathlib import Path

import pytest
from london_market.coverage import bind_report_coverage,CoverageValidationError
from london_market.domain import Claim,ChartPayload,CoverageIntent,Evidence,Observation


def packet(): return json.loads((Path(__file__).parent/'fixtures/completion/attempt9-completion-source-packet.json').read_text())


def bind(p):
    return bind_report_coverage(CoverageIntent.model_validate(p['candidate']['coverage']),[Claim.model_validate(c) for c in p['candidate']['claims']],{cid:ChartPayload.model_validate(c) for cid,c in p['charts'].items()},{e['id']:Evidence.model_validate(e) for e in p['evidence']},[Observation.model_validate(o) for o in p['observations']],require_full_market=True,inspected_evidence_ids={e['id'] for e in p['evidence']})


def test_actual_footprint_talent_prose_cannot_close_local_hybrid_lane():
    p=packet()
    with pytest.raises(CoverageValidationError,match='no substantive attendance') as error: bind(p)
    assert 'ev_b0ed77551ac698c8584db80c' in str(error.value)
    p['candidate']['claims'][7]['headline']='Hybrid policy outlook'
    with pytest.raises(CoverageValidationError,match='no substantive attendance'): bind(p)


def test_actual_attendance_leaf_finding_and_other_area_controls_remain_valid():
    p=packet();p['candidate']['claims'][7]['text']='Semafor reports staff pushing back against employers seeking three, four or five office days a week, while Canary Wharf tenants have differing attendance policies. This is attributed qualitative context, not an area-wide measurement.'
    coverage=bind(p);rows={e.requirement_id:e for e in coverage.entries}
    assert rows['hybrid'].basis=='qualitative' and 7 in rows['hybrid'].claim_indexes
    assert any('Great Britain' in gap for gap in rows['hybrid'].gaps)
    assert rows['midtown_fringe'].basis=='measurement' and any('Midtown' in gap for gap in rows['midtown_fringe'].gaps)
    assert rows['wider_london'].basis=='measurement' and any('whole area' in gap or 'all wider London' in gap for gap in rows['wider_london'].gaps)


def test_explicit_context_gap_and_gap_only_prose_do_not_become_local_findings():
    p=packet();p['candidate']['coverage']['gaps']['hybrid']=['No London office attendance measurement; retained GB survey is broader context.']
    p['candidate']['claims'][7]['text']='No London hybrid measurement is established by these source references.'
    row=next(e for e in bind(p).entries if e.requirement_id=='hybrid')
    assert row.basis=='context' and 7 not in row.claim_indexes
    assert 27 in row.claim_indexes
