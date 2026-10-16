from authoring_support import analysis_intents, analysis_sources, seed_analyses, materialize_analysis
import json
from pathlib import Path


import pytest

from london_market.domain import Claim, RunResult
from london_market.landlord_analysis import SourceViewDraft
from london_market.runner import CompletionAuditError, RejectedDraft, ReviseFinishArgs, apply_completion_revision, completion_metadata_audit


def retained_analysis_author(draft):
    if isinstance(draft, SourceViewDraft):
        raise ValueError("This retention fixture has no registered commentary for the supplied ID")
    return materialize_analysis(draft)


def analysis(text='If the cited unlet pipeline is delivered, occupiers would have additional competing options.'):
    return Claim(kind='interpretation', interpretation_role='landlord_analysis', text=text, evidence_ids=['ev_source'])


def test_role_is_typed_and_legacy_interpretations_are_not_reassigned():
    legacy = Claim.model_validate({'kind': 'interpretation', 'text': 'Attributed source context.', 'evidence_ids': ['ev_source']})
    assert legacy.interpretation_role is None
    with pytest.raises(ValueError, match='only for interpretation'):
        Claim(kind='fact', interpretation_role='landlord_analysis', text='Observed.', evidence_ids=['ev_source'])
    with pytest.raises(ValueError):
        Claim(kind='interpretation', interpretation_role='methodology', text='Context.')
    raw = {'claims': [legacy.model_dump()], 'coverage': {'scope': 'focused', 'gaps': {}}}
    assert completion_metadata_audit(raw, [legacy])['audit_issues'] == []
    audit = completion_metadata_audit(raw, [legacy], require_cited_kinds=True)
    assert audit['cited_kind_counts']['landlord_analysis'] == 0
    assert audit['required_cited_kind_minima'] == {'landlord_analysis': 2, 'watchpoint': 2}
    assert any('interpretation_role' in item for item in audit['audit_issues'])


@pytest.mark.parametrize('response_type', ['answer', 'insufficient_data'])
def test_full_report_minima_depend_on_content_including_exhibits(response_type):
    empty = {'response_type': response_type, 'claims': [], 'chart_ids': [], 'coverage': {'scope': 'full_market'}}
    assert completion_metadata_audit(empty, [], require_cited_kinds=True)['required_cited_kind_minima'] == {}
    for content in ({'chart_ids': ['chart_prepared']}, {'claims': [analysis().model_dump()]}, {'claims': [{'kind': 'invalid'}]}):
        audited = completion_metadata_audit({**empty, **content}, [], require_cited_kinds=True)
        assert audited['required_cited_kind_minima'] == {'landlord_analysis': 2, 'watchpoint': 2}
        assert len(audited['audit_issues']) == 2
    assert completion_metadata_audit({**empty, 'chart_ids': ['chart_prepared']}, [], require_cited_kinds=False)['audit_issues'] == []






def test_shared_report_groups_and_labels_preserve_explicit_and_legacy_roles():
    from london_market.presentation import claim_label
    from london_market.report import export_report_html, report_groups
    claims = [analysis(), Claim(kind='interpretation', interpretation_role='source_view', text='The publisher expects supply constraints.', evidence_ids=['ev_source']), Claim(kind='interpretation', text='Legacy attribution.', evidence_ids=['ev_source'])]
    result = RunResult(status='partial', reason='Scoped fixture', answer='Scoped result', claims=claims)
    groups = list(report_groups(result))
    assert [(title, indexes) for _, title, indexes, _ in groups] == [('Source views', [1]), ('Landlord analysis', [0]), ('Interpretation', [2])]
    assert [claim_label(item) for item in claims] == ['Landlord analysis', 'Source view', 'Interpretation']
    html = export_report_html(result, [], [])
    assert 'Source views' in html and 'Landlord analysis' in html and 'Landlord interpretation' not in html
