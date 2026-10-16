import json
from pathlib import Path

import pytest

from london_market.runner import CompletionValidationError, LiveBriefingFinishArgs, parse_completion_args


def test_actual_oversized_draft_reports_metadata_and_cited_kind_counts_before_repair():
    raw = json.loads((Path(__file__).parent / 'fixtures/completion/attempt5-rejected.json').read_text())
    with pytest.raises(CompletionValidationError) as caught:
        parse_completion_args(LiveBriefingFinishArgs, raw, require_cited_kinds=True)
    error = caught.value
    assert error.actual_claim_count == 41
    assert any('limitations[1]' in issue and '0.4' in issue and '0.5' in issue for issue in error.issues)
    assert any('limitations[4]' in issue and '16' in issue and '64' in issue for issue in error.issues)
    assert error.cited_kind_counts['watchpoint'] == 2
    assert error.required_cited_kind_minima == {'landlord_analysis': 2, 'watchpoint': 2}


def parsed(source_id):
    import hashlib
    from urllib.parse import urlsplit
    from london_market.domain import Document
    from london_market.providers import FetchedPage
    from london_market.sources import MONITORED_SOURCES, parse_monitored_source
    source = MONITORED_SOURCES[source_id]
    text = (Path(__file__).parent / 'fixtures/source_catalogue' / f'{source_id}.txt').read_text()
    page = FetchedPage(document=Document(mode='live', url=source.url, publisher=urlsplit(source.url).hostname, title=source.header, content_hash=hashlib.sha256(text.encode()).hexdigest()), text=text)
    return parse_monitored_source(source_id, page), page


def test_ons_exact_definitions_support_cohorts_without_whitelist_or_duplicate_observations(tmp_path, monkeypatch):
    from dataclasses import replace
    from london_market.claim_audit import unsupported_figures
    from london_market.coverage import evidence_facets, retained_coverage_index
    from london_market.domain import Claim
    from london_market.sources import MONITORED_SOURCES, parse_monitored_source
    from london_market.store import Store
    sid = 'ons-regional-labour-september-2026'
    current, page = parsed(sid)
    definitions = [item for item in current.evidence if item.category == 'labour_definitions']
    assert len(definitions) == 3 and all(item.excerpt in page.text for item in definitions)
    assert {(item.native_geography, item.mapped_geography) for item in definitions} == {('United Kingdom', 'Unmapped')}
    assert all(evidence_facets(item) == set() for item in definitions)
    row = current.evidence[0]
    claim = Claim(kind='fact', text='For May-July 2026, London employment was 73.9% and inactivity 20.5% for residents aged 16 to 64; unemployment was 6.8% for economically active residents aged 16 and over.', evidence_ids=[row.id, *[item.id for item in definitions]])
    assert unsupported_figures(claim, [row]) == [16, 64]
    assert unsupported_figures(claim, [row, *definitions]) == []
    assert unsupported_figures(claim.model_copy(update={'text': claim.text + ' Age 31 was also covered.'}), [row, *definitions]) == [31]
    with monkeypatch.context() as patch:
        patch.setitem(MONITORED_SOURCES, sid, replace(MONITORED_SOURCES[sid], qualitative_patterns=(), version='2'))
        old = parse_monitored_source(sid, page)
    with Store(tmp_path / 'db') as store:
        store.ingest(old.document, old.evidence, old.observations)
        assert store.ingest(current.document, current.evidence, current.observations)['observations'] == 0
        assert len(store.list_observations('live')) == 3
        assert store.ingest(current.document, current.evidence, current.observations) == {'documents': 0, 'evidence': 0, 'observations': 0, 'revisions': 0}
        pointer = next(item for item in retained_coverage_index({item.id: item for item in store.list_evidence('live')}, store.list_observations('live')) if item['requirement_id'] == 'employment')
        assert set(pointer['related_evidence_ids']) == {item.id for item in definitions}


def test_exact_gla_source_kinds_and_legacy_segments_do_not_promote_candidates():
    from london_market.coverage import reviewed_nonfact_segments, reviewed_repair_evidence_ids
    from london_market.domain import Evidence
    gla, _ = parsed('gla-london-economy')
    assert [item.statement_kind for item in gla.evidence] == ['interpretation', 'fact', 'forecast']
    observed, estimated = gla.evidence[1:]
    legacy = Evidence.model_validate({**observed.model_dump(exclude={'id'}), 'statement_kind': None, 'excerpt': observed.excerpt + ' ' + estimated.excerpt})
    assert reviewed_nonfact_segments(legacy) == [estimated.excerpt]
    assert reviewed_nonfact_segments(observed) == []
    assert reviewed_nonfact_segments(legacy.model_copy(update={'source_url': 'https://unknown.example/economy'})) == []
    candidate = legacy.model_copy(update={'validation_status': 'candidate'})
    assert reviewed_nonfact_segments(candidate) == [estimated.excerpt]
    assert candidate.id == legacy.id and candidate.validation_status == 'candidate' and candidate.statement_kind is None
    assert reviewed_nonfact_segments(legacy.model_copy(update={'native_geography': 'Unknown boundary'})) == []
    assert set(reviewed_repair_evidence_ids([legacy], gla.evidence)) == {item.id for item in gla.evidence}
    old_employment = gla.evidence[0].model_copy(update={'statement_kind': None})
    assert reviewed_nonfact_segments(old_employment) == [gla.evidence[0].excerpt]


def test_kind_minimum_drop_and_qualitative_metadata_policy_are_shared():
    from london_market.domain import Claim
    from london_market.runner import completion_metadata_audit
    raw = json.loads((Path(__file__).parent / 'fixtures/completion/attempt5-rejected.json').read_text())
    assert completion_metadata_audit(raw, [Claim.model_validate(item) for item in raw['claims']], require_cited_kinds=True)['cited_kind_counts']['watchpoint'] == 2
    raw['claims'].pop(40)
    metadata = completion_metadata_audit(raw, [Claim.model_validate(item) for item in raw['claims']], require_cited_kinds=True)
    assert metadata['cited_kind_counts']['watchpoint'] == 1
    assert any('two cited watchpoint' in issue for issue in metadata['audit_issues'])
    raw['coverage']['gaps']['employment'] = ['Unemployment was 6.8%.']
    assert any('employment: coverage gaps must be qualitative' in issue for issue in completion_metadata_audit(raw, [], require_cited_kinds=True)['audit_issues'])
    raw['limitations'] = [42]
    with pytest.raises(CompletionValidationError) as caught:
        parse_completion_args(LiveBriefingFinishArgs, raw, require_cited_kinds=True)
    assert any(field['field'] == ['limitations', 0] for field in caught.value.invalid_fields)


@pytest.mark.parametrize('legacy_index', [0, 1])
def test_runner_rejects_legacy_mixed_or_interpretation_facts_and_exposes_exact_repairs(tmp_path, legacy_index):
    from london_market.config import Settings
    from london_market.domain import Evidence, Task, Usage
    from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    current, _ = parsed('gla-london-economy')
    original = current.evidence[legacy_index]
    excerpt = original.excerpt + (' ' + current.evidence[2].excerpt if legacy_index == 1 else '')
    legacy = Evidence.model_validate({**original.model_dump(exclude={'id'}), 'statement_kind': None, 'excerpt': excerpt})

    class Model:
        step = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.step == 0:
                name, args = 'query_evidence', {'evidence_ids': [legacy.id, *[item.id for item in current.evidence]]}
            elif self.step == 1:
                name, args = 'finish', {'claims': [{'kind': 'fact', 'text': legacy.excerpt, 'evidence_ids': [legacy.id]}]}
            else:
                feedback = json.loads(next(item['content'] for item in reversed(messages) if item['role'] == 'tool'))
                assert any('exact registered' in issue for issue in feedback['audit_issues'])
                assert set(feedback['repair_evidence_ids']) >= {item.id for item in current.evidence}
                name, args = 'finish', {'claims': [{'kind': 'interpretation', 'interpretation_role': 'source_view', 'source_view': {'evidence_id': item.id}} if item.statement_kind == 'interpretation' else {'kind': item.statement_kind, 'text': item.excerpt, 'evidence_ids': [item.id]} for item in current.evidence]}
            self.step += 1
            return ModelReply(model='synthetic-followup', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.step), name=name, arguments=json.dumps(args))])
        def classify_signals(self, *args, **kwargs):
            raise AssertionError('Research repairs do not invoke weekly classification')

    with Store(tmp_path / 'db') as store:
        store.ingest(current.document, [*current.evidence, legacy], [])
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(mode='live', objective='Inspect London economic evidence', entry_skill='research_market'))
        assert model.step == 3 and run.status == 'partial'
        assert [item.kind for item in run.result.claims] == ['interpretation', 'fact', 'forecast']
        assert store.get_evidence(legacy.id, 'live').excerpt == excerpt
        assert store.list_reports('live') == []
        assert [event.outcome for event in store.list_traces(run.id) if event.name == 'finish'] == ['failed', 'success']


def test_filtered_completion_retains_source_conflict_chart_repair_and_window_closure():
    from copy import deepcopy
    from london_market.runner import build_completion_context, request_bound
    selected = {'id': 'ev_selected', 'excerpt': 'Exact observed source.', 'published_at': '2026-10-06', 'validation_reason': 'Native scope qualification.'}
    counter = {'id': 'ev_counter', 'excerpt': 'Conflicting source; requires qualification.'}
    definition = {'id': 'ev_definition', 'excerpt': 'Exact age and denominator definition.', 'statement_kind': 'fact'}
    unwanted = {'id': 'ev_unrelated', 'excerpt': 'Unrelated retained material. ' * 1500}
    observations = [{'id': 'obs_selected', 'evidence_ids': ['ev_selected']}, {'id': 'obs_counter', 'evidence_ids': ['ev_counter']}]
    prefix = [{'role': 'system', 'content': 'Pinned weekly context: {"window_start":"2026-10-04","window_end":"2026-10-10"}'}, {'role': 'user', 'content': 'Prepare the report.'}]
    messages = deepcopy(prefix)
    def add(name, arguments, output):
        identity = str(len(messages))
        messages.extend([{'role': 'assistant', 'content': None, 'tool_calls': [{'id': identity, 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}, {'role': 'tool', 'tool_call_id': identity, 'content': json.dumps(output)}])
    window = {'published_since': '2026-10-04', 'published_until': '2026-10-10', 'offset': 0}
    add('query_evidence', window, {'evidence': [selected, counter, unwanted], 'observations': observations, 'conflicts': [{'observation_ids': ['obs_selected', 'obs_counter']}], 'total_evidence': 99})
    add('query_evidence', {'evidence_ids': ['ev_selected', 'ev_definition']}, {'evidence': [{**selected, 'statement_kind': 'fact'}, definition], 'observations': []})
    add('compare', {}, {'evidence_ids': ['ev_selected', 'ev_counter'], 'compatible': False, 'reasons': ['Conflict']})
    add('compare', {}, {'evidence_ids': ['ev_unrelated'], 'compatible': True, 'absolute_change': 999})
    chart = {'id': 'chart_prepared', 'source_refs': ['ev_selected'], 'payload': {'points': [{'value': 1, 'evidence_ids': ['ev_selected']}]}}
    add('prepare_visualization_data', {'plan_id': 'plan_selected'}, chart)
    diagnostics = {'error': 'Unsupported age.', 'draft_id': 'draft_latest', 'audit_issues': ['claim 0: [16]'], 'error_claims': [{'index': 0, 'claim': {'text': 'Age 16'}}], 'repair_evidence_ids': ['ev_definition']}
    add('finish', {'rejected_draft_id': 'draft_latest'}, diagnostics)
    original = deepcopy(messages)
    pointers = [{'requirement_id': 'employment', 'evidence_ids': ['ev_selected'], 'related_evidence_ids': ['ev_definition']}]
    bounds = build_completion_context(messages, evidence_ids={'ev_selected', 'ev_definition'}, retained_pointers=pointers)
    assert bounds['after_byte_bound'] < bounds['before_byte_bound'] / 2
    assert messages[:2] == prefix and original != messages
    assert json.loads(original[3]['content'])['evidence'][0] == selected
    snapshot = json.loads(messages[2]['content'])
    outcomes = snapshot['observed_tool_results']
    sources = [item for outcome in outcomes for item in outcome['output'].get('evidence', [])]
    assert {item['id'] for item in sources} == {'ev_selected', 'ev_counter', 'ev_definition'}
    assert len(sources) == 3 and next(item for item in sources if item['id'] == 'ev_selected') == {**selected, 'statement_kind': 'fact'}
    assert outcomes[0]['arguments'] == window and outcomes[0]['output']['total_evidence'] == 99
    assert outcomes[0]['output']['conflicts'] == [{'observation_ids': ['obs_selected', 'obs_counter']}]
    assert snapshot['retained_coverage_pointers'] == pointers
    assert [outcome['output'] for outcome in outcomes if outcome['tool'] == 'prepare_visualization_data'] == [chart]
    assert [outcome['output']['evidence_ids'] for outcome in outcomes if outcome['tool'] == 'compare'] == [['ev_selected', 'ev_counter']]
    assert json.loads(messages[-1]['content']) == diagnostics
    encoded = json.dumps(messages)
    build_completion_context(messages, evidence_ids={'ev_selected', 'ev_definition'}, retained_pointers=pointers)
    assert json.dumps(messages) == encoded
    assert request_bound(messages) == len(json.dumps(messages, ensure_ascii=False, separators=(',', ':')).encode()) + 256


def test_numeric_audit_accepts_exact_linked_measurement_definitions_only():
    from london_market.claim_audit import unsupported_figures
    from london_market.domain import Claim, Evidence, Observation
    packet = json.loads((Path(__file__).parent / "fixtures/completion/attempt8-signal-plan.json").read_text())
    row = next(Evidence.model_validate(e) for e in packet["evidence"] if e["id"] == "ev_6aebe2c2306bb7666ba1c728")
    linked = [Observation.model_validate(o) for o in packet["observations"] if row.id in o["evidence_ids"]]
    assert len(linked) == 3
    claim = Claim(kind="fact", text="ONS regional labour data published 15 September 2026 report London resident rates for May to July 2026: employment 73.9% and inactivity 20.5% for residents aged 16 to 64, and unemployment 6.8% for economically active residents aged 16 and over. Residence-based, not office employment.", headline="London resident labour rates, May-July 2026", evidence_ids=[row.id])
    assert unsupported_figures(claim, [row]) == [16, 64]
    assert unsupported_figures(claim, [row], reference_definitions=[o.definition for o in linked]) == []
    assert unsupported_figures(claim.model_copy(update={"text": claim.text + " Age 65 was covered."}), [row], reference_definitions=[o.definition for o in linked]) == [65]
    assert unsupported_figures(claim.model_copy(update={"text": "This source establishes a 4% rate."}), [row], reference_definitions=[o.definition for o in linked]) == [4]


@pytest.mark.parametrize("linked,number,accepted", [(True, 64, True), (True, 65, False), (False, 64, False), (True, 4, False)])
def test_runner_uses_only_evidence_linked_observation_definitions(tmp_path, linked, number, accepted):
    from london_market.config import Settings
    from london_market.domain import Claim, Document, Evidence, Observation, Task, Usage
    from london_market.providers import ModelReply, ToolCall, FixtureWebProvider
    from london_market.runner import Runner
    from london_market.store import Store
    packet = json.loads((Path(__file__).parent / "fixtures/completion/attempt8-signal-plan.json").read_text())
    raw = next(e for e in packet["evidence"] if e["id"] == "ev_6aebe2c2306bb7666ba1c728")
    row = Evidence.model_validate(raw)
    other = Evidence.model_validate({**row.model_dump(exclude={"id"}), "excerpt": "A different definition record."})
    definitions = [Observation.model_validate({**{k: v for k, v in o.items() if k != "id"}, "evidence_ids": [row.id if linked else other.id]}) for o in packet["observations"] if row.id in o["evidence_ids"]]
    document = Document.model_validate(next(d for d in packet["documents"] if d["id"] == row.document_id))
    class Model:
        def complete(self, messages, tools, max_output_tokens, timeout):
            args = {"claims": [{"kind": "fact", "text": f"London residents aged 16 to {number} had employment of 73.9%.", "evidence_ids": [row.id]}]}
            return ModelReply(model="synthetic-definition-audit", provider="synthetic", usage=Usage(), calls=[ToolCall(id="finish", name="finish", arguments=json.dumps(args))])
    with Store(tmp_path / "db") as store:
        store.ingest(document, [row, other], definitions)
        run = Runner(Settings(), store, model=Model(), web=FixtureWebProvider()).run(Task(objective="Inspect the London employment definition", entry_skill="research_market", mode="live"))
        failures = [e for e in store.list_traces(run.id) if e.name == "finish" and e.outcome == "failed"]
        assert bool(run.result.claims) == accepted
        if accepted:
            assert failures == [] and run.result.claims[0].evidence_ids == [row.id]
        else:
            assert any(f"{number}.0" in issue for event in failures for issue in event.data.get("audit_issues", []))
