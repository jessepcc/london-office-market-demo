from authoring_support import analysis_intents, analysis_sources, seed_analyses, materialize_analysis
"""Run-local editorial revision shares the ordinary completion contract."""

import json
from pathlib import Path

import pytest

from london_market.runner import RejectedDraft, ReviseFinishArgs, apply_completion_revision, parse_completion_args, ReportFinishArgs


def test_saved_forty_five_card_draft_uses_simultaneous_original_indexes():
    raw = json.loads((Path(__file__).parent / 'fixtures/completion/attempt4-rejected.json').read_text())
    rejected = RejectedDraft.create('run-a', raw, 750000)
    replacements = {
        7: {**raw['claims'][7], 'text': 'Watch whether the Visa One Canada Square fit-out contract is awarded later in October as reported, and whether the Paddington space at 1 Sheldon Square is subsequently released to the market.'},
        37: {**raw['claims'][37], 'text': "ONS residence-based LFS estimates for May-July 2026 put London employment at 73.9%, unemployment at 6.8% and inactivity at 20.5%. The measures have separate age and population definitions and do not measure London office jobs."},
        39: {**raw['claims'][39], 'headline': 'Observed London GVA growth in 2023', 'text': 'GLA Economics reported that the most recent ONS release showed London real GVA grew 0.3% in 2023 versus 0.4% for the UK.'},
    }
    drops = [1, 17, 27, 31, 42]
    patch = ReviseFinishArgs(draft_id=rejected.draft_id, drop_claim_indexes=drops, replacements=[{'index': i, 'claim': claim} for i, claim in replacements.items()])
    candidate = apply_completion_revision(rejected, patch)
    expected = [replacements.get(i, claim) for i, claim in enumerate(raw['claims']) if i not in drops]
    assert candidate['claims'] == expected and len(candidate['claims']) == 40
    assert {key: value for key, value in candidate.items() if key != 'claims'} == {key: value for key, value in raw.items() if key != 'claims'}
    assert rejected.raw == raw
    with pytest.raises(ValueError, match="Completion input failed typed validation") as caught:
        parse_completion_args(ReportFinishArgs, candidate)
    assert any("source-bound analysis or source_view intent" in item["message"] for item in caught.value.invalid_fields)


def watch(text='Inspect the next release.'):
    return {'kind': 'watchpoint', 'text': text, 'topic_tags': ['supply']}


@pytest.mark.parametrize('patch', [
    {'draft_id': 'stale', 'drop_claim_indexes': [0]},
    {'drop_claim_indexes': [0, 0]},
    {'replacements': [{'index': 0, 'claim': watch()}, {'index': 0, 'claim': watch()}]},
    {'drop_claim_indexes': [0], 'replacements': [{'index': 0, 'claim': watch()}]},
    {'drop_claim_indexes': [-1]}, {'drop_claim_indexes': [True]}, {'drop_claim_indexes': ['0']},
    {'drop_claim_indexes': [2]}, {}, {'replacements': [{'index': 0, 'claim': watch()}]},
    {'coverage': None, 'limitations': None},
])
def test_invalid_revision_does_not_change_candidate(patch):
    raw = {'claims': [watch(), watch('Inspect the other release.')], 'coverage': {'scope': 'full_market', 'gaps': {}}}
    rejected = RejectedDraft.create('run-a', raw, 500000)
    with pytest.raises(ValueError):
        apply_completion_revision(rejected, ReviseFinishArgs.model_validate({'draft_id': rejected.draft_id, **patch}))
    assert rejected.raw == raw


def test_revision_identity_is_run_bound_and_order_independent_and_optional_clear_is_explicit():
    raw = {'claims': [watch(), watch('Other.'), watch('Third.')], 'limitations': ['Access incomplete.']}
    rejected = RejectedDraft.create('run-a', raw, 500000)
    assert rejected.draft_id != RejectedDraft.create('run-b', raw, 500000).draft_id
    first = apply_completion_revision(rejected, ReviseFinishArgs(draft_id=rejected.draft_id, drop_claim_indexes=[0, 2], limitations=[]))
    second = apply_completion_revision(rejected, ReviseFinishArgs(draft_id=rejected.draft_id, drop_claim_indexes=[2, 0], limitations=[]))
    assert first == second == {'claims': [watch('Other.')], 'limitations': []}
    assert RejectedDraft.create('run-a', first, 500000).draft_id == RejectedDraft.create('run-a', second, 500000).draft_id
    assert RejectedDraft.create('run-a', raw, 5) is None
    assert RejectedDraft.create('run-a', {'claims': [watch()] * 81}, 500000) is None


def test_revision_chart_selection_retains_null_and_replaces_or_clears_explicitly():
    raw = {'claims': [watch()], 'chart_ids': ['chart-first', 'chart-second']}
    rejected = RejectedDraft.create('run-a', raw, 500000)
    retained = apply_completion_revision(rejected, ReviseFinishArgs(draft_id=rejected.draft_id, chart_ids=None, replacements=[{'index': 0, 'claim': watch('Other watchpoint.')}]))
    assert retained['chart_ids'] == ['chart-first', 'chart-second']
    selected = apply_completion_revision(rejected, ReviseFinishArgs(draft_id=rejected.draft_id, chart_ids=['chart-first', 'chart-second', 'chart-ready']))
    assert selected == {**raw, 'chart_ids': ['chart-first', 'chart-second', 'chart-ready']}
    cleared = apply_completion_revision(rejected, ReviseFinishArgs(draft_id=rejected.draft_id, chart_ids=[]))
    assert cleared == {**raw, 'chart_ids': []}
    assert rejected.raw == raw


@pytest.mark.parametrize('surface', ['schema', 'context'])
def test_second_revision_targets_latest_shrunken_draft_without_replaying_applied_delta(tmp_path, surface):
    from london_market.config import Settings
    from london_market.domain import Task, Usage
    from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store

    # Actual r3 applied these eleven drops to 44 cards, then replayed them
    # against the new 33-card draft (including out-of-range indexes 33+).
    drops = [3, 7, 19, 24, 29, 30, 31, 32, 33, 34, 39]
    raw = {'claims': [watch(f'Inspect condition {chr(65 + index)}.') for index in range(44)]}
    class Model:
        step = 0
        old_id = None
        new_id = None
        first_delta = None
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.step == 0:
                name, args = 'finish', raw
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                if self.step == 1:
                    assert feedback['actual_claim_count'] == 44
                    schema = next(tool['function']['parameters'] for tool in tools if tool['function']['name'] == 'revise_finish')
                    assert schema['properties']['drop_claim_indexes']['items']['maximum'] == 43
                    assert schema['$defs']['ClaimReplacement']['properties']['index']['maximum'] == 43
                    self.old_id = feedback['draft_id']
                    self.first_delta = {'draft_id': self.old_id, 'drop_claim_indexes': drops}
                    name, args = 'revise_finish', self.first_delta
                else:
                    self.new_id = feedback['draft_id']
                    assert self.new_id != self.old_id and feedback['actual_claim_count'] == 33
                    assert feedback['valid_claim_indexes'] == list(range(33))
                    assert feedback['draft_claim_index'][3]['subject'] == 'Inspect condition E.'
                    if surface == 'schema':
                        schema = next(tool['function'] for tool in tools if tool['function']['name'] == 'revise_finish')
                        assert schema['parameters']['properties']['drop_claim_indexes']['items']['maximum'] == 32
                        assert schema['parameters']['$defs']['ClaimReplacement']['properties']['index']['maximum'] == 32
                        assert 'latest draft_claim_index' in schema['description']
                        current = RejectedDraft.create('current-run', {'claims': [claim for index, claim in enumerate(raw['claims']) if index not in drops]}, 500000)
                        with pytest.raises(ValueError, match='exact latest rejected draft'):
                            apply_completion_revision(current, ReviseFinishArgs(draft_id=current.draft_id, drop_claim_indexes=drops))
                        assert len(current.raw['claims']) == 33
                    else:
                        call = next(call for message in messages for call in message.get('tool_calls', []) if call['id'] == 'revision-1')
                        assert json.loads(call['function']['arguments']) == {'rejected_draft_id': self.new_id, 'claim_count': 33, 'full_proposal_retained_in_trace': True, 'revision_applied': True}
                    name, args = 'revise_finish', {'draft_id': self.new_id, 'drop_claim_indexes': list(range(1, 33))}
            reply = ModelReply(model='synthetic-r3-index-repair', provider='synthetic', usage=Usage(), calls=[ToolCall(id=f'revision-{self.step}', name=name, arguments=json.dumps(args))])
            self.step += 1
            assert self.step <= 3
            return reply
        def classify_signals(self, *args):
            raise AssertionError('Focused watchpoint revision does not classify')
    with Store(tmp_path / 'shrunken.db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(objective='Focused latest-draft editorial repair', entry_skill='research_market'))
        assert run.status == 'partial' and [claim.text for claim in run.result.claims] == ['Inspect condition A.']
        revision = next(event for event in store.list_traces(run.id) if event.name == 'revise_finish' and event.outcome == 'failed')
        assert revision.data['arguments'] == model.first_delta
        assert len(revision.data['candidate']['claims']) == 33
        assert revision.data['draft_id'] == model.new_id


def test_empty_rejected_draft_exposes_no_revision_indexes_and_keeps_fresh_finish(tmp_path):
    from london_market.config import Settings
    from london_market.domain import Task, Usage
    from london_market.providers import ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    class Model:
        step = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.step == 0:
                args = {'claims': []}
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                assert feedback['draft_claim_index'] == [] and feedback['valid_claim_indexes'] == []
                schema = next(tool['function']['parameters'] for tool in tools if tool['function']['name'] == 'revise_finish')
                assert schema['properties']['drop_claim_indexes']['maxItems'] == schema['properties']['replacements']['maxItems'] == 0
                args = {'response_type': 'insufficient_data'}
            self.step += 1
            return ModelReply(model='synthetic-empty-index-map', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.step), name='finish', arguments=json.dumps(args))])
    with Store(tmp_path / 'empty.db') as store:
        run = Runner(Settings(), store, model=Model()).run(Task(objective='Focused empty-draft recovery', entry_skill='research_market'))
        assert run.status == 'partial' and run.result.claims == []


def reviewed_evidence(source_id, predicate):
    import hashlib
    from urllib.parse import urlsplit
    from london_market.domain import Document
    from london_market.providers import FetchedPage
    from london_market.sources import MONITORED_SOURCES, parse_monitored_source
    source = MONITORED_SOURCES[source_id]
    text = (Path(__file__).parent / 'fixtures/source_catalogue' / f'{source_id}.txt').read_text()
    doc = Document(mode='live', url=source.url, title=source.header, publisher=urlsplit(source.url).hostname, content_hash=hashlib.sha256(text.encode()).hexdigest())
    parsed = parse_monitored_source(source_id, FetchedPage(document=doc, text=text))
    return parsed.document, next(item for item in parsed.evidence if predicate(item))


def full_gap_intent():
    from london_market.domain import COVERAGE_REQUIREMENTS
    return {'scope': 'full_market', 'gaps': {item.id: ['The inspected source does not support a separate substantive finding here.'] for item in COVERAGE_REQUIREMENTS}}






def test_canonical_candidate_repetition_cannot_be_evaded_by_reordered_replacements(tmp_path):
    from london_market.config import Settings
    from london_market.domain import Task, Usage
    from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    original = {'claims': [watch('Inspect 999 first events.'), watch('Inspect 999 other events.')], 'coverage': {'scope': 'focused', 'gaps': {}}}

    class Model:
        step = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.step == 0:
                name, args = 'finish', original
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                number = 888 if self.step % 2 else 999
                replacements = [{'index': i, 'claim': watch(f'Inspect {number} {label} events.')} for i, label in enumerate(('first', 'other'))]
                if self.step % 2: replacements.reverse()
                name, args = 'revise_finish', {'draft_id': feedback['draft_id'], 'replacements': replacements}
            self.step += 1
            return ModelReply(model='synthetic-repetition', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.step), name=name, arguments=json.dumps(args))])

        def classify_signals(self, *args):
            raise AssertionError('Repeated rejected candidates cannot classify')

    with Store(tmp_path / 'db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(objective='Focused repair repetition regression', entry_skill='research_market'))
        assert run.status == 'stopped' and model.step == 5
        assert 'canonical completion candidate' in run.result.reason
        assert not run.result.claims and store.list_reports('fixture') == []
        assert not any(event.kind == 'signal' for event in store.list_traces(run.id))


def test_revision_unavailable_before_rejection_and_collection_never_offers_it(tmp_path):
    from london_market.config import Settings
    from london_market.domain import Limits, Task, Usage
    from london_market.providers import ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    class Model:
        step = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            assert 'revise_finish' not in {item['function']['name'] for item in tools}
            name, args = ('revise_finish', {'draft_id': 'unknown', 'drop_claim_indexes': [0]}) if self.step == 0 else ('finish', {'response_type': 'collection'})
            self.step += 1
            return ModelReply(model='synthetic-unavailable', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.step), name=name, arguments=json.dumps(args))])
    with Store(tmp_path / 'db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model).run(Task(objective='Collection repair must be unavailable', entry_skill='collect_market_evidence', limits=Limits(max_steps=8)))
        assert model.step == 2 and run.status == 'failed'
        failed = next(event for event in store.list_traces(run.id) if event.name == 'revise_finish')
        assert failed.outcome == 'failed' and 'only after' in failed.data['error']
        assert not run.result.claims and store.list_reports('fixture') == []


@pytest.mark.parametrize('bad_patch', ['stale', 'range', 'noop', 'duplicate', 'invalid_claim'])
def test_failed_patch_keeps_latest_candidate_and_diagnostics_then_valid_repair_passes(tmp_path, bad_patch):
    from london_market.config import Settings
    from london_market.domain import Task, Usage
    from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    raw = {'claims': [watch('Inspect 999 unsupported events.')], 'coverage': {'scope': 'focused', 'gaps': {}}}
    class Model:
        step = 0
        draft_id = None
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.step == 0:
                name, args = 'finish', raw
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                if self.step == 1:
                    self.draft_id = feedback['draft_id']
                    name, args = 'revise_finish', {'draft_id': self.draft_id}
                    if bad_patch == 'stale': args.update(draft_id='stale', drop_claim_indexes=[0])
                    if bad_patch == 'range': args['drop_claim_indexes'] = [1]
                    if bad_patch == 'duplicate': args['drop_claim_indexes'] = [0, 0]
                    if bad_patch == 'invalid_claim': args['replacements'] = [{'index': 0, 'claim': {'kind': 'unknown', 'text': 'Bad kind.'}}]
                    self.failed_args = args
                else:
                    assert feedback['draft_id'] == self.draft_id
                    assert feedback['error_claims'] == [{'index': 0, 'claim': raw['claims'][0]}]
                    assert feedback['draft_diagnostics']['audit_issues'] == ['claim 0: [999.0]']
                    call = next(call for message in messages for call in message.get('tool_calls', []) if call['id'] == '2')
                    assert json.loads(call['function']['arguments']) == self.failed_args
                    name, args = 'revise_finish', {'draft_id': self.draft_id, 'replacements': [{'index': 0, 'claim': watch()}]}
            self.step += 1
            return ModelReply(model='synthetic-stale-repair', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.step), name=name, arguments=json.dumps(args))])
        def classify_signals(self, *args):
            raise AssertionError('Watchpoints or invalid patches cannot classify')
    with Store(tmp_path / 'db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(objective='Focused repair identity regression', entry_skill='research_market'))
        assert model.step == 3 and run.status == 'partial'
        assert run.result.claims[0].text == watch()['text']
        traces = store.list_traces(run.id)
        assert [event.outcome for event in traces if event.name == 'revise_finish'] == ['failed', 'success']
        assert store.list_reports('fixture') == []


@pytest.mark.parametrize('mode,limits,expected_calls,reason', [
    ('fixture', {'max_steps': 4}, 2, 'Step limit'),
    ('live', {'max_tokens': 40000}, 1, 'conservative reservation'),
])
def test_revision_does_not_extend_explicit_action_or_token_limits(tmp_path, mode, limits, expected_calls, reason):
    from london_market.config import Settings
    from london_market.domain import Limits, Task, Usage
    from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
    from london_market.runner import Runner
    from london_market.store import Store
    class Model:
        calls = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.calls == 0:
                name, args = 'finish', {'claims': [watch('Inspect 999 unsupported events.')]}
            else:
                feedback = json.loads(next(message['content'] for message in reversed(messages) if message['role'] == 'tool'))
                name, args = 'revise_finish', {'draft_id': feedback['draft_id'], 'replacements': [{'index': 0, 'claim': watch()}]}
            self.calls += 1
            return ModelReply(model='synthetic-limited-revision', provider='synthetic', usage=Usage(input_tokens=39500 if mode == 'live' else 0, cost_known=True), calls=[ToolCall(id=str(self.calls), name=name, arguments=json.dumps(args))])
    with Store(tmp_path / 'db') as store:
        model = Model()
        run = Runner(Settings(), store, model=model, web=FixtureWebProvider()).run(Task(objective='Focused bounded repair', mode=mode, entry_skill='research_market', limits=Limits(**limits)))
        assert model.calls == expected_calls and run.status == 'stopped'
        assert reason in run.result.reason and not run.result.claims
        assert not any(event.name == 'revise_finish' and event.outcome == 'success' for event in store.list_traces(run.id))
        assert store.list_reports(mode) == []


def test_topic_gap_revision_preserves_on_null_and_clears_on_empty():
    raw={'claims':[watch(),watch('Other release.')],'topic_gaps':['Missing comparable supply history.']}
    draft=RejectedDraft.create('gap-repair',raw,500000)
    retained=apply_completion_revision(draft,ReviseFinishArgs.model_validate({'draft_id':draft.draft_id,'drop_claim_indexes':[1],'topic_gaps':None}))
    assert retained['topic_gaps']==raw['topic_gaps']
    cleared=apply_completion_revision(draft,ReviseFinishArgs.model_validate({'draft_id':draft.draft_id,'topic_gaps':[]}))
    assert cleared['topic_gaps']==[] and cleared['claims']==raw['claims']
    assert draft.raw==raw
