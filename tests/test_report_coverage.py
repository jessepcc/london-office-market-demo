from authoring_support import analysis_intents, seed_analyses
import hashlib

import json
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.coverage import CoverageValidationError, bind_report_coverage, evidence_facets, local_exhibit_candidates, missing_local_exhibit, report_coverage_rows, retained_coverage_index
from london_market.domain import Claim, CoverageIntent, Document, Evidence, Observation, ReportCoverage, RunResult, Task, Usage
from london_market.fixtures import fixture_batches, load_fixtures, scripted_coverage
from london_market.providers import FetchedPage, FixtureWebProvider, ModelReply, ToolCall
from london_market.report import export_report_html
from london_market.runner import LiveBriefingFinishArgs, Runner
from london_market.sources import MONITORED_SOURCES, parse_monitored_source
from london_market.store import Store
from london_market.tools import MarketInventoryArgs, PrepareArgs, ToolContext, VisualizationArgs, choose_visualization, market_inventory, prepare_visualization_data


def reviewed(source_id):
    source = MONITORED_SOURCES[source_id]
    path = Path(__file__).parent / 'fixtures' / 'source_catalogue' / f'{source_id}.txt'
    if source_id in ('savills-central-london-q1-2026', 'savills-central-london-q2-2026'):
        path = Path(__file__).parent / 'fixtures' / 'live_sources' / ('q1.txt' if 'q1' in source_id else 'q2.txt')
    if source_id == 'savills-central-london-june-2026':
        path = Path(__file__).parent / 'fixtures' / 'live_sources' / 'june.txt'
    text = path.read_text()
    document = Document(mode='live', url=source.url, title=source.header, publisher=urlsplit(source.url).hostname, content_hash=hashlib.sha256(text.encode()).hexdigest())
    return parse_monitored_source(source_id, FetchedPage(document=document, text=text))


def bind(claims=(), evidence=(), observations=(), charts=None, *, scope='focused', gaps=None, inspected=()):
    return bind_report_coverage(CoverageIntent(scope=scope, gaps=gaps or {}), list(claims), charts or {}, {item.id: item for item in evidence}, list(observations), require_full_market=scope == 'full_market', inspected_evidence_ids=set(inspected))


def entry(coverage, lane):
    return next(item for item in coverage.entries if item.requirement_id == lane)


def test_model_authors_only_scope_and_gaps_while_saved_contract_stays_complete():
    with pytest.raises(ValidationError, match='coverage'):
        LiveBriefingFinishArgs(response_type='insufficient_data', claims=[])
    with pytest.raises(ValidationError, match='entries'):
        CoverageIntent(scope='full_market', entries=[])
    with pytest.raises(ValidationError, match='unknown coverage'):
        CoverageIntent(scope='full_market', gaps={'made_up': ['Unknown support']})
    with pytest.raises(ValidationError):
        CoverageIntent(scope="full_market", gaps={"employment": []})
    coverage = bind(scope='full_market')
    assert len(coverage.entries) == 20
    with pytest.raises(ValidationError, match='employment'):
        ReportCoverage(scope='full_market', entries=[item for item in coverage.entries if item.requirement_id != 'employment'])
    assert ReportCoverage.model_validate(coverage.model_dump()) == coverage


def test_exact_reviewed_leaf_tuple_support_remains_qualitative_without_invented_period():
    parsed = reviewed('savills-central-london-q2-2026')
    prime = next(item for item in parsed.evidence if '116.73' in item.excerpt)
    assert evidence_facets(prime) == {'prime_rents'}
    claim = Claim(kind='fact', text=prime.excerpt, evidence_ids=[prime.id])
    coverage = bind([claim], [prime])
    assert entry(coverage, 'prime_rents').basis == 'qualitative'
    row = report_coverage_rows(RunResult(status='partial', answer='', reason='Exact qualitative support', claims=[claim], coverage=coverage), [prime], [])[0]
    assert row['periods'] == [] and row['publication_dates'] == ['2026-08-06']
    assert evidence_facets(prime.model_copy(update={'excerpt': 'A nearby rent passage'})) == set()
    grade = reviewed('savills-central-london-june-2026').evidence
    assert any(evidence_facets(item) == {'grade_a_rents'} for item in grade)


@pytest.mark.parametrize('reverse', [False, True])
def test_binding_owns_indexes_and_filters_unrelated_uk_support_per_lane(reverse):
    rates = reviewed('boe-september-2026')
    bank = next(item for item in rates.evidence if item.category == 'macro' and 'Bank Rate' in item.excerpt)
    cw = reviewed('deutsche-bank-canary-wharf-2026').evidence[0]
    claims = [Claim(kind='fact', text=bank.excerpt, evidence_ids=[bank.id]), Claim(kind='interpretation', text='The announced occupier plan remains a scheme-level Canary Wharf example.', evidence_ids=[cw.id, bank.id])]
    if reverse:
        claims.reverse()
    coverage = bind(claims, [bank, cw], rates.observations)
    assert entry(coverage, 'rates').basis == 'context'
    area = entry(coverage, 'canary_wharf')
    assert area.basis == 'qualitative' and area.claim_indexes == ([0] if reverse else [1])
    row = next(row for row in report_coverage_rows(RunResult(status='partial', answer='', reason='Lane filter', claims=claims, coverage=coverage), [bank, cw], rates.observations) if row['id'] == 'canary_wharf')
    assert row['evidence_ids'] == [cw.id]
    assert row['native_geographies'] == ['Canary Wharf']
    assert row['source_urls'] == [str(cw.source_url)]
    assert row['periods'] == []


def test_exact_facets_keep_employment_economy_esg_and_hybrid_separate():
    gla = reviewed('gla-london-economy')
    assert [evidence_facets(item) for item in gla.evidence] == [{'employment'}, {'economy'}, {'economy'}]
    esg = next(item for item in reviewed('deutsche-bank-canary-wharf-2026').evidence if item.category == 'occupier_demand')
    assert evidence_facets(esg) == {'esg'}
    claim = Claim(kind='fact', text=esg.excerpt, evidence_ids=[esg.id], topic_tags=['hybrid-work'])
    coverage = bind([claim], [esg])
    assert not any(item.requirement_id == 'hybrid' for item in coverage.entries)
    assert not any(item.requirement_id in ('city', 'west_end') for item in coverage.entries)


def test_candidate_leaf_support_is_qualified_and_broad_category_and_rejected_sources_do_not_close_topic():
    original = fixture_batches()[0][1][0]
    candidate = Evidence.model_validate({**original.model_dump(exclude={'id'}), 'validation_status': 'candidate', 'category': 'employment'})
    claim = Claim(kind='fact', text=candidate.excerpt, evidence_ids=[candidate.id])
    coverage = bind([claim], [candidate])
    assert entry(coverage, 'employment').basis == 'qualitative'
    assert 'monitoring validation' in entry(coverage, 'employment').gaps[0]
    for replacement in (candidate.model_copy(update={'category': 'macro'}), candidate.model_copy(update={'validation_status': 'rejected'})):
        assert not any(item.requirement_id == 'employment' for item in bind([claim], [replacement]).entries)


@pytest.mark.parametrize('native,lane', [('City Fringe', 'midtown_fringe'), ('Hammersmith', 'wider_london')])
def test_native_measurements_get_scope_gaps_and_actual_periods(native, lane):
    parsed = reviewed('savills-central-london-q4-2025')
    observation = next(item for item in parsed.observations if item.native_geography == native)
    sources = [item for item in parsed.evidence if item.id in observation.evidence_ids]
    claim = Claim(kind='fact', text=sources[0].excerpt, evidence_ids=observation.evidence_ids)
    coverage = bind([claim], sources, [observation])
    assert entry(coverage, lane).basis == 'measurement' and entry(coverage, lane).gaps
    row = next(row for row in report_coverage_rows(RunResult(status='partial', answer='', reason='Native scope', claims=[claim], coverage=coverage), sources, [observation]) if row['id'] == lane)
    assert row['native_geographies'] == [native]
    assert row['periods'] == ['2025-10-01 to 2025-12-31']


def test_missing_available_lane_requires_inspection_before_an_explicit_exclusion_and_watchpoint_does_not_close_it():
    parsed = reviewed('ons-hybrid-work-2025')
    claim = Claim(kind='watchpoint', text='Watch later hybrid-work evidence.', evidence_ids=[parsed.evidence[0].id])
    with pytest.raises(CoverageValidationError, match='hybrid: retained support exists; inspect exact evidence_ids=ev_'):
        bind([claim], parsed.evidence, parsed.observations, scope='full_market')
    exclusion = {'hybrid': ['The inspected Great Britain measure cannot answer a London office-sector prevalence question; a sector-specific source remains required.']}
    with pytest.raises(CoverageValidationError, match='retained support exists'):
        bind([claim], parsed.evidence, parsed.observations, scope='full_market', gaps=exclusion)
    coverage = bind([claim], parsed.evidence, parsed.observations, scope='full_market', gaps=exclusion, inspected=[parsed.evidence[0].id])
    assert entry(coverage, 'hybrid').basis == 'gap' and entry(coverage, 'hybrid').claim_indexes == []
    assert bind([claim], parsed.evidence, parsed.observations).entries == []


def test_unpaginated_retained_pointers_expose_employment_hybrid_and_fringe():
    sources, observations = [], []
    for source_id in ('ons-regional-labour-september-2026', 'ons-hybrid-work-2025', 'savills-central-london-q4-2025'):
        parsed = reviewed(source_id)
        sources.extend(parsed.evidence); observations.extend(parsed.observations)
    rows = retained_coverage_index({item.id: item for item in sources}, observations)
    for lane in ('employment', 'hybrid', 'midtown_fringe'):
        assert next(row for row in rows if row['requirement_id'] == lane)['evidence_ids']
    assert next(row for row in rows if row['requirement_id'] == 'employment')['reference_periods'] == ['2026-05-01 to 2026-07-31']


def test_unpaginated_local_chart_is_required_then_bound_without_model_allocations(tmp_path):
    with Store(tmp_path / 'db') as store:
        parsed = reviewed('savills-central-london-q4-2025')
        store.ingest(parsed.document, parsed.evidence, parsed.observations)
        ctx = ToolContext(Settings(), store, FixtureWebProvider(), 'live', 'test', 1)
        inventory = market_inventory(ctx, MarketInventoryArgs(limit=1, offset=100))
        assert inventory.series == [] and inventory.retained_coverage_pointers
        candidate = next(item for item in inventory.local_exhibit_candidates if len(item['observation_ids']) == 5)
        claims = [Claim(kind='fact', text=item.excerpt, evidence_ids=[item.id]) for item in parsed.evidence]
        with pytest.raises(CoverageValidationError, match='candidate observation_ids=obs_'):
            bind(claims, parsed.evidence, parsed.observations, scope='full_market')
        plan = choose_visualization(ctx, VisualizationArgs(observation_ids=candidate['observation_ids'], preferred='submarkets', title='Native vacancy comparisons', purpose='Compare dated vacancy'))
        chart = prepare_visualization_data(ctx, PrepareArgs(plan_id=plan.plan_id))
        central_ids = [item.id for item in parsed.observations if item.id in candidate['observation_ids'] and item.mapped_geography in ('City', 'West End')]
        central_plan = choose_visualization(ctx, VisualizationArgs(observation_ids=central_ids, preferred='submarkets', title='Central vacancy', purpose='Compare central vacancy'))
        central = prepare_visualization_data(ctx, PrepareArgs(plan_id=central_plan.plan_id))
        required = missing_local_exhibit(parsed.observations, [central.payload])
        assert required[0]['missing_area_groups'] == ['Midtown and Fringe', 'Wider London']
        assert required[0]['observation_ids'] == candidate['observation_ids']
        assert 'Canary Wharf' not in required[0]['missing_area_groups']
        with pytest.raises(CoverageValidationError, match='missing area groups=Midtown and Fringe,Wider London'):
            bind(claims, parsed.evidence, parsed.observations, {central.chart_id: central.payload}, scope='full_market')
        assert missing_local_exhibit(parsed.observations, [chart.payload]) == []
        coverage = bind(claims, parsed.evidence, parsed.observations, {chart.chart_id: chart.payload}, scope='full_market')
        for lane in ('vacancy', 'city', 'west_end', 'midtown_fringe', 'wider_london'):
            assert entry(coverage, lane).chart_ids == [chart.chart_id]
        assert sorted(point.value for point in chart.payload.points) == [7, 7.9, 8.2, 18, 26]


def test_incompatible_local_pair_has_honest_fallback_and_conflicts_are_excluded():
    batches = fixture_batches()
    city, west = batches[1][2][0], batches[2][2][0]
    incompatible = west.model_copy(update={'definition': 'Different measurement definition'})
    assert local_exhibit_candidates([city, incompatible]) == []
    assert len(bind(scope='full_market').entries) == 20
    conflict = Observation.model_validate({**city.model_dump(exclude={'id'}), 'value': 8.1})
    assert local_exhibit_candidates([city, west, conflict]) == []


class ScopeModel:
    def __init__(self, scope, reporting=False):
        self.scope, self.reporting, self.round = scope, reporting, 0
    def complete(self, messages, tools, max_output_tokens, timeout):
        self.round += 1
        if self.reporting and self.round == 1:
            name, args = 'load_skill', {'skill_id': 'prepare_weekly_briefing'}
        else:
            name, args = 'finish', {'claims': [{'kind': 'watchpoint', 'text': 'Inspect sources before drawing a market conclusion.'}], 'coverage': {'scope': self.scope, 'gaps': {}}}
        return ModelReply(model='scripted-scope-proof', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.round), name=name, arguments=json.dumps(args))])


@pytest.mark.parametrize('scope,reporting,expected', [('focused', False, 'partial'), ('full_market', False, 'partial'), ('focused', True, 'stopped'), ('full_market', True, 'partial')])
def test_focused_chat_and_loaded_report_use_typed_scope(tmp_path, scope, reporting, expected):
    with Store(tmp_path / 'db') as store:
        run = Runner(Settings(), store, model=ScopeModel(scope, reporting)).run(Task(objective='Office research', entry_skill='research_market'))
        assert run.status == expected
        if expected == 'partial':
            assert len(run.result.coverage.entries) == (20 if scope == 'full_market' else 0)


def test_fixture_full_report_persists_and_renders_all_lanes_without_fake_weekly(tmp_path):
    with Store(tmp_path / 'db') as store:
        load_fixtures(store)
        run = Runner(Settings(), store).run(Task(objective='Prepare London office weekly report', entry_skill='prepare_weekly_briefing', briefing_date=date(2026, 10, 10)))
        assert run.status == 'partial' and run.prompt_version == '16'
        saved = store.get_run(run.id).result
        traces = store.list_traces(run.id)
        recorded_intent = next(event.data['coverage_intent'] for event in traces if event.name == 'finish' and event.outcome == 'success')
        inspected = {item['id'] for event in traces if event.name == 'query_evidence' and event.outcome == 'success' for item in event.data['output']['evidence']}
        from london_market.domain import stable_id
        rebound = bind_report_coverage(CoverageIntent.model_validate(recorded_intent), saved.claims, {stable_id('chart', chart.model_dump(mode='json')): chart for chart in saved.charts}, {item.id: item for item in store.list_evidence('fixture')}, store.list_observations('fixture'), require_full_market=True, inspected_evidence_ids=inspected)
        assert rebound == saved.coverage
        assert len(saved.coverage.entries) == 20 and saved.topic_gaps == []
        assert saved.weekly.week_over_week == [] and any(chart.chart_type == 'bar' for chart in saved.charts)
        artifact = export_report_html(saved, store.list_evidence('fixture'), store.list_observations('fixture'))
        for label in ['Topics', 'Across London', 'Employment', 'Grade A rents', 'Canary Wharf', 'Pre-lets']:
            assert label in artifact
        assert 'Original source' in artifact and 'No comparable weekly measurements are available.' in artifact
        assert 'coverage record unavailable' in export_report_html(saved.model_copy(update={'coverage': None}), [], [])


@pytest.mark.parametrize('typed_first', [False, True])
@pytest.mark.parametrize('entry_skill', ['prepare_weekly_briefing', 'research_market'])
def test_runner_uses_new_schema_and_keeps_required_chart_tools_after_typed_rejection(tmp_path, typed_first, entry_skill):
    class RepairModel:
        round = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            self.round += 1
            outputs = [json.loads(item['content']) for item in messages if item['role'] == 'tool']
            finish = next(item for item in tools if item['function']['name'] == 'finish')['function']['parameters']
            coverage_schema = finish['$defs']['CoverageIntent']
            assert set(coverage_schema['properties']) == {'scope', 'gaps'}
            if self.round == 1:
                pinned = next((line.removeprefix('Pinned weekly context: ') for item in messages for line in (item.get('content') or '').splitlines() if line.startswith('Pinned weekly context: ')), None)
                window = json.loads(pinned) if pinned else None
                name, args = 'query_evidence', {'published_since': window['window_start'], 'published_until': window['window_end']} if window else {}
            elif self.round == 2:
                name, args = 'finish', {'claims': [{'kind': 'watchpoint', 'text': 'Inspect later releases.', 'topic_tags': ['not-a-topic' if typed_first else 'availability']}] * (9 if typed_first else 1), 'coverage': {'scope': 'full_market', 'gaps': {}}}
            elif self.round == 3:
                feedback = outputs[-1]
                if typed_first:
                    assert len(feedback['invalid_fields']) == 9
                else:
                    assert any('compatible local exhibit' in issue for issue in feedback['audit_issues'])
                names = [item['function']['name'] for item in tools]
                assert names.count('choose_visualization') == names.count('prepare_visualization_data') == 1
                name, args = 'choose_visualization', {'observation_ids': ids, 'preferred': 'submarkets', 'title': 'Synthetic local vacancy', 'purpose': 'Compare dated local measurements'}
            elif self.round == 4:
                name, args = 'prepare_visualization_data', {'plan_id': outputs[-1]['plan_id']}
            elif self.round == 5:
                name, args = 'query_evidence', {'evidence_ids': availability_ids}
            else:
                name, args = 'finish', {'claims': [{'kind': 'fact', 'text': sources[0].excerpt, 'evidence_ids': [sources[0].id], 'topic_tags': ['availability']}], 'chart_ids': chart_ids, 'coverage': {'scope': 'full_market', 'gaps': {}}}
            if self.round == 5:
                chart_ids[:] = [output['chart_id'] for output in outputs if 'chart_id' in output]
            actions = [(name, args)]
            if self.round == 3:
                actions.append(('choose_visualization', feedback['visual_purposes']['trend']['candidate']))
            elif self.round == 4:
                actions = [('prepare_visualization_data', {'plan_id': output['plan_id']}) for output in outputs if 'plan_id' in output]
            return ModelReply(model='synthetic-new-contract', provider='synthetic', usage=Usage(), calls=[ToolCall(id=f'{self.round}-{i}', name=tool, arguments=json.dumps(arguments)) for i, (tool, arguments) in enumerate(actions)])
    with Store(tmp_path / 'db') as store:
        load_fixtures(store)
        ids = local_exhibit_candidates(store.list_observations('fixture'))[0]['observation_ids']
        sources = [item for item in store.list_evidence('fixture') if item.category == 'availability']
        availability_ids, chart_ids = [item.id for item in sources], []
        model = RepairModel()
        run = Runner(Settings(), store, model=model).run(Task(objective='Prepare London report', entry_skill=entry_skill))
        assert run.status == 'partial' and model.round == 6
        assert [trace.outcome for trace in store.list_traces(run.id) if trace.name == 'finish'] == ['failed', 'success']


@pytest.mark.parametrize('prose', ['Prepared exhibit chart_123 shows dated rents.', 'Pinned publication-window check returned two sources.', 'query_evidence returned the observation.'])
def test_business_prose_rejects_internal_ids_and_process_bookkeeping(tmp_path, prose):
    class ProseModel:
        round = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            self.round += 1
            if self.round == 1:
                args = {'claims': [{'kind': 'watchpoint', 'text': prose}]}
            else:
                feedback = next(json.loads(item['content']) for item in reversed(messages) if item['role'] == 'tool')
                assert any('business' in issue for issue in feedback['audit_issues'])
                args = {'response_type': 'insufficient_data'}
            return ModelReply(model='synthetic-business-prose', provider='synthetic', usage=Usage(), calls=[ToolCall(id=str(self.round), name='finish', arguments=json.dumps(args))])
    with Store(tmp_path / 'db') as store:
        model = ProseModel()
        run = Runner(Settings(), store, model=model).run(Task(objective='Office research', entry_skill='research_market'))
        assert run.status == 'partial' and model.round == 2


def test_vacancy_and_news_tags_keep_three_unique_tags():
    assert Claim(kind='watchpoint', text='Inspect sources.', topic_tags=['vacancy', 'news']).topic_tags == ['vacancy', 'news']
    for tags in (['news', 'news'], ['vacancy', 'news', 'supply', 'demand']):
        with pytest.raises(ValidationError):
            Claim(kind='watchpoint', text='Inspect sources.', topic_tags=tags)
