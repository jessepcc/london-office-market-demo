import json
from pathlib import Path

import pytest
from london_market.claim_audit import qualification_issues
from london_market.domain import Claim,Evidence,Observation,SourceReportedChange,ForecastDeliveryRisk
from london_market.sources import reviewed_source_qualifications,comparison_discrepancies
from london_market.tools import ComparisonResult


def packet():return json.loads((Path(__file__).parent/'fixtures/completion/attempt9-completion-source-packet.json').read_text())


def sources():return [Evidence.model_validate(e) for e in packet()['evidence']]


def qualifications():return [q for e in sources() for q in reviewed_source_qualifications(e)]


def comparisons():
    p=packet();obs={o['id']:Observation.model_validate(o) for o in p['observations']}
    results=[]
    for event in p['comparisons']:
        result=ComparisonResult.model_validate(event['output'])
        result=result.model_copy(update={'discrepancies':comparison_discrepancies(obs[result.from_observation_id],obs[result.to_observation_id],sources(),result.absolute_change)})
        results.append(result)
    return results


def assert_typed_repair(issues, receipt=None):
    assert len(issues) == 1
    replacement, _ = json.JSONDecoder().raw_decode(issues[0].split('typed replacement ', 1)[1])
    receipt = receipt or next(c for c in comparisons() if c.discrepancies)
    assert replacement == {'kind': 'fact', 'comparison': {'kind': 'publisher_endpoint_discrepancy', 'from_observation_id': receipt.from_observation_id, 'to_observation_id': receipt.to_observation_id}}
    assert all(eid in issues[0] for eid in receipt.evidence_ids)


def test_exact_source_leaf_resolves_actual_reported_delta_and_forecast_denominator():
    qs=qualifications();assert len(qs)==2
    reported=next(q for q in qs if isinstance(q,SourceReportedChange));risk=next(q for q in qs if isinstance(q,ForecastDeliveryRisk))
    assert reported.change.value==-.5 and reported.change.unit=='million sq ft'
    assert risk.unstarted_share.value==28 and risk.share_basis=='future_supply'
    assert risk.forecast_quantity.value==22 and risk.scheme_count==148
    assert 'withdrawal' in risk.risk_span and risk.horizon_end.isoformat()=='2029-12-31'
    discrepancy=next(d for c in comparisons() for d in c.discrepancies)
    assert discrepancy.calculated_change.value==-.4 and discrepancy.reported.change.value==-.5


@pytest.mark.parametrize('change', [{'validation_status':'candidate'},{'source_url':'https://example.org/unreviewed'},{'category':'pipeline-other'},{'native_geography':'City'},{'published_at':None},{'excerpt':'Other excerpt with 22 million sq ft and 28%.'}])
def test_unknown_sibling_candidate_or_changed_leaf_cannot_get_reviewed_authority(change):
    e=next(e for e in sources() if reviewed_source_qualifications(e))
    assert reviewed_source_qualifications(e.model_copy(update=change))==[]


@pytest.mark.parametrize('kind',['fact','forecast','interpretation','watchpoint'])
def test_forecast_total_requires_exact_supply_risk_across_all_kinds(kind):
    q=next(q for q in qualifications() if isinstance(q,ForecastDeliveryRisk))
    claim=Claim(kind=kind,text='Savills forecasts 22 million sq ft by end-2029 across 148 schemes. Projected completions are not current availability.',evidence_ids=[q.evidence_id])
    assert qualification_issues(claim,[q],[])
    valid=claim.model_copy(update={'text':'Savills forecasts 22 million sq ft by end-2029 across 148 schemes. It says 28% of this future supply has not yet started construction and remains vulnerable to delays or withdrawal.'})
    assert qualification_issues(valid,[q],[])==[]
    assert qualification_issues(valid.model_copy(update={'text':valid.text.replace('future supply','schemes')}),[q],[])
    generic=claim.model_copy(update={'text':'Monitor whether the forecast changes or delivery is delayed.'})
    assert qualification_issues(generic,[q],[])==[]


def test_known_reported_calculated_discrepancy_and_level_only_positive():
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange));cs=comparisons()
    bad=Claim(kind='fact',headline='Reported supply down 0.4m sq ft',text='Supply declined 0.4 million sq ft from 19.1 million to 18.7 million.',evidence_ids=[q.evidence_id])
    assert qualification_issues(bad,[q],cs)
    good=bad.model_copy(update={'headline':'Calculated supply decline differs from source narrative','text':"Rounded Savills endpoint levels imply a calculated 0.4 million sq ft decline, from 19.1 million to 18.7 million. Savills' narrative reports a 0.5 million sq ft decline; the rounded-level calculation differs from that reported change."})
    assert_typed_repair(qualification_issues(good,[q],cs))
    assert qualification_issues(good.model_copy(update={'headline':bad.headline}),[q],cs)
    assert qualification_issues(bad.model_copy(update={'headline':'Central London supply level','text':'Savills reports 18.7 million sq ft of Central London supply in Q2 2026.'}),[q],cs)==[]
    assert qualification_issues(bad,[q],[])==[]  # No unexecuted arithmetic authority is invented.


def test_each_change_amount_requires_typed_authoring_even_with_faithful_prose():
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange));cs=comparisons()
    c=Claim(kind='fact',headline='Supply change comparison',text='Savills reports a 0.4 million sq ft decline; the calculated endpoint decline is 0.5 million sq ft, and the figures differ.',evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],cs)
    assert_typed_repair(qualification_issues(c.model_copy(update={'text':'The calculated endpoint decline is 0.4 million sq ft; Savills reports a 0.5 million sq ft decline, and the figures differ.'}),[q],cs))


def test_actual_pair_identity_period_unit_and_publisher_are_required():
    p=packet();obs={o['id']:Observation.model_validate(o) for o in p['observations']}
    receipt=next(c for c in comparisons() if c.discrepancies)
    a,b=obs[receipt.from_observation_id],obs[receipt.to_observation_id]
    for changed in ({'publisher':'Other'}, {'native_geography':'City'}, {'unit':'sq ft'}, {'period_start':__import__('datetime').date(2026,4,2)}):
        assert comparison_discrepancies(a,b.model_copy(update=changed),sources(),-.4)==[]
    assert comparison_discrepancies(a,b,sources(),-.5)==[]
    assert comparison_discrepancies(a,b.model_copy(update={'value':18.6}),sources(),-.5)==[]


def test_equivalent_bounded_forecast_risk_and_wrong_scattered_qualification():
    q=next(q for q in qualifications() if isinstance(q,ForecastDeliveryRisk))
    c=Claim(kind='forecast',text='Savills forecasts 22 million sq ft. It says 28% of future supply has not commenced construction and remains at risk of delay or withdrawal.',evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],[])==[]
    assert qualification_issues(c.model_copy(update={'text':'Savills forecasts 22 million sq ft. Another survey reported 28%. Supply faces risk. Construction has not commenced.'}),[q],[])


def test_query_and_executed_compare_expose_exact_qualification_before_drafting(tmp_path):
    from tests.test_live_sources import page_for
    from london_market.config import Settings
    from london_market.providers import FixtureWebProvider, FetchedPage
    from london_market.sources import parse_monitored_source
    from london_market.store import Store
    from london_market.tools import ToolContext, QueryArgs, CompareArgs, query_evidence, compare
    from london_market.weekly import signal_card

    with Store(tmp_path/'db') as store:
        for source_id in ('savills-central-london-q1-2026','savills-central-london-q2-2026'):
            fetched=page_for(source_id)
            parsed=parse_monitored_source(source_id,fetched)
            store.ingest(parsed.document,parsed.evidence,parsed.observations)
        context=ToolContext(Settings(),store,FixtureWebProvider(),'live','test',10)
        supply=sorted((o for o in store.list_observations('live') if o.metric=='reported_supply'),key=lambda o:o.period_start)
        result=compare(context,CompareArgs(observation_ids=[o.id for o in supply]))
        assert result.absolute_change==-.4
        assert result.discrepancies[0].reported.change.value==-.5
        result=ComparisonResult.model_validate_json(result.model_dump_json())
        endpoint=store.get_evidence(supply[-1].evidence_ids[0],'live')
        queried=query_evidence(context,QueryArgs(evidence_ids=[endpoint.id]))
        assert queried.source_qualifications[endpoint.id]==[result.discrepancies[0].reported]
        risk=next(e for e in store.list_evidence('live') if any(isinstance(q,ForecastDeliveryRisk) for q in reviewed_source_qualifications(e)))
        assert query_evidence(context,QueryArgs(evidence_ids=[risk.id])).source_qualifications[risk.id][0].risk_span in risk.excerpt
        card=signal_card(Claim(kind='fact',text='Savills reports 18.7 million sq ft of supply.',evidence_ids=[endpoint.id]),store,'live')
        span=result.discrepancies[0].reported.support_span
        assert card.qualifications.count(span)==1


def test_focused_runner_rejects_all_material_issues_then_applies_typed_repair(tmp_path):
    import hashlib
    from urllib.parse import urlsplit
    from london_market.config import Settings
    from london_market.domain import Document,Task,Usage
    from london_market.providers import FixtureWebProvider,ModelReply,ToolCall
    from london_market.runner import Runner
    from london_market.store import Store

    p=packet()
    originals=[Claim.model_validate(p['candidate']['claims'][i]) for i in (17,33)]
    chosen=[e for e in sources() if e.id in {eid for c in originals for eid in c.evidence_ids}]
    mapped={}
    with Store(tmp_path/'db') as store:
        for old in chosen:
            document=Document(mode='live',url=old.source_url,title='Literal completion source',publisher=urlsplit(str(old.source_url)).hostname,published_at=old.published_at,content_hash=hashlib.sha256(old.excerpt.encode()).hexdigest())
            current=Evidence.model_validate({**old.model_dump(),'id':'','document_id':document.id})
            mapped[old.id]=current.id
            store.ingest(document,[current],[])
        raw={'claims':[c.model_copy(update={'evidence_ids':[mapped[e] for e in c.evidence_ids]}).model_dump(mode='json') for c in originals], 'coverage':{'scope':'focused','gaps':{}}}
        class Model:
            step=0
            def complete(self,messages,tools,max_output_tokens,timeout):
                if self.step==0:
                    actions=[('query_evidence',{'evidence_ids':list(mapped.values())})]
                elif self.step==1:
                    actions=[('finish',raw)]
                else:
                    feedback=json.loads(next(m['content'] for m in reversed(messages) if m['role']=='tool'))
                    assert any('claim 0:' in issue and 'future-supply' in issue for issue in feedback['audit_issues'])
                    assert any('claim 1:' in issue and 'upper_inclusive' in issue for issue in feedback['audit_issues'])
                    replacements=[{'index':0,'claim':{**raw['claims'][0],'text':'Savills forecasts 22 million sq ft by end-2029 across 148 schemes. It says 28% of this future supply has not yet started construction and remains vulnerable to delays or withdrawal.'}}, {'index':1,'claim':{**raw['claims'][1],'text':originals[1].text.replace('the 60% laboratory allocation','the original allocation of up to 60% for laboratories')}}]
                    actions=[('revise_finish',{'draft_id':feedback['draft_id'],'replacements':replacements})]
                self.step+=1
                assert self.step<=3
                return ModelReply(model='synthetic-source-repair',provider='synthetic',usage=Usage(),calls=[ToolCall(id=f'{self.step}-{i}',name=name,arguments=json.dumps(args)) for i,(name,args) in enumerate(actions)])
            def classify_signals(self,*args,**kwargs):
                raise AssertionError('Focused forecast/watchpoint repair needs no fact classification')
        model=Model()
        run=Runner(Settings(),store,model=model,web=FixtureWebProvider()).run(Task(objective='Focused forecast and refurbishment check',entry_skill='research_market',mode='live'))
        assert model.step==3 and len(run.result.claims)==2
        traces=store.list_traces(run.id)
        assert any(t.name=='finish' and t.outcome=='failed' for t in traces)
        assert any(t.name=='revise_finish' and t.outcome=='success' for t in traces)
        assert not any(t.kind=='signal' for t in traces)
        assert store.list_reports('live')==[]


def test_actual_reduced_forecast_watchpoint_preserves_material_risk():
    q=next(q for q in qualifications() if isinstance(q,ForecastDeliveryRisk))
    c=Claim(kind='watchpoint',headline='Watch whether unstarted pipeline schemes break ground',text='Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply not yet started and vulnerable to delay or withdrawal; watch whether those schemes commence or are pulled.',evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],[])==[]


@pytest.mark.parametrize('mutation',[
    {'text':'Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of those schemes not yet started and vulnerable to delay or withdrawal.'},
    {'text':'Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply already started and vulnerable to delay or withdrawal.'},
    {'text':'Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply not yet started. Other projects are vulnerable to delay or withdrawal.'},
    {'text':'Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply not yet started marketing and vulnerable to delay or withdrawal.'},
    {'text':'Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply not yet started; unrelated projects are vulnerable to delay or withdrawal.'},
    {'headline':'Future plans','text':'Savills forecasts 22 million sq ft by end-2029 across 148 schemes, with 28% of that future supply not yet started and vulnerable to delay or withdrawal.'},
])
def test_reduced_forecast_watchpoint_still_requires_its_exact_material_relationship(mutation):
    q=next(q for q in qualifications() if isinstance(q,ForecastDeliveryRisk))
    c=Claim(kind='watchpoint',headline='Watch whether unstarted pipeline schemes break ground',text='Savills forecasts 22 million sq ft of Central London completions by end-2029 across 148 schemes, with 28% of that future supply not yet started and vulnerable to delay or withdrawal.',evidence_ids=[q.evidence_id])
    assert qualification_issues(c.model_copy(update=mutation),[q],[])


@pytest.mark.parametrize('kind',['fact','forecast','interpretation','watchpoint'])
def test_publisher_change_alone_cannot_conceal_inspected_endpoint_discrepancy(kind):
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange));cs=comparisons()
    c=Claim(kind=kind,headline='Central London reported supply 18.7m sq ft',text="Savills reported Central London office supply fell to 18.7 million sq ft during Q2 2026, down 0.5 million sq ft on Q1 2026 in the publisher's own text. Reported supply is not a vacancy rate.",evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],cs)
    repaired=c.model_copy(update={'text':c.text+' The calculated endpoint decline is 0.4 million sq ft; that calculation differs from the publisher-reported change.'})
    assert_typed_repair(qualification_issues(repaired,[q],cs))
    assert 'requires an actual compatible compare' in qualification_issues(c,[q],[])[0]


def test_nonchange_equal_quantity_does_not_activate_supply_discrepancy():
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange));cs=comparisons()
    c=Claim(kind='fact',text='A separate building has 0.5 million sq ft; Savills reports Central London supply of 18.7 million sq ft.',evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],cs)==[]


@pytest.mark.parametrize('text',[
    'The calculated decline in reported supply is 0.4 million sq ft. The publisher reports a decline of 0.5 million sq ft. The figures differ.',
    'The calculated 0.4 million sq ft decline differs from the publisher-reported 0.5 million sq ft decline.',
])
def test_reported_supply_metric_name_still_routes_to_typed_authoring(text):
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange))
    c=Claim(kind='fact',text=text,evidence_ids=[q.evidence_id])
    assert_typed_repair(qualification_issues(c,[q],comparisons()))


@pytest.mark.parametrize('text',[
    'West End take-up fell by 0.5 million sq ft. Savills reports Central London supply of 18.7 million sq ft.',
    'West End supply fell by 0.5 million sq ft. Savills reports Central London supply of 18.7 million sq ft.',
    'A separate building fell in size by 0.5 million sq ft. Savills reports Central London supply of 18.7 million sq ft.',
    'West End take-up fell by 0.5 million sq ft, while Savills reports Central London supply of 18.7 million sq ft.',
    'West End supply fell by 0.5 million sq ft, while Savills reports Central London supply of 18.7 million sq ft.',
])
def test_other_metric_or_geography_change_does_not_activate_supply_discrepancy(text):
    q=next(q for q in qualifications() if isinstance(q,SourceReportedChange))
    c=Claim(kind='fact',text=text,evidence_ids=[q.evidence_id])
    assert qualification_issues(c,[q],comparisons())==[]




@pytest.mark.parametrize('assertion,execution,accepted',[
    ('publisher','compatible',False),
    ('calculated','compatible',False),
    ('unrelated','compatible',False),
    ('disclosed','compatible',True),
    ('publisher','none',False),
    ('publisher','incompatible',False),
    ('publisher','failed',False),
    ('calculated','none',False),
    ('calculated','failed',False),
    ('calculated','incompatible',False),
])
def test_runner_keeps_executed_disclosure_separate_from_cited_arithmetic(tmp_path,assertion,execution,accepted):
    from tests.test_live_sources import page_for
    from london_market.config import Settings
    from london_market.domain import Task,Usage
    from london_market.providers import FixtureWebProvider,ModelReply,ToolCall
    from london_market.runner import Runner
    from london_market.sources import parse_monitored_source
    from london_market.store import Store

    with Store(tmp_path/'db') as store:
        for sid in ('savills-central-london-q1-2026','savills-central-london-q2-2026'):
            parsed=parse_monitored_source(sid,page_for(sid))
            store.ingest(parsed.document,parsed.evidence,parsed.observations)
        supply=sorted((o for o in store.list_observations('live') if o.metric=='reported_supply'),key=lambda o:o.period_start)
        endpoint=supply[-1].evidence_ids[0]
        cited=sorted({eid for o in supply for eid in o.evidence_ids}) if assertion=='disclosed' else [endpoint]
        texts={
            'publisher':'Savills reports Central London supply fell by 0.5 million sq ft on Q1 2026.',
            'calculated':'The calculated Central London supply decline is 0.4 million sq ft.',
            'unrelated':'A separate building changed by 0.4 million sq ft. Savills reports Central London supply of 18.7 million sq ft.',
            'disclosed':'The calculated Central London supply decline is 0.4 million sq ft. Savills reports a decline of 0.5 million sq ft. The rounded endpoint calculation differs from the publisher narrative.',
        }
        original={'claims':[{'kind':'fact','text':texts[assertion],'evidence_ids':cited}], 'coverage':{'scope':'focused','gaps':{}}}
        if assertion == 'disclosed':
            original['claims'] = [{'kind': 'fact', 'comparison': {'kind': 'publisher_endpoint_discrepancy', 'from_observation_id': supply[0].id, 'to_observation_id': supply[1].id}}]
        class Model:
            step=0
            def complete(self,messages,tools,max_output_tokens,timeout):
                if self.step==0:
                    actions=[('query_evidence',{'evidence_ids':cited})]
                    if execution!='none':
                        ids=[o.id for o in supply]
                        if execution=='failed':ids=['obs_missing',supply[-1].id]
                        elif execution=='incompatible':ids=[supply[0].id,next(o.id for o in store.list_observations('live') if o.metric=='vacancy_rate')]
                        actions.append(('compare',{'observation_ids':ids}))
                elif self.step==1:
                    actions=[('finish',original)]
                else:
                    feedback=json.loads(next(m['content'] for m in reversed(messages) if m['role']=='tool'))
                    assert not accepted
                    issues=feedback['audit_issues']
                    if execution!='compatible' or assertion=='unrelated':
                        assert not any('matching executed compatible comparison' in issue for issue in issues)
                    if assertion=='publisher':assert any('publisher_endpoint_discrepancy' in issue for issue in issues)
                    elif assertion=='calculated':assert any('claim 0: [0.4]'==issue for issue in issues)
                    else:
                        assert any('claim 0: [0.4]'==issue for issue in issues)
                        assert not any('publisher_endpoint_discrepancy' in issue for issue in issues)
                    actions=[('revise_finish',{'draft_id':feedback['draft_id'],'replacements':[{'index':0,'claim':{'kind':'fact','text':'Savills reports Central London supply of 18.7 million sq ft in Q2 2026.','evidence_ids':[endpoint]}}]})]
                self.step+=1
                assert self.step<=3
                return ModelReply(model='synthetic-disclosure-boundary',provider='synthetic',usage=Usage(),calls=[ToolCall(id=f'{self.step}-{i}',name=name,arguments=json.dumps(args)) for i,(name,args) in enumerate(actions)])
            def classify_signals(self,*args,**kwargs):raise AssertionError('Focused boundary proof requires no classifier')
        model=Model()
        run=Runner(Settings(),store,model=model,web=FixtureWebProvider()).run(Task(objective='Focused reported supply source check',entry_skill='research_market',mode='live'))
        assert model.step==(2 if accepted else 3)
        finishes=[t for t in store.list_traces(run.id) if t.name=='finish']
        assert finishes[0].outcome==('success' if accepted else 'failed')
        assert run.result.claims and store.list_reports('live')==[]


def v2_packet_and_supply_receipt():
    root=Path(__file__).parents[1]/'evidence/acceptance/requirements-coverage'
    p=json.loads((root/'build-20261011-v2-first-source-packet.json').read_text())
    trace=json.loads((root/'build-20261011-v2-trace.json').read_text())
    receipt=next(ComparisonResult.model_validate(e['data']['output']) for e in trace if e['name']=='compare' and e['outcome']=='success' and e['data']['output'].get('absolute_change')==-.4)
    return p,receipt








def v3_reduced_supply_claim_and_executed_compare(tmp_path):
    from london_market.config import Settings
    from london_market.domain import Document
    from london_market.store import Store
    from london_market.tools import CompareArgs, ToolContext, compare

    claim = Claim(
        kind='fact',
        headline='Reported supply 18.7m sq ft; reported and calculated changes differ',
        text='Savills reported Central London supply fell to 18.7 million sq ft during Q2 2026, down 0.5 million sq ft on Q1 2026 as published. The executed endpoint comparison of the two stored observations gives a calculated fall of 0.4 million sq ft, so the publisher-reported and calculated changes differ. This is reported supply, not currently available space.',
        evidence_ids=['ev_3098889467f5c7595e1aeab4', 'ev_9eac9e5e138886c912a5490b', 'ev_b14d903ddf16593631d59452'],
        topic_tags=['availability', 'supply'],
    )
    # The actual v3 retained leaves/pair are unchanged from the committed v2 source packet.
    p, recorded = v2_packet_and_supply_receipt()
    supporting = [Evidence.model_validate(e) for e in p['evidence'] if e['id'] in claim.evidence_ids]
    assert len(supporting) == 3
    observations = [Observation.model_validate(o) for o in p['observations'] if o['id'] in (recorded.from_observation_id, recorded.to_observation_id)]
    baseline = next(o for o in observations if o.id == recorded.from_observation_id)
    observations = [o.model_copy(update={'evidence_ids': sorted(set(recorded.evidence_ids)-set(baseline.evidence_ids))}) if o.id == recorded.to_observation_id else o for o in observations]
    with Store(tmp_path / 'v3-supply.db') as store:
        for raw in p['documents']:
            doc = Document.model_validate(raw)
            leaves = [e for e in supporting if e.document_id == doc.id]
            if leaves:
                store.ingest(doc, leaves, [])
        store.ingest(store.get_document(supporting[0].document_id, 'live'), [], observations)
        receipt = compare(ToolContext(Settings(), store, None, 'live', 'unpaid-v3-supply', 5), CompareArgs(observation_ids=[recorded.from_observation_id, recorded.to_observation_id]))
    assert receipt.compatible and receipt.absolute_change == -.4
    assert set(receipt.evidence_ids) == set(claim.evidence_ids)
    assert {d.reported.change.value for d in receipt.discrepancies} == {-.5}
    qs = [q for e in supporting for q in reviewed_source_qualifications(e)]
    assert len(qs) == 2
    return claim, qs, receipt








def v3_claim_source(index):
    from london_market.domain import Document
    root = Path(__file__).parents[1] / 'evidence/acceptance/requirements-coverage'
    p = json.loads((root / 'build-20261011-v3-source-review.json').read_text())
    candidate = next(row['body']['data']['candidate'] for row in reversed(p['traces']) if 'candidate' in row['body']['data'])
    original = candidate['claims'][index]
    evidence = next(Evidence.model_validate(row['body']) for row in p['evidence'] if row['id'] == original['evidence_ids'][0])
    document = next(Document.model_validate(row['body']) for row in p['documents'] if row['id'] == evidence.document_id)
    return original, evidence, document












def test_generic_default_absolute_obligation_is_not_promoted_with_known_source():
    current=next(q for q in qualifications() if isinstance(q,SourceReportedChange))
    assert current.requires_comparison is True
    historical=current.model_dump(exclude={'requires_comparison','basis'})
    historical['source_contract_id']='unflagged-absolute-contract'
    generic=SourceReportedChange.model_validate(historical)
    assert generic.basis=='absolute' and generic.requires_comparison is False
    claim=Claim(kind='fact',text='The publisher reports a Central London supply decline of 0.5 million sq ft.',evidence_ids=[generic.evidence_id])
    assert qualification_issues(claim,[generic],[])==[]
    assert qualification_issues(claim,[current],[])
