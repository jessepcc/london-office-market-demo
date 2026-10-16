from authoring_support import analysis_intents, analysis_sources, seed_analyses
import json


import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import Claim, Task, Usage
from london_market.providers import FixtureWebProvider, ModelReply, ToolCall
from london_market.runner import FinishArgs, Runner
from london_market.store import Store


def test_insufficient_data_cannot_publish_factual_message_outside_claims(tmp_path):
    class Model:
        def complete(self, messages, tools, max_output_tokens, timeout):
            return ModelReply(model='boundary-proof', provider='fixture', usage=Usage(), calls=[ToolCall(id='done', name='finish', arguments=json.dumps({'response_type':'insufficient_data','message':'Canary Wharf vacancy is 99% and rents doubled.'}))])
    with Store(tmp_path / 'market.db') as store:
        run = Runner(Settings(), store, model=Model(), web=FixtureWebProvider()).run(Task(objective='Research Canary Wharf', entry_skill='research_market'))
        assert run.status == 'partial'
        assert '99%' not in run.result.answer and 'rents doubled' not in run.result.answer
        assert 'insufficient' in run.result.answer


def test_partial_completion_keeps_typed_claims():
    result = FinishArgs(response_type='insufficient_data', claims=[Claim(kind='fact', text='A supported dated finding.', evidence_ids=['source'])])
    assert result.claims[0].evidence_ids == ['source']


def test_progress_observer_receives_persisted_outcomes(tmp_path):
    with Store(tmp_path / 'market.db') as store:
        observed = []
        def on_event(event):
            runs = store.list_runs('fixture')
            assert event == store.list_traces(runs[0].id)[-1]
            observed.append((event.kind, event.name, event.outcome))
        run = Runner(Settings(), store).run(Task(objective='Research London office vacancy', entry_skill='research_market'), on_event=on_event)
        assert run.status in ('completed','partial')
        assert ('progress','query_evidence','started') in observed
        assert ('tool','query_evidence','success') in observed
        assert observed[-1][0] == 'termination'
        assert len(observed) == len(store.list_traces(run.id))


def test_fresh_interpretation_requires_source_intent_while_history_stays_readable():
    saved = Claim(kind='interpretation', interpretation_role='source_view', text='The publisher expects tighter competition, subject to local supply.', evidence_ids=['source'])
    assert Claim.model_validate(saved.model_dump()).text == saved.text
    with pytest.raises(ValidationError, match='source-bound analysis or source_view intent'):
        FinishArgs(claims=[saved])
    supported = FinishArgs(claims=[{'kind': 'interpretation', 'interpretation_role': 'source_view', 'source_view': {'evidence_id': 'source'}}])
    assert supported.claims[0].source_view.evidence_id == 'source'


def test_citation_cannot_support_figures_outside_its_excerpt(tmp_path):
    from london_market.claim_audit import unsupported_figures
    from london_market.fixtures import load_fixtures
    with Store(tmp_path / 'market.db') as store:
        load_fixtures(store)
        source = store.list_evidence('fixture')[0]
        claim = Claim(kind='fact', text='Vacancy is 99%.', evidence_ids=[source.id])
        assert unsupported_figures(claim, [source]) == [99.0]
        exact = Claim(kind='fact', text=source.excerpt, evidence_ids=[source.id])
        assert unsupported_figures(exact, [source]) == []


def test_runner_rejects_a_cited_but_unsupported_numeric_claim(tmp_path):
    from london_market.fixtures import load_fixtures
    class Model:
        def __init__(self, evidence_id):
            self.evidence_id = evidence_id
        def complete(self, messages, tools, max_output_tokens, timeout):
            errors = [json.loads(item['content']) for item in messages if item['role'] == 'tool']
            claims = [{'kind':'fact','text':'Vacancy is 99%.','evidence_ids':[self.evidence_id]}]
            args = {'claims':claims} if not errors else {'response_type':'insufficient_data'}
            return ModelReply(model='numeric-audit-proof', provider='fixture', usage=Usage(), calls=[ToolCall(id=str(len(errors)), name='finish', arguments=json.dumps(args))])
    with Store(tmp_path / 'market.db') as store:
        load_fixtures(store)
        source = store.list_evidence('fixture')[0]
        run = Runner(Settings(), store, model=Model(source.id)).run(Task(objective='Research vacancy', entry_skill='research_market'))
        assert '99%' not in run.result.answer
        failure = next(event for event in store.list_traces(run.id) if event.name == 'finish' and event.outcome == 'failed')
        assert 'Completion audit failed' in failure.data['error']


def test_gap_notes_allow_dates_but_refuse_uncited_statistics():
    from london_market.claim_audit import note_figures
    assert note_figures('Retained Q2 2026 data retrieved on 2026-10-09.') == {2026.0}
    assert note_figures('Unverified vacancy 99% found on 2026-10-09.') == {99.0}
    assert 99 in note_figures('Invalid calendar date 2026-99-09')


def test_compact_figures_and_dated_gap_notes():
    from london_market.claim_audit import figures, note_figures
    assert figures('2.4m sq ft; 20bps; vacancy 7.2%. Q2 2026.') == {2.4, 20, 7.2, 2026}
    assert note_figures('Retrieved 9 October 2026; no Q3 2026 evidence.') == {2026}


def test_uncited_numeric_watchpoint_is_rejected(tmp_path):
    class Model:
        def complete(self, messages, tools, max_output_tokens, timeout):
            error = any(item.get('role') == 'tool' for item in messages)
            args = {'response_type':'insufficient_data'} if error else {'claims':[{'kind':'watchpoint','text':'Canary Wharf vacancy is 99%. Monitor whether it falls.'}]}
            return ModelReply(model='watchpoint-proof',provider='fixture',usage=Usage(),calls=[ToolCall(id='done'+str(error),name='finish',arguments=json.dumps(args))])
    with Store(tmp_path / 'market.db') as store:
        run = Runner(Settings(),store,model=Model()).run(Task(objective='Research Canary Wharf',entry_skill='research_market'))
        assert '99%' not in run.result.answer
        assert any(event.name == 'finish' and event.outcome == 'failed' for event in store.list_traces(run.id))




@pytest.mark.parametrize('always_empty', [False, True])
def test_empty_provider_response_gets_only_one_bounded_retry(tmp_path, always_empty):
    class Model:
        phase = 0
        def complete(self, messages, tools, max_output_tokens, timeout):
            self.phase += 1
            if self.phase == 1 or always_empty:
                return ModelReply(model='empty-response-proof',provider='fixture',finish_reason='stop',usage=Usage())
            assert {tool['function']['name'] for tool in tools} == {'finish'}
            return ModelReply(model='empty-response-proof',provider='fixture',usage=Usage(),calls=[ToolCall(id='done',name='finish',arguments='{"response_type":"insufficient_data"}')])
    with Store(tmp_path / 'market.db') as store:
        model = Model()
        run = Runner(Settings(),store,model=model).run(Task(objective='Research Canary Wharf',entry_skill='research_market'))
        assert run.status == ('failed' if always_empty else 'partial')
        assert model.phase == 2
        assert len([event for event in store.list_traces(run.id) if event.name=='model_response' and event.outcome=='retry']) == 1


def test_gap_notes_distinguish_address_numbers_from_statistics():
    from london_market.claim_audit import note_figures
    assert note_figures('The 10 Upper Bank Street lease expiry is not a handover date.') == set()
    assert note_figures('Vacancy is 10% near 10 Upper Bank Street.') == {10}


def test_forecast_evidence_cannot_finish_as_observed_fact(tmp_path):
    from london_market.fixtures import load_fixtures
    from london_market.domain import Evidence
    class Model:
        def __init__(self, evidence_id): self.evidence_id = evidence_id
        def complete(self, messages, tools, max_output_tokens, timeout):
            repaired = any(item["role"] == "tool" for item in messages)
            claim = {"kind": "forecast" if repaired else "fact", "text": "The source projects future completions.", "evidence_ids": [self.evidence_id]}
            return ModelReply(model="forecast-boundary", provider="fixture", usage=Usage(), calls=[ToolCall(id="done", name="finish", arguments=json.dumps({"claims": [claim]}))])
    with Store(tmp_path / "market.db") as store:
        load_fixtures(store)
        original = store.list_evidence("fixture")[0]
        projected = Evidence.model_validate({**original.model_dump(exclude={"id"}), "excerpt": "The source projects future completions.", "statement_kind": "forecast"})
        store.ingest(store.get_document(original.document_id), [projected], [])
        run = Runner(Settings(), store, model=Model(projected.id), web=FixtureWebProvider()).run(Task(objective="Research supply", entry_skill="research_market"))
        assert run.result.claims[0].kind == "forecast"
        assert any(event.name == "finish" and event.outcome == "failed" for event in store.list_traces(run.id))


def test_completion_returns_all_indexed_repairs_in_one_response(tmp_path):
    from london_market.fixtures import load_fixtures
    from london_market.domain import Evidence
    class Model:
        def __init__(self, evidence_id): self.evidence_id, self.feedback = evidence_id, None
        def complete(self, messages, tools, max_output_tokens, timeout):
            schema = next(item for item in tools if item["function"]["name"] == "finish")["function"]["parameters"]
            assert self.evidence_id in schema["$defs"]["Claim"]["properties"]["evidence_ids"]["items"]["enum"]
            replies = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
            if replies:
                self.feedback = replies[-1]
                draft = next(item for item in reversed(messages) if item.get("tool_calls"))["tool_calls"][0]
                assert json.loads(draft["function"]["arguments"])["rejected_draft_id"] == self.feedback["draft_id"]
                assert self.feedback["draft_claim_index"][0]["subject"] == "Future completions are projected."
                assert self.feedback["error_claims"][0]["claim"]["text"] == "Future completions are projected."
                args = {"response_type": "insufficient_data"}
            else:
                args = {"claims": [{"kind": "fact", "text": "Future completions are projected.", "evidence_ids": [self.evidence_id]}, {"kind": "fact", "text": "An unknown source says so.", "evidence_ids": ["ev_missing"]}], "topic_gaps": ["Vacancy is 99%."]}
            return ModelReply(model="audit-repair", provider="fixture", usage=Usage(), calls=[ToolCall(id="done", name="finish", arguments=json.dumps(args))])
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        original = store.list_evidence("fixture")[0]
        source = Evidence.model_validate({**original.model_dump(exclude={"id"}), "excerpt": "Future completions are projected.", "statement_kind": "forecast"})
        store.ingest(store.get_document(original.document_id), [source], [])
        model = Model(source.id)
        run = Runner(Settings(),store,model=model,web=FixtureWebProvider()).run(Task(objective="Research London",entry_skill="research_market"))
        assert run.status == "partial"
        issues = model.feedback["audit_issues"]
        assert len(issues) == 3
        assert "topic_gaps[0]" in issues[0]
        assert "claim 0" in issues[1] and "forecast" in issues[1]
        assert "claim 1" in issues[2] and "ev_missing" in issues[2]
        failed = next(event for event in store.list_traces(run.id) if event.name == "finish" and event.outcome == "failed")
        assert failed.data["arguments"]["claims"][0]["text"] == "Future completions are projected."
        assert failed.data["arguments"]["claims"][1]["evidence_ids"] == ["ev_missing"]


def test_reference_period_year_is_supported_even_when_publication_is_later():
    from datetime import date, datetime, timezone
    from london_market.domain import Evidence
    from london_market.claim_audit import unsupported_figures
    source = Evidence(document_id="source",mode="live",excerpt="City Fringe vacancy stood at 8.2%.",published_at=date(2026,1,30),retrieved_at=datetime(2026,10,9,tzinfo=timezone.utc),source_url="https://example.com/quarter",native_geography="City Fringe",mapped_geography="Midtown and Fringe",category="vacancy")
    claim = Claim(kind="fact",text="City Fringe vacancy was 8.2% in Q4 2025.",evidence_ids=[source.id])
    assert unsupported_figures(claim,[source]) == [2025]
    assert unsupported_figures(claim,[source],reference_dates=[date(2025,10,1),date(2025,12,31)]) == []
    unrelated = claim.model_copy(update={"text":"City Fringe vacancy was 8.2% in Q4 2024."})
    assert unsupported_figures(unrelated,[source],reference_dates=[date(2025,10,1),date(2025,12,31)]) == [2024]


@pytest.mark.parametrize("initial_reference",["valid","foreign","omitted"])
def test_claim_chart_reference_audits_sources_without_changing_direct_citations(tmp_path,initial_reference):
    from london_market.fixtures import load_fixtures
    from london_market.report import chart_reference_id, export_report_html
    class Model:
        def __init__(self,ids,text):self.ids,self.text,self.round=ids,text,0
        def complete(self,messages,tools,max_output_tokens,timeout):
            self.round += 1
            outputs = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
            if self.round == 1:
                name,args="choose_visualization",{"observation_ids":self.ids,"title":"Fictional vacancy history","purpose":"Show synthetic reported values","preferred":"trend"}
            elif self.round == 2:
                name,args="prepare_visualization_data",{"plan_id":outputs[-1]["plan_id"]}
            else:
                chart_id=next(output["chart_id"] for output in outputs if "chart_id" in output)
                reference="chart_foreign" if self.round==3 and initial_reference=="foreign" else chart_id
                selected=[] if self.round==3 and initial_reference=="omitted" else [reference]
                name,args="finish",{"claims":[{"kind":"fact","text":self.text,"chart_ids":[reference]}],"chart_ids":selected}
            return ModelReply(model="synthetic-chart-citation",provider="fixture",usage=Usage(),calls=[ToolCall(id=str(self.round),name=name,arguments=json.dumps(args))])
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        items=sorted([item for item in store.list_observations("fixture") if item.mapped_geography=="City" and item.metric=="vacancy_rate"],key=lambda item:item.period_start)
        text=f"Fictional City vacancy readings were {items[0].value}% and {items[1].value}% across the displayed quarters."
        run=Runner(Settings(),store,model=Model([item.id for item in items],text),web=FixtureWebProvider()).run(Task(objective="Chart City vacancy",entry_skill="research_market"))
        assert run.status=="partial"
        chart=run.result.charts[0]
        claim=run.result.claims[0]
        assert claim.evidence_ids == []
        assert run.result.evidence_ids == sorted(chart.source_refs)
        assert claim.chart_ids==[chart_reference_id(chart)]
        assert store.get_run(run.id).result.claims[0]==claim
        if initial_reference!="valid": assert any(event.name=="finish" and event.outcome=="failed" for event in store.list_traces(run.id))
        sources=[store.get_evidence(eid) for eid in run.result.evidence_ids]
        artifact=export_report_html(run.result,sources,items)
        assert 'href="#exhibit-0"' in artifact and 'id="exhibit-0"' in artifact


def test_completion_projection_retains_exact_qualifications_segments_and_provenance():
    from london_market.runner import compact_completion_context
    evidence={"id":"source","document_id":"document","source_url":"https://example.com/source","excerpt":"A projected measure remains a forecast.","published_at":"2026-08-06","statement_kind":"forecast","validation_reason":"Future supply is not available space."}
    point={"value":23.5,"unit":"%","segment":"Media & Tech","publisher":"BNP","definition":"Source-reported occupier-sector share","observation_id":"observation","evidence_ids":["source"],"period_start":"2026-01-01","period_end":"2026-03-31"}
    messages=[{"role":"assistant","tool_calls":[{"id":"query","function":{"name":"query_evidence"}},{"id":"chart","function":{"name":"prepare_visualization_data"}}]},{"role":"tool","tool_call_id":"query","content":json.dumps({"evidence":[evidence]})},{"role":"tool","tool_call_id":"chart","content":json.dumps({"chart_id":"chart","payload":{"points":[point],"source_refs":["source"]}})}]
    compact_completion_context(messages)
    retained=json.loads(messages[1]["content"])["evidence"][0]
    assert (retained["excerpt"],retained["published_at"],retained["statement_kind"],retained["validation_reason"]) == (evidence["excerpt"],evidence["published_at"],"forecast",evidence["validation_reason"])
    payload=json.loads(messages[2]["content"])["payload"]
    assert payload["points"][0]["segment"]=="Media & Tech" and payload["points"][0]["evidence_ids"]==["source"]
    assert payload["measurement_scopes"]==[["BNP","Source-reported occupier-sector share"]]
    snapshot=json.dumps(messages)
    compact_completion_context(messages)
    assert json.dumps(messages)==snapshot
