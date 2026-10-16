from datetime import date

import pytest
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import Claim, Observation, ReportSection, RunResult, Task
from london_market.fixtures import scripted_coverage, load_fixtures
from london_market.runner import FinishArgs
from london_market.store import Store
from london_market.tools import CompareArgs, ToolContext, VisualizationArgs, compare, choose_visualization, prepare_visualization_data, PrepareArgs


def test_sections_references_and_legacy():
    claim = Claim(kind="fact", text="Supported", evidence_ids=["ev"])
    assert RunResult(status="partial", answer="Legacy", reason="Old").sections == []
    for model, kwargs in ((FinishArgs, {"claims": [claim]}), (RunResult, {"status": "partial", "answer": "", "reason": "test", "claims": [claim]})):
        with pytest.raises(ValidationError, match="dangling"):
            model(**kwargs, sections=[ReportSection(kind="key_trends", claim_indexes=[1])])
    with pytest.raises(ValidationError):
        ReportSection(kind="key_trends", body="Uncited prose")


def test_rendered_report_does_not_repeat_claims_across_sections():
    from london_market.report import export_report_html, report_groups
    result = RunResult(status="partial", answer="", reason="review", claims=[Claim(kind="watchpoint", text="Check the next release.")], sections=[ReportSection(kind="executive_findings", claim_indexes=[0]), ReportSection(kind="outlook", claim_indexes=[0])])
    assert list(report_groups(result)) == [("executive_findings", "Executive findings", [0], [])]
    assert export_report_html(result, [], []).count("Check the next release.") == 1


def test_key_figures_use_only_cited_latest_unambiguous_central_london_measurements(tmp_path):
    from london_market.report import export_report_html, report_key_figures
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        source = store.list_evidence("fixture")[0]
        base = store.list_observations("fixture")[0]
    observation = Observation(**{**base.model_dump(exclude={"id"}), "metric": "vacancy_rate", "value": 8, "unit": "%", "native_geography": "Central London", "mapped_geography": "Central London", "evidence_ids": [source.id]})
    latest = Observation(**{**observation.model_dump(exclude={"id"}), "value": 7.5, "period_start": date(2026, 4, 1), "period_end": date(2026, 6, 30)})
    result = RunResult(status="partial", answer="", reason="test", observation_ids=[observation.id, latest.id])
    assert report_key_figures(result, [source], [observation, latest]) == [(latest, "Vacancy rate")]
    rendered = export_report_html(result, [source], [observation, latest])
    assert '7.5<small>%</small>' in rendered
    assert 'Vacancy rate · Central London' in rendered and '2026-04-01 to 2026-06-30' in rendered
    assert latest.definition in rendered and f'href="#evidence-{source.id}"' in rendered
    assert report_key_figures(result, [], [latest]) == []
    assert report_key_figures(result.model_copy(update={"observation_ids": []}), [source], [latest]) == []
    conflict = Observation(**{**latest.model_dump(exclude={"id"}), "value": 9})
    disputed = result.model_copy(update={"observation_ids": [latest.id, conflict.id]})
    assert report_key_figures(disputed, [source], [latest, conflict]) == []
    submarket = Observation(**{**latest.model_dump(exclude={"id"}), "native_geography": "City", "mapped_geography": "City"})
    assert report_key_figures(result.model_copy(update={"observation_ids": [submarket.id]}), [source], [submarket]) == [(submarket, "Vacancy rate")]


def test_source_qualification_and_chart_dedup_survive_report_export(tmp_path):
    from london_market.domain import ChartPayload
    from london_market.report import export_report_html, report_groups
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        source = store.list_evidence("fixture")[0].model_copy(update={"validation_reason": "Qualified publisher extract; no evidence for wider London."})
    chart = ChartPayload(chart_type="insufficient_data", title="Missing comparable history", purpose="Retain missing periods")
    result = RunResult(status="partial", answer="", reason="test", charts=[chart], sections=[ReportSection(kind="key_trends", chart_indexes=[0]), ReportSection(kind="outlook", chart_indexes=[0])])
    assert len(list(report_groups(result))) == 1
    artifact = export_report_html(result, [source], [])
    assert artifact.count("Missing comparable history") == 1
    assert source.validation_reason in artifact


def test_deterministic_percentage_points_relative_growth_and_zero(tmp_path):
    with Store(tmp_path / "db") as store:
        load_fixtures(store)
        ctx = ToolContext(Settings(), store, None, "fixture", "test", 5)
        items = sorted([item for item in store.list_observations("fixture") if item.metric == "vacancy_rate" and item.mapped_geography == "City"], key=lambda item: item.period_start)
        result = compare(ctx, CompareArgs(observation_ids=[items[0].id, items[1].id]))
        assert result.unit == "percentage points"
        assert result.absolute_change == -0.5
        assert result.relative_growth_percent == -6.25
        assert result.baseline_period_start == items[0].period_start
        assert result.endpoint_period_end == items[1].period_end
        zero = Observation(**{**items[0].model_dump(exclude={"id"}), "metric": "zero_demo", "value": 0})
        end = Observation(**{**items[1].model_dump(exclude={"id"}), "metric": "zero_demo", "value": 2})
        store.ingest(store.get_document(store.get_evidence(zero.evidence_ids[0], "fixture").document_id, "fixture"), [], [zero, end])
        change = compare(ctx, CompareArgs(observation_ids=[zero.id, end.id]))
        assert change.absolute_change == 2 and change.relative_growth_percent is None


def test_multiseries_history_provenance_and_honest_fallbacks(tmp_path):
    from london_market.report_demo import seed_report_demo
    with Store(tmp_path / "db") as store:
        seed_report_demo(store)
        ctx = ToolContext(Settings(), store, None, "fixture", "test", 5)
        items = [item for item in store.list_observations("fixture") if item.metric == "vacancy_rate"]
        def plan(selection, preferred="trend"):
            return choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in selection], title="Evidence", purpose="Compare compatible measures", preferred=preferred))
        result = plan(items)
        assert result.plan.chart_type == "multi_line"
        chart = prepare_visualization_data(ctx, PrepareArgs(plan_id=result.plan_id)).payload
        assert len(chart.points) == 8
        assert sorted(point.value for point in chart.points if point.geography == "City") == [7, 7.5, 7.8, 8]
        assert all(store.get_observation(point.observation_id).evidence_ids == point.evidence_ids for point in chart.points)
        assert plan(items[:-1]).plan.chart_type == "table"
        gapped = plan([item for item in items if not (item.mapped_geography == "City" and item.period_start == date(2025, 10, 1))])
        assert gapped.plan.chart_type == "table" and gapped.plan.caveats
        assert plan(items, "submarkets").plan.chart_type == "table"
        one = items[0]
        incompatible = Observation(**{**one.model_dump(exclude={"id"}), "publisher": "Another publisher"})
        aggregate = Observation(**{**one.model_dump(exclude={"id"}), "native_geography": "Central London", "mapped_geography": "Central London"})
        conflict = Observation(**{**one.model_dump(exclude={"id"}), "value": 99})
        document = store.get_document(store.get_evidence(one.evidence_ids[0], "fixture").document_id, "fixture")
        store.ingest(document, [], [incompatible, aggregate])
        assert plan([*items[1:], incompatible]).plan.chart_type == "table"
        assert plan([*items, aggregate]).plan.chart_type == "table"
        store.ingest(document, [], [conflict])
        assert plan(items).plan.chart_type == "insufficient_data"


def test_report_demo_shared_runner_saved_export_and_no_requests(tmp_path, monkeypatch):
    import socket
    from london_market.report import export_report_html
    from london_market.report_demo import generate_report_demo
    monkeypatch.setattr(socket, "socket", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("no external requests")))
    with Store(tmp_path / "demo.db") as store:
        run, report_id = generate_report_demo(Settings(), store)
        assert run.status == "partial"
        assert run.task.limits.max_steps == 192
        assert len(run.result.sections) == 7
        assert [chart.chart_type for chart in run.result.charts] == ["multi_line", "multi_line", "bar", "bar", "relation", "total"]
        assert "fell by 1 percentage point" in run.result.claims[0].text
        assert "-12.5%" in run.result.claims[0].text
        assert "Meridian Analytics" in run.result.answer and "Orchard Design" in run.result.answer
        assert all("fictional" in claim.text.lower() for claim in run.result.claims[3:5])
        assert store.list_reports("fixture")[0]["id"] == report_id
        assert store.list_traces(run.id)[-1].data["steps"] <= run.task.limits.max_steps
        evidence = [store.get_evidence(eid, "fixture") for eid in run.result.evidence_ids]
        observations = [store.get_observation(oid, "fixture") for oid in run.result.observation_ids]
        artifact = export_report_html(run.result, evidence, observations)
        assert 'Plotly.newPlot' in artifact and 'Source appendix' in artifact and 'Fictional London Office Observatory' in artifact
        assert '<script src=' not in artifact
        assert 'Meridian Analytics' in artifact and 'source-appendix' in artifact
        for item in observations:
            assert item.id in artifact


def test_demo_honors_configured_steps_without_explicit_override(tmp_path):
    from london_market.domain import Limits
    from london_market.report_demo import generate_report_demo
    with Store(tmp_path / "db") as store:
        run, report_id = generate_report_demo(Settings(limits=Limits(max_steps=2)), store)
        assert run.status == "stopped" and report_id is None
        assert run.task.limits.max_steps == 2
        with pytest.raises(ValidationError):
            generate_report_demo(Settings(), store, max_steps=193)


def test_export_escapes_prose_source_urls_and_chart_serialization(tmp_path):
    from london_market.domain import ChartPayload, ChartPoint
    from london_market.report import export_report_html
    from london_market.report_demo import seed_report_demo
    attack = '</script><script>alert("bad")</script><img src=x onerror=alert(1)>'
    with Store(tmp_path / "db") as store:
        seed_report_demo(store)
        source = store.list_evidence("fixture")[0].model_copy(update={"excerpt": attack})
        observation = store.list_observations("fixture")[0]
        point = ChartPoint(label=attack, value=observation.value, observation_id=observation.id, evidence_ids=[source.id], period_start=observation.period_start, period_end=observation.period_end, geography=observation.mapped_geography, metric=observation.metric, unit=attack, publisher=attack, definition=attack, category=observation.category)
        chart = ChartPayload(chart_type="line", title=attack, purpose=attack, unit=attack, x_label=attack, points=[point], source_refs=[source.id])
        result = RunResult(status="partial", answer=attack, reason="Test", claims=[Claim(kind="fact", text=attack, evidence_ids=[source.id])], charts=[chart])
        artifact = export_report_html(result, [source], [observation], title=attack)
        assert attack not in artifact
        assert '&lt;/script&gt;' in artifact
        assert '<script>alert(' not in artifact
        assert '\\u003c' in artifact or '\\u0026lt;' in artifact


def test_structured_report_actual_streamlit_app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from london_market.report_demo import generate_report_demo
    database = tmp_path / "ui.db"
    with Store(database) as store:
        generate_report_demo(Settings(), store)
    monkeypatch.setenv("LONDON_MARKET_DB", str(database))
    monkeypatch.setenv("LONDON_MARKET_MODE", "fixture")
    from ui_support import app_test
    app = app_test(database)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 6
    assert {"This week", "Market evidence", "Market exhibits", "Landlord analysis", "What to watch"}.issubset({item.value for item in app.subheader})
    assert any("Meridian Analytics" in item.value for item in app.markdown)
    assert len(app.get("download_button")) == 1


def test_weekly_output_limit_is_reserved_exactly_and_budget_still_blocks(tmp_path):
    from london_market.domain import Limits
    from london_market.providers import ModelReply, ToolCall
    from london_market.runner import Runner
    import json
    class Model:
        called = False
        def complete(self, messages, tools, max_output_tokens, timeout):
            assert max_output_tokens == 6000
            if not self.called:
                self.called = True
                context_line = next(line for message in messages if message["role"] == "system" for line in message["content"].splitlines() if line.startswith("Pinned weekly context: "))
                context = json.loads(context_line.removeprefix("Pinned weekly context: "))
                return ModelReply(model="mock", provider="mock", usage={"cost_known": False}, calls=[ToolCall(id="current-window", name="query_evidence", arguments=json.dumps({"published_since": context["window_start"], "published_until": context["window_end"], "offset": 0}))])
            observed = next(message for message in messages if message["role"] == "tool" and message["tool_call_id"] == "current-window")
            assert json.loads(observed["content"])["evidence"] == []
            return ModelReply(model="mock", provider="mock", usage={"cost_known": False}, calls=[ToolCall(id="finish", name="finish", arguments=json.dumps({"response_type": "insufficient_data", "claims": [], "coverage": scripted_coverage([], [], [])}))])
    settings = Settings(mode="live")
    with Store(tmp_path / "db") as store:
        model = Model()
        run = Runner(settings, store, model=model).run(Task(objective="Prepare a London office report", entry_skill="prepare_weekly_briefing", mode="live"))
        assert model.called and run.status == "partial"
        reserved = [event for event in store.list_traces(run.id) if event.kind == "model" and event.outcome == "reserved"]
        assert [event.data["output_token_bound"] for event in reserved] == [6000, 6000]
        assert run.usage.output_tokens == 12000
        blocked_model = Model()
        task = run.task.model_copy(update={"limits": Limits(max_tokens=3999)})
        blocked = Runner(settings, store, model=blocked_model).run(task)
        assert blocked.status == "stopped" and not blocked_model.called


def test_skill_frontmatter_versions_and_links_are_bounded():
    import re
    from london_market.skills import default_registry
    registry = default_registry()
    for skill in registry.skills.values():
        text = skill.instructions_path.read_text()
        assert text.startswith('---\nname: ') and '\ndescription: ' in text
        assert len(text.encode()) <= 12000
        expected_versions = {"research_market": "29", "prepare_weekly_briefing": "45", "collect_market_evidence": "6", "analyze_market_trends": "5", "choose_visualization": "8", "prepare_visualization_data": "5", "audit_market_data": "3", "analyze_leasing_activity": "6", "analyze_macro_drivers": "7", "analyze_office_rents": "6"}
        assert skill.version == expected_versions.get(skill.id, "5" if skill.id.startswith("analyze_") else "3")
        for relative in re.findall(r'\]\(([^)]+)\)', text):
            assert (skill.instructions_path.parent / relative).is_file()


def test_export_missing_sources_are_unsupported_without_dangling_links():
    from london_market.domain import ChartPayload
    from london_market.report import export_report_html
    result = RunResult(status="partial", answer="Restored report", reason="Source unavailable", evidence_ids=["missing-source"], claims=[Claim(kind="fact", text="Dated claim", evidence_ids=["missing-source"]), Claim(kind="forecast", text="Source forecast", evidence_ids=["missing-forecast"])], charts=[ChartPayload(chart_type="insufficient_data", title="No history", purpose="Missing source", source_refs=["missing-chart-source"])])
    artifact = export_report_html(result, [], [])
    assert 'Some cited sources are unavailable' in artifact
    assert 'Unsupported fact' in artifact and 'Unsupported forecast' in artifact
    assert 'Missing-source manifest' in artifact
    assert 'missing-chart-source. Citation cannot currently be inspected.' in artifact
    assert 'href="#evidence-missing' not in artifact


def test_legacy_claimless_chart_result_renders_in_streamlit(tmp_path, monkeypatch):
    from datetime import datetime
    from streamlit.testing.v1 import AppTest
    from london_market.domain import ChartPayload, RunRecord, utc_now
    database = tmp_path / "claimless.db"
    result = RunResult(status="partial", answer="Insufficient measured history", reason="Legacy report", charts=[ChartPayload(chart_type="insufficient_data", title="No comparable history", purpose="Retain honest chart fallback")])
    run = RunRecord(task=Task(objective="Legacy report", entry_skill="prepare_weekly_briefing"), status="partial", main_model="fixture", router_model="fixture", result=result, finished_at=utc_now())
    with Store(database) as store:
        store.save_run(run)
        store.save_report(run.id, datetime(2026, 10, 5).date(), result)
    monkeypatch.setenv("LONDON_MARKET_DB", str(database))
    monkeypatch.setenv("LONDON_MARKET_MODE", "fixture")
    from ui_support import app_test
    app = app_test(database)
    app.radio(key="view").set_value("Weekly reports").run()
    assert not app.exception
    assert any("Insufficient comparable observations" in item.value for item in app.info)
    assert any("No comparable history" in item.value for item in app.markdown)


def test_displayed_flow_total_excludes_stocks_overlap_and_mixed_units():
    from london_market.domain import ChartPayload, ChartPoint
    from london_market.report import displayed_flow_total
    base = ChartPoint(label="Q1",value=2.2,observation_id="first",evidence_ids=["source"],period_start=date(2026,1,1),period_end=date(2026,3,31),geography="Central London",native_geography="Central London",metric="take_up",unit="million sq ft",publisher="Savills",definition="Quarterly take-up",category="take-up")
    later = base.model_copy(update={"label":"Q2","value":2.4,"observation_id":"second","period_start":date(2026,4,1),"period_end":date(2026,6,30)})
    chart = ChartPayload(chart_type="total",title="Quarterly take-up",purpose="Leasing volumes",points=[base,later],source_refs=["source"])
    assert displayed_flow_total(chart) == "Calculated sum of displayed periods: 4.6 million sq ft"
    assert displayed_flow_total(chart.model_copy(update={"points":[base,later.model_copy(update={"unit":"sq ft"})]})) is None
    assert displayed_flow_total(chart.model_copy(update={"points":[base,base]})) is None
    assert displayed_flow_total(chart.model_copy(update={"points":[point.model_copy(update={"metric":"vacancy_rate"}) for point in chart.points]})) is None


@pytest.mark.parametrize('response_type', ['answer', 'insufficient_data'])
def test_runner_rejects_omitted_available_purposes_then_tools_repair_before_classification(tmp_path, response_type):
    import json
    from london_market.report_demo import ReportDemoModel, seed_report_demo
    from london_market.providers import FixtureModelProvider, ModelReply, ToolCall
    from london_market.domain import Usage
    from london_market.runner import Runner

    class RepairModel(ReportDemoModel):
        def __init__(self):
            super().__init__()
            self.initial_budget = None
            self.missing = None

        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.phase == 0:
                self.initial_budget = json.loads(next(m['content'].removeprefix('Run budget: ') for m in messages if m.get('content', '').startswith('Run budget: ')))
            if self.phase in (4, 5):
                errors = [json.loads(m['content']) for m in messages if m.get('role') == 'tool' and 'visual_purposes' in m.get('content', '')]
                feedback = errors[-1]
                if self.phase == 4:
                    self.missing = feedback['visual_purposes']
                    assert feedback['ready_visual_purpose_chart_ids'] == {}
                    assert all('Choose and prepare' in issue for issue in feedback['audit_issues'] if 'Missing supported visual purpose:' in issue)
                    actions = [('choose_visualization', entry['candidate']) for entry in self.missing.values() if entry['status'] == 'missing']
                else:
                    ids = {c['id'] for m in messages for c in m.get('tool_calls', []) if c['id'].startswith('repair-4-')}
                    actions = [('prepare_visualization_data', {'plan_id': json.loads(m['content'])['plan_id']}) for m in messages if m.get('role') == 'tool' and m['tool_call_id'] in ids]
                phase = self.phase
                self.phase += 1
                return ModelReply(model='synthetic-purpose-repair', provider='synthetic', usage=Usage(), calls=[ToolCall(id=f'repair-{phase}-{i}', name=name, arguments=json.dumps(args)) for i, (name, args) in enumerate(actions)])
            if self.phase == 6:
                ids = {c['id'] for m in messages for c in m.get('tool_calls', []) if c['id'].startswith('repair-5-')}
                repaired_ids = [json.loads(m['content'])['chart_id'] for m in messages if m.get('role') == 'tool' and m['tool_call_id'] in ids]
                raw = {**self.original, 'chart_ids': self.original['chart_ids'] + repaired_ids}
                self.phase += 1
                return ModelReply(model='synthetic-purpose-repair', provider='synthetic', usage=Usage(), calls=[ToolCall(id='repaired-finish', name='finish', arguments=json.dumps(raw))])
            reply = super().complete(messages, tools, max_output_tokens, timeout)
            if self.phase == 2:
                reply = reply.model_copy(update={'calls': [c for c in reply.calls if c.name != 'choose_visualization' or json.loads(c.arguments)['preferred'] not in ('relation', 'total')]})
            # Start with the former four charts even after the fixture caller migrates.
            if self.phase == 4:
                raw = json.loads(reply.calls[0].arguments)
                raw['chart_ids'] = raw['chart_ids'][:4]
                raw['response_type'] = response_type
                self.original = raw
                reply = reply.model_copy(update={'calls': [reply.calls[0].model_copy(update={'arguments': json.dumps(raw)})]})
            return reply

        def classify_signals(self, cards, timeout):
            assert self.phase == 7, 'classification must wait until chart repair'
            return FixtureModelProvider(task).classify_signals(cards, timeout)

    task = Task(objective='Prepare a whole London weekly report', entry_skill='prepare_weekly_briefing', mode='fixture', trigger='fixture', briefing_date=date(2026, 10, 9), report_week_start=date(2026, 10, 5))
    model = RepairModel()
    with Store(tmp_path / 'repair.db') as store:
        seed_report_demo(store)
        run = Runner(Settings(), store, model=model).run(task)
        assert run.status == 'partial', [(t.name, t.outcome, t.data) for t in store.list_traces(run.id) if t.outcome in ('failed', 'error')]
        assert model.phase == 7
        assert {k: v['status'] for k, v in model.initial_budget['visual_purposes'].items()} == {'trend': 'missing', 'relationship': 'missing', 'sum': 'missing', 'proportion': 'unsupported'}
        assert model.initial_budget['visualization_tool_calls_allowance'] == 8
        assert model.missing['relationship']['status'] == model.missing['sum']['status'] == 'missing'
        traces = store.list_traces(run.id)
        rejected = next(i for i, t in enumerate(traces) if t.name == 'finish' and t.outcome == 'failed')
        assert all(t.kind != 'signal' for t in traces[:rejected])
        final = next(t for t in traces if t.name == 'finish' and t.outcome == 'success')
        assert final.data['visual_purposes']['relationship']['status'] == 'included'
        assert final.data['visual_purposes']['sum']['status'] == 'included'
        assert 'SYNTHETIC visual-purpose gap: No compatible directly reported occupier-sector partition' in run.result.answer



@pytest.mark.parametrize('method', ['finish', 'revise_finish'])
def test_selected_registered_payloads_only_and_prepared_omission_repaired_by_selection(tmp_path, method):
    import json
    from london_market.report_demo import ReportDemoModel, seed_report_demo
    from london_market.providers import ModelReply, ToolCall
    from london_market.domain import Usage
    from london_market.runner import Runner

    class SelectionModel(ReportDemoModel):
        def complete(self, messages, tools, max_output_tokens, timeout):
            if self.phase == 2:
                budget = json.loads(next(m['content'].removeprefix('Run budget: ') for m in messages if m.get('content', '').startswith('Run budget: ')))
                assert budget['visualization_tool_calls_allowance'] == 3  # Shared multi-line plan also covers native-local work.
                assert budget['completion_actions_reserved'] == 6
            if self.phase == 4:
                feedback = json.loads(next(m['content'] for m in reversed(messages) if m.get('role') == 'tool'))
                assert any('deterministic preparation tool' in issue for issue in feedback['audit_issues'])
                assert feedback['visual_purposes']['relationship']['status'] == feedback['visual_purposes']['sum']['status'] == 'missing'
                for purpose, chart_id in zip(('relationship', 'sum'), self.original['chart_ids'][4:]):
                    issue = next(issue for issue in feedback['audit_issues'] if f'visual purpose: {purpose}.' in issue)
                    assert chart_id in issue and 'Choose and prepare' not in issue
                    assert feedback['ready_visual_purpose_chart_ids'][purpose] == [chart_id]
                budget = json.loads(next(m['content'].removeprefix('Run budget: ') for m in messages if m.get('content', '').startswith('Run budget: ')))
                assert budget['visualization_tool_calls_allowance'] == 0
                assert budget['completion_actions_reserved'] == 2
                self.phase += 1
                args = self.original if method == 'finish' else {'draft_id': feedback['draft_id'], 'chart_ids': self.original['chart_ids']}
                return ModelReply(model='synthetic-selection-repair', provider='synthetic', usage=Usage(), calls=[ToolCall(id='selected-finish', name=method, arguments=json.dumps(args))])
            reply = super().complete(messages, tools, max_output_tokens, timeout)
            if self.phase == 4:
                self.original = json.loads(reply.calls[0].arguments)
                raw = {**self.original, 'chart_ids': self.original['chart_ids'][:4] + ['chart_raw_not_prepared']}
                reply = reply.model_copy(update={'calls': [reply.calls[0].model_copy(update={'arguments': json.dumps(raw)})]})
            return reply

    with Store(tmp_path / 'selected.db') as store:
        seed_report_demo(store)
        run = Runner(Settings(), store, model=SelectionModel()).run(Task(objective='Prepare weekly London report', entry_skill='prepare_weekly_briefing', briefing_date=date(2026, 10, 9), report_week_start=date(2026, 10, 5)))
        assert run.status == 'partial'
        traces = store.list_traces(run.id)
        assert [t.outcome for t in traces if t.name == 'finish'] == ['failed', 'success']
        rejected = next(i for i, t in enumerate(traces) if t.name == 'finish' and t.outcome == 'failed')
        assert not any(t.kind == 'signal' for t in traces[:rejected])
        assert [chart.chart_type for chart in run.result.charts] == ['multi_line', 'multi_line', 'bar', 'bar', 'relation', 'total']
        report_id = store.save_report(run.id, date(2026, 10, 5), run.result)
        saved = next(report for report in store.list_reports('fixture') if report['id'] == report_id)
        assert saved['result'].charts == run.result.charts


def test_plan_invalidated_by_retained_conflict_does_not_abort_budget_or_allow_preparation(tmp_path):
    import json
    from london_market.runner import Runner
    from london_market.providers import ModelReply, ToolCall
    from london_market.domain import Usage

    class ConflictModel:
        phase = 0
        plan_id = None
        def complete(self, messages, tools, max_output_tokens, timeout):
            outputs = [json.loads(m['content']) for m in messages if m.get('role') == 'tool']
            if self.phase == 0:
                pinned = next(line.removeprefix('Pinned weekly context: ') for m in messages for line in (m.get('content') or '').splitlines() if line.startswith('Pinned weekly context: '))
                window = json.loads(pinned)
                actions = [('query_evidence', {'published_since': window['window_start'], 'published_until': window['window_end']}), ('choose_visualization', {'observation_ids': [item.id for item in history], 'title': 'Synthetic history', 'purpose': 'Dated vacancy trajectory', 'preferred': 'trend'})]
            elif self.phase == 1:
                self.plan_id = outputs[-1]['plan_id']
                item = history[-1]
                conflict = Observation(**{**item.model_dump(exclude={'id'}), 'value': 99})
                source = store.get_evidence(item.evidence_ids[0], 'fixture')
                store.ingest(store.get_document(source.document_id, 'fixture'), [], [conflict])
                actions = [('query_evidence', {'limit': 12})]
            elif self.phase == 2:
                budget = json.loads(next(m['content'].removeprefix('Run budget: ') for m in messages if m.get('content', '').startswith('Run budget: ')))
                assert budget['visual_purposes']['trend']['status'] == 'unsupported'
                assert budget['visualization_tool_calls_allowance'] == 0
                assert budget['completion_actions_reserved'] == 2
                actions = [('prepare_visualization_data', {'plan_id': self.plan_id})]
            else:
                assert outputs[-1]['error'] == 'unresolved conflicting values for the same measurement and period'
                actions = [('finish', {'response_type': 'insufficient_data', 'coverage': scripted_coverage([], [item.model_dump(mode='json') for item in store.list_observations('fixture')], [], evidence=[item.model_dump(mode='json') for item in store.list_evidence('fixture')])})]
            self.phase += 1
            return ModelReply(model='synthetic-conflict', provider='synthetic', usage=Usage(), calls=[ToolCall(id=f'conflict-{self.phase}-{i}', name=name, arguments=json.dumps(args)) for i, (name, args) in enumerate(actions)])

    with Store(tmp_path / 'conflict.db') as store:
        load_fixtures(store)
        history = sorted([item for item in store.list_observations('fixture') if item.metric == 'vacancy_rate' and item.mapped_geography == 'City'], key=lambda item: item.period_start)
        model = ConflictModel()
        run = Runner(Settings(), store, model=model).run(Task(objective='Prepare weekly London report', entry_skill='prepare_weekly_briefing', briefing_date=date(2026, 10, 9), report_week_start=date(2026, 10, 5)))
        assert run.status == 'partial', run.result.reason
        assert model.phase == 4
        assert run.result.charts == []
        assert any(t.name == 'prepare_visualization_data' and t.outcome == 'failed' for t in store.list_traces(run.id))
