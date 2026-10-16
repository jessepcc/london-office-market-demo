from pathlib import Path
import json

from streamlit.testing.v1 import AppTest

from london_market.domain import ChartPair, ChartPayload, ChartPoint, RunResult, TraceEvent
from london_market.fixtures import fixture_batches
from london_market.report import chart_figure, export_report_html, observation_scope, report_key_figures, source_catalogue_rows
from london_market.sources import MONITORED_SOURCES
from london_market.store import Store


def point(observation, **changes):
    fields = dict(label=observation.mapped_geography, value=observation.value, observation_id=observation.id, evidence_ids=observation.evidence_ids, period_start=observation.period_start, period_end=observation.period_end, geography=observation.mapped_geography, metric=observation.metric, unit=observation.unit, publisher=observation.publisher, definition=observation.definition, category=observation.category)
    return ChartPoint(**(fields | changes))


def test_relation_retains_independent_axes_and_source_ids_in_offline_export():
    _, sources, observations = fixture_batches()[0]
    x = point(observations[0], value=110, unit="GBP per sq ft", metric="prime_rent", definition="Transaction rent")
    y = point(observations[0], value=8, observation_id="vacancy-observation", unit="%", definition="Unoccupied office stock")
    chart = ChartPayload(chart_type="relation", title="Rent and vacancy", purpose="Same-period relationship; no causality established", pairs=[ChartPair(x=x, y=y)], x_label="Prime rent", y_label="Vacancy rate", x_unit=x.unit, y_unit=y.unit, source_refs=x.evidence_ids, caveats=["Association does not establish causality."])
    figure = chart_figure(chart)
    assert list(figure.data[0].x) == [110] and list(figure.data[0].y) == [8]
    assert figure.layout.yaxis.title.text == "Vacancy rate (%)"
    assert x.observation_id in figure.data[0].customdata[0] and y.observation_id in figure.data[0].customdata[0]
    exported = export_report_html(RunResult(status="partial", answer="", reason="test", charts=[chart]), sources, [])
    assert "Plotly.newPlot" in exported and '<script src="https://' not in exported
    assert "Association does not establish causality." in exported
    assert "vacancy-observation" in exported and 'href="#evidence-' in exported
    dark = chart_figure(chart, theme="dark")
    assert list(dark.data[0].x) == [110] and list(dark.data[0].y) == [8]
    assert dark.data[0].customdata == figure.data[0].customdata
    assert dark.layout.paper_bgcolor == "#172b3e"


def test_source_reported_segment_share_stays_a_bar_without_invented_remainder():
    _, _, observations = fixture_batches()[0]
    share = point(observations[0], value=92, segment="Grade A", dimension="grade", metric="take_up_share")
    chart = ChartPayload(chart_type="proportion", title="Reported Grade A take-up share", purpose="Source-reported share", unit="%", points=[share])
    figure = chart_figure(chart)
    assert figure.data[0].type == "bar"
    assert list(figure.data[0].y) == [92]
    assert list(figure.data[0].x) == ["City · Grade A"]


def test_dated_total_bars_keep_quarters_values_and_observation_ids_distinct(tmp_path):
    import hashlib
    import json
    import re
    from london_market.config import Settings
    from london_market.domain import Document
    from london_market.providers import FetchedPage
    from london_market.sources import parse_monitored_source
    from london_market.tools import PrepareArgs, ToolContext, VisualizationArgs, choose_visualization, prepare_visualization_data

    with Store(tmp_path / "market.sqlite3") as store:
        observations = []
        for quarter in ("q1", "q2"):
            source_id = f"savills-central-london-{quarter}-2026"
            source = MONITORED_SOURCES[source_id]
            text = (Path(__file__).parent / "fixtures" / "live_sources" / f"{quarter}.txt").read_text()
            document = Document(mode="live", url=source.url, publisher="www.savills.com", title="Reviewed source fixture", content_hash=hashlib.sha256(text.encode()).hexdigest())
            parsed = parse_monitored_source(source_id, FetchedPage(document=document, text=text))
            store.ingest(parsed.document, parsed.evidence, parsed.observations)
            observations.append(next(item for item in parsed.observations if item.metric == "reported_supply"))
        ctx = ToolContext(Settings(), store, None, "live", "dated-total-render-test", 5)
        plan = choose_visualization(ctx, VisualizationArgs(observation_ids=[item.id for item in reversed(observations)], preferred="total", title="Central London reported supply", purpose="Inspect reported quarter-end supply"))
        payload = prepare_visualization_data(ctx, PrepareArgs(plan_id=plan.plan_id)).payload
        assert payload.chart_type == "total"
        figure = chart_figure(payload)
        assert list(figure.data[0].x) == ["Central London · Q1 2026", "Central London · Q2 2026"]
        assert list(figure.data[0].y) == [19.1, 18.7]
        assert [row[0] for row in figure.data[0].customdata] == [item.id for item in observations]
        exported = export_report_html(RunResult(status="partial", answer="", reason="Test", charts=[payload]), store.list_evidence("live"), observations)
        exported_figure = json.loads(re.search(r"\{const f=(.*?);Plotly\.newPlot", exported).group(1))
        assert exported_figure["data"][0]["x"] == ["Central London · Q1 2026", "Central London · Q2 2026"]


def test_scoped_key_figure_does_not_disappear_with_another_publisher():
    _, sources, observations = fixture_batches()[0]
    first = observations[0]
    second = first.model_copy(update={"id": "independent-publisher", "publisher": "Other publisher", "value": 9})
    result = RunResult(status="partial", answer="", reason="test", observation_ids=[first.id, second.id])
    figures = report_key_figures(result, sources, [first, second])
    assert len(figures) == 1 and figures[0][0].value in (first.value, second.value)
    exported = export_report_html(result, sources, [first, second])
    assert "Vacancy rate · City" in exported and "Central London</p>" not in exported


def test_catalogue_retains_secondary_role_and_district_identity():
    _, sources, observations = fixture_batches()[0]
    secondary = next(source for source in MONITORED_SOURCES.values() if source.source_kind == "secondary_news")
    source = sources[0].model_copy(update={"source_url": secondary.url, "mode": "live"})
    row = source_catalogue_rows([source], [])[0]
    assert row["Publisher"] == "City AM" and row["Role"] == "Secondary news"
    assert row["Latest measured period"] == "Qualitative only" and row["Measurements"] == 0
    district = observations[0].model_copy(update={"native_geography": "Hammersmith", "mapped_geography": "Wider London"})
    assert observation_scope(district) == "Hammersmith (Wider London)"


def test_prompt_starts_research_and_saved_log_matches_real_run(tmp_path, monkeypatch):
    database = tmp_path / "market.sqlite3"
    for name, value in {"LONDON_MARKET_DB": str(database), "LONDON_MARKET_MODE": "fixture", "OPENROUTER_API_KEY": "", "TAVILY_API_KEY": ""}.items():
        monkeypatch.setenv(name, value)
    from ui_support import app_test
    app = app_test(database)
    assert any(item.value == "Chat over London offices" for item in app.subheader)
    assert sum(item.label == "Ask this question" for item in app.button) == 3
    app.button(key="prompt-fixture-Leasing conditions").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert any(item.label == "Chat log" for item in app.expander)
    with Store(database) as store:
        runs = store.list_runs("fixture")
        traces = store.list_traces(runs[0].id)
    assert len(runs) == 1 and any(event.kind == "tool" and event.outcome == "success" for event in traces)
    assert runs[0].task.objective == "What do the latest rents and vacancy evidence show across London submarkets?"
    assert runs[0].task.trigger == "chat" and runs[0].task.entry_skill == "research_market"
    assert any(runs[0].id in item.value for item in app.caption)
    assert any("research_market" == event.name for event in traces)
    assert any('Research pipeline' in item.value for item in app.markdown)
    assert any(item.label == 'Runtime skills' for item in app.expander)
    app.button(key='open-run-history').click().run()
    assert not app.exception
    assert any('Research pipeline' in item.value for item in app.markdown)
    assert any('Evidence analysis' in item.value and 'returned' in item.value for item in app.markdown)
    assert any('Inspect stored evidence' in item.value for item in app.caption)
    assert any(item.label == 'Observable actions and usage' for item in app.expander)




def test_pipeline_counts_terminal_outcomes_and_keeps_failed_checks_visible():
    from london_market.presentation import research_pipeline_rows, runtime_skill_rows
    traces = [
        TraceEvent(kind='skill', name='research_market', outcome='loaded', data={'version': '6'}),
        TraceEvent(kind='model', name='main-model', outcome='reserved'),
        TraceEvent(kind='progress', name='search', outcome='started'),
        TraceEvent(kind='tool', name='search', outcome='success'),
        TraceEvent(kind='tool', name='retrieve', outcome='failed'),
        TraceEvent(kind='tool', name='collect_monitored_source', outcome='failed', data={'error': 'Source access blocked'}),
        TraceEvent(kind='signal', name='classifier', outcome='reserved'),
        TraceEvent(kind='signal', name='batch_plan', outcome='prepared'),
        TraceEvent(kind='signal', name='classifier', outcome='classified'),
        TraceEvent(kind='signal', name='classifier', outcome='unavailable'),
    ]
    assert research_pipeline_rows(traces) == [
        {'Work': 'Source research', 'Recorded actions': 'Search market sources (1); Retrieve source content (1)', 'Recorded outcomes': '1 returned; 1 failed'},
        {'Work': 'Source checks', 'Recorded actions': 'Check a monitoring source (1)', 'Recorded outcomes': '1 failed'},
        {'Work': 'Finalization', 'Recorded actions': 'Assess factual leasing signals (2)', 'Recorded outcomes': '1 classified; 1 unavailable'},
    ]
    assert runtime_skill_rows(traces) == [{'Skill': 'Research market evidence', 'Version': '6'}]
    assert research_pipeline_rows([]) == [] and runtime_skill_rows([]) == []


def test_export_display_terms_preserve_unknown_dates_and_original_evidence():
    _, sources, observations = fixture_batches()[0]
    source = sources[0].model_copy(update={"published_at": None, "event_date": None, "validation_status": "candidate", "validation_reason": "Original extraction qualification."})
    exported = export_report_html(RunResult(status="partial", answer="", reason="Original run reason", observation_ids=[observations[0].id]), [source], observations)
    assert "Completed with gaps" in exported and "Awaiting monitoring review" in exported
    assert "Publication date Unknown" in exported and "Event date Unknown" in exported
    assert "Coverage grouping City" in exported and "Vacancy rate" in exported
    assert "Original extraction qualification." in exported
    assert source.excerpt in exported and observations[0].definition in exported
    assert source.id in exported and observations[0].id in exported
    assert 'class="eyebrow"' not in exported and "border-left:3px" not in exported
