"""Render the saved report and deterministic exhibits for UI and offline export."""

from __future__ import annotations

import html
import json
from calendar import monthrange
from collections import defaultdict
from urllib.parse import urlparse

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from .coverage import report_coverage_rows
from .domain import ChartPayload, Evidence, Observation, RunResult, SECTION_TITLES, SUBMARKETS, stable_id
from .sources import MONITORED_SOURCES
from .weekly import saved_claim_source_ids
from .presentation import CLAIM_LABELS as LABELS, MONITORING_LABELS, SOURCE_CHECK_LABELS, STATUS_LABELS, claim_label, claim_metadata, market_evidence_groups, measure_label, partition_market_evidence, weekly_indicator_groups, weekly_summary_copy

COLORS = {"City": "#173a59", "West End": "#168582", "Canary Wharf": "#bb782c", "Midtown and Fringe": "#7c65a3", "Wider London": "#65834d", "Central London": "#168582", "Whole London": "#173a59"}


def chart_reference_id(chart: ChartPayload) -> str:
    return stable_id("chart", chart.model_dump(mode="json"))


def source_catalogue_rows(evidence: list[Evidence], observations: list[Observation]) -> list[dict]:
    contracts = {source.url.rstrip("/"): source for source in MONITORED_SOURCES.values()}
    documents = defaultdict(list)
    for item in evidence:
        documents[str(item.source_url)].append(item)
    rows = []
    roles = {"primary_research": "Primary research", "primary_research_commentary": "Research commentary", "primary_news": "Primary announcement", "official_statistics": "Primary official statistics", "official_analysis": "Primary official analysis", "secondary_news": "Secondary news"}
    for url, records in documents.items():
        contract = contracts.get(url.rstrip("/"))
        ids = {item.id for item in records}
        measured = [item for item in observations if ids.intersection(item.evidence_ids)]
        rows.append({"Publisher": contract.publisher if contract else urlparse(url).hostname or "Unknown publisher", "Role": roles.get(contract.source_kind, contract.source_kind) if contract else "Synthetic demonstration" if records[0].mode == "fixture" else "Candidate research", "Topics": ", ".join(sorted({measure_label(item.category) for item in records})), "Source geography": ", ".join(sorted({item.native_geography for item in records})), "Coverage grouping": ", ".join(sorted({item.mapped_geography for item in records})), "Publication date": str(max((item.published_at for item in records if item.published_at), default="Unknown")), "Latest measured period": str(max((item.period_end for item in measured), default="Qualitative only")), "Measurements": len(measured), "Original publication": url})
    return rows


def observation_scope(item: Observation) -> str:
    return f"{item.native_geography} ({item.mapped_geography})" if item.native_geography != item.mapped_geography else item.mapped_geography

REPORT_CSS = """
:root{--ink:#173a59;--teal:#116b68;--muted:#566975;--rule:#dce5ed}
*{box-sizing:border-box}body{font:14px/1.5 Arial,Helvetica,sans-serif;color:var(--ink);max-width:1240px;margin:0 auto;padding:0 36px;background:#f7f9fc}
header{padding:32px 0 24px;border-bottom:1px solid var(--teal)}
h1{font-size:36px;line-height:1.15;letter-spacing:-.035em;margin:12px 0}h2{font-size:23px;line-height:1.25;margin:0 0 18px}h3{font-size:15px;margin:0 0 12px}
.metadata,.scope{color:var(--muted);font-size:12px}.status{color:var(--teal);font-weight:700}nav{display:flex;gap:9px 20px;flex-wrap:wrap;padding:18px 0;font-size:12px;border-bottom:1px solid var(--rule)}
a{color:#116b78;text-decoration:none}a:hover{text-decoration:underline}small{font-size:11px;display:block;color:var(--muted);overflow-wrap:anywhere}
.key-figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin:24px 0}.key-figure{background:#fff;border:1px solid var(--rule);border-radius:12px;padding:18px}.key-value{font-size:32px;font-weight:700;line-height:1.15}.key-figure p{font-size:12px;margin:9px 0}.key-figure small{line-height:1.6}
section{margin:30px 0}section+section{border-top:1px solid var(--rule);padding-top:24px}.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px}
.claim{background:#fff;border:1px solid var(--rule);border-radius:10px;padding:18px;margin:0}.claim p{margin:5px 0}.claim h3{margin:8px 0}.claim-label{font-size:12px;font-weight:700;color:var(--teal)}.citations{display:inline;font-size:11px;white-space:nowrap;margin-left:4px}
.claim .timing{margin-top:12px;line-height:1.6}.topic-tags{font-size:12px;color:var(--muted);margin-top:10px}.signal-heading{margin:24px 0 10px;font-size:18px}.signal-bullish{color:#17633e}.signal-bearish{color:#963535}.signal-neutral,.signal-unavailable{color:var(--muted)}.findings{grid-template-columns:repeat(2,minmax(0,1fr))}.week-window{font-size:16px;margin:10px 0}.weekly-indicators{overflow-x:auto}.weekly-indicators table{min-width:700px}.context-figures{margin:20px 0}.coverage-detail{margin:22px 0}
.coverage-lane{padding:12px 0;border-bottom:1px solid var(--rule);max-width:90ch}.coverage-lane details{margin:0}.coverage-lane summary{font-size:14px;line-height:1.55;color:var(--ink)}.coverage-lane p{margin:7px 0;font-size:12px}.coverage-gap{color:var(--muted)}.coverage-links a{margin-right:8px}.exhibits{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px;margin-bottom:18px}figure{background:#fff;border:1px solid var(--rule);border-radius:12px;padding:18px;margin:0;min-width:0}figure p{font-size:12px;color:var(--muted);margin:0 0 12px}figcaption{font-size:11px;color:var(--muted);padding-top:8px}.chart{width:100%}details{margin-top:10px}summary{cursor:pointer;color:var(--teal);font-size:12px}a:focus-visible,summary:focus-visible{outline:2px solid var(--teal);outline-offset:3px}
blockquote{font-size:12px;border-left:1px solid var(--rule);padding-left:16px;margin:14px 0}article{background:#fff;padding:18px;border:1px solid var(--rule);border-radius:8px;margin-bottom:12px;overflow-wrap:anywhere}article p{font-size:12px}.synthetic{background:#fff0c9;padding:14px;border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid var(--rule);padding:8px;text-align:left;vertical-align:top}th{background:#eaf1f1}footer{border-top:2px solid var(--teal);padding:20px 0 32px;font-size:11px;color:var(--muted)}
@media(max-width:700px){body{padding:0 20px}h1{font-size:30px}.exhibits,.findings{display:block}.claim{margin-bottom:14px}figure{margin-bottom:14px}}
@media print{body{background:#fff;font-size:10pt;padding:0;max-width:none}header{padding:12px 0 18px}h1{font-size:28pt}h2{font-size:18pt}nav{display:none}section{margin:24px 0}section+section{padding-top:16px}.exhibits,.findings{display:block}figure{break-inside:avoid;margin:14px 0}h2,h3{break-after:avoid}.claim{break-inside:avoid;margin-bottom:10px}.key-figures{gap:12px}.key-figure{padding:12px}#source-appendix{break-before:page}article{break-inside:avoid}footer{display:none}}
"""


def coverage_html(rows: list[dict], dimension: str, *, prefix: str = "", chart_indexes: dict[str, int] | None = None, link_findings: bool = True) -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    parts = []
    chart_indexes = chart_indexes or {}
    for row in rows:
        if row["dimension"] != dimension:
            continue
        finding = row["findings"][0] if row["findings"] else "No supported finding included."
        if row["candidate"]:
            finding = "Candidate source awaiting monitoring validation. " + finding
        native = ", ".join(row["native_geographies"])
        chronology = []
        if row["periods"]:
            chronology.append("Measured period " + "; ".join(row["periods"]))
        if row["publication_dates"]:
            chronology.append("Published " + ", ".join(row["publication_dates"]))
        elif row["evidence_ids"]:
            chronology.append("Publication date unknown")
        if row["event_dates"]:
            chronology.append("Event " + ", ".join(row["event_dates"]))
        label = {"measurement": "Measured", "qualitative": "Qualitative", "context": "Broader context", "gap": "Gap"}[row["basis"]]
        if row["candidate"]:
            label += " · Candidate"
        links = [f'<a href="#{escape(prefix)}claim-{index}">Finding {index + 1}</a>' for index in row["claim_indexes"]] if link_findings else []
        links += [f'<a href="#{escape(prefix)}exhibit-{chart_indexes[cid]}">Exhibit {chart_indexes[cid] + 1}</a>' for cid in row["chart_ids"] if cid in chart_indexes]
        links += [f'<a href="{escape(url)}" target="_blank" rel="noreferrer">Original source</a>' for url in row["source_urls"]]
        parts.append(f'<div class="coverage-lane"><details><summary><strong>{escape(row["label"])}</strong> · {escape(label)}</summary>')
        parts.append(f'<p>{escape(finding)}</p>')
        for additional in row["findings"][1:]:
            parts.append(f'<p>{escape(additional)}</p>')
        parts.append('<p class="coverage-links">' + ' · '.join(links) + '</p>')
        parts.append('<small>' + escape(label + (" · " + native if native else "")) + '<br>' + escape(" · ".join(chronology)) + '</small>')
        parts.append('</details>')
        if row["gaps"]:
            parts.append('<p class="coverage-gap">' + escape("Gap: " + " ".join(row["gaps"])) + '</p>')
        if native or chronology:
            parts.append('<small>' + escape(" · ".join([native, *chronology]).strip(" ·")) + '</small>')
        parts.append('</div>')
    return ''.join(parts)


def report_groups(result: RunResult):
    if result.coverage is not None and result.weekly is None:
        yield "coverage_topics", "Topics", [], []
        yield "coverage_areas", "Across London", [], []
    if result.weekly is not None:
        if result.charts:
            yield "exhibits", "Market exhibits", [], list(range(len(result.charts)))
        for key, title, kind, role in (("implications", "Landlord analysis", "interpretation", "landlord_analysis"), ("market_facts", "Market evidence", "fact", None), ("source_forecasts", "Source forecasts", "forecast", None), ("source_views", "Source views", "interpretation", "source_view"), ("interpretations", "Interpretation", "interpretation", None), ("watchpoints", "What to watch", "watchpoint", None)):
            claims = [index for index, claim in enumerate(result.claims) if claim.kind == kind and (kind != "interpretation" or claim.interpretation_role == role)]
            if claims:
                yield key, title, claims, []
        if result.coverage is not None:
            yield "coverage_topics", "Topics", [], []
            yield "coverage_areas", "Across London", [], []
        return
    if not result.sections and result.briefing_date or any(claim.interpretation_role for claim in result.claims):
        for key, title, kind, role in (("market_facts", "Market evidence", "fact", None), ("source_forecasts", "Source forecasts", "forecast", None), ("source_views", "Source views", "interpretation", "source_view"), ("implications", "Landlord analysis", "interpretation", "landlord_analysis"), ("interpretations", "Interpretation", "interpretation", None), ("watchpoints", "What to watch", "watchpoint", None)):
            claims = [index for index, claim in enumerate(result.claims) if claim.kind == kind and (kind != "interpretation" or claim.interpretation_role == role)]
            if claims:
                yield key, title, claims, []
            if kind == "fact" and result.charts:
                yield "exhibits", "Market exhibits", [], list(range(len(result.charts)))
        return
    used_claims, used_charts = set(), set()
    for section in result.sections:
        claims = [index for index in section.claim_indexes if index not in used_claims]
        charts = [index for index in section.chart_indexes if index not in used_charts]
        used_claims.update(claims)
        used_charts.update(charts)
        if claims or charts:
            yield section.kind, SECTION_TITLES[section.kind], claims, charts
    claims = [index for index in range(len(result.claims)) if index not in used_claims]
    charts = [index for index in range(len(result.charts)) if index not in used_charts]
    if claims or charts:
        yield "additional_findings", "Additional findings" if result.sections else "Findings", claims, charts


def report_key_figures(result: RunResult, evidence: list[Evidence], observations: list[Observation]) -> list[tuple[Observation, str]]:
    source_ids = {source.id for source in evidence}
    observation_ids = set(result.observation_ids) | {point.observation_id for chart in result.charts for point in chart_points(chart)}
    figures = []
    for metric, label in (("take_up", "Office take-up"), ("vacancy_rate", "Vacancy rate"), ("reported_supply", "Source-reported supply"), ("prime_rent", "Prime rent"), ("grade_a_rent", "Grade A rent")):
        groups = defaultdict(list)
        for item in observations:
            if item.id in observation_ids and item.metric == metric and all(eid in source_ids for eid in item.evidence_ids):
                groups[(item.mapped_geography, item.native_geography, item.publisher, item.definition, item.unit, item.category)].append(item)
        latest_values = []
        for candidates in groups.values():
            latest = max(candidates, key=lambda item: (item.period_end, item.period_start))
            same_period = [item for item in candidates if (item.period_start, item.period_end) == (latest.period_start, latest.period_end)]
            if len({item.value for item in same_period}) == 1:
                latest_values.append(latest)
        latest_values.sort(key=lambda item: (item.mapped_geography not in ("Central London", "Whole London"), -item.period_end.toordinal(), item.publisher, item.mapped_geography))
        if latest_values:
            figures.append((latest_values[0], label))
    return figures[:4]


def chart_points(chart: ChartPayload):
    points = list(chart.points)
    points.extend(point for pair in chart.pairs for point in (pair.x, pair.y))
    if chart.denominator is not None:
        points.append(chart.denominator)
    return list({point.observation_id: point for point in points}.values())


def chart_geography(point) -> str:
    native = point.native_geography or point.geography
    return f"{native} ({point.geography})" if native != point.geography else native


def chart_period(point) -> str:
    start, end = point.period_start, point.period_end
    if start.day == 1 and start.month in (1, 4, 7, 10) and end.year == start.year and end.month == start.month + 2 and end.day == monthrange(end.year, end.month)[1]:
        return f"Q{(start.month - 1) // 3 + 1} {start.year}"
    return f"{start} to {end}"


def displayed_flow_total(chart: ChartPayload) -> str | None:
    from .tools import displayed_flow_sum
    total = displayed_flow_sum(chart.points) if chart.chart_type == "total" else None
    return f"Calculated sum of displayed periods: {total:,.8g} {chart.points[0].unit}" if total is not None else None


def chart_figure(chart: ChartPayload, *, theme: str = "light") -> go.Figure:
    groups = defaultdict(list)
    for point in chart.points:
        groups[chart_geography(point)].append(point)
    figure = go.Figure()
    x_label = measure_label(chart.x_label)
    y_label = measure_label(chart.y_label)
    if chart.chart_type in ("line", "multi_line"):
        for geography in sorted(groups, key=lambda value: SUBMARKETS.index(value) if value in SUBMARKETS else len(SUBMARKETS)):
            points = groups[geography]
            points.sort(key=lambda point: point.period_start)
            figure.add_trace(go.Scatter(x=[point.period_end for point in points], y=[point.value for point in points], name=html.escape(geography), mode="lines+markers", connectgaps=False, line={"color": COLORS.get(points[0].geography, "#697780"), "width": 3}, marker={"size": 8}, customdata=[[point.observation_id, point.publisher, point.definition, f"{point.period_start} to {point.period_end}"] for point in points], hovertemplate="%{y:,.2~f} " + html.escape(chart.unit) + "<br>%{customdata[3]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[0]}<extra>%{fullData.name}</extra>"))
    elif chart.chart_type in ("bar", "total", "proportion"):
        multiple_periods = len({(point.period_start, point.period_end) for point in chart.points}) > 1
        points = sorted(chart.points, key=lambda point: ((point.period_start, point.period_end) if multiple_periods else (), SUBMARKETS.index(point.geography) if point.geography in SUBMARKETS else len(SUBMARKETS), chart_geography(point), point.segment or ""))
        labels = [html.escape(f"{chart_geography(point)} · {point.segment}" if point.segment else chart_geography(point)) for point in points]
        if multiple_periods:
            labels = [f"{label} · {chart_period(point)}" for label, point in zip(labels, points, strict=True)]
        figure.add_trace(go.Bar(x=labels, y=[point.value for point in points], marker_color=[COLORS.get(point.geography, "#697780") for point in points], name=html.escape(chart.unit), customdata=[[point.observation_id, point.publisher, point.definition, f"{point.period_start} to {point.period_end}"] for point in points], hovertemplate="%{y:,.2~f} " + html.escape(chart.unit) + "<br>%{customdata[3]}<br>%{customdata[1]}<br>%{customdata[2]}<br>%{customdata[0]}<extra>%{x}</extra>"))
    elif chart.chart_type == "relation":
        for pair in chart.pairs:
            label = f"{chart_geography(pair.x)} · {pair.x.segment}" if pair.x.segment else chart_geography(pair.x)
            label += f" · {chart_period(pair.x)}"
            figure.add_trace(go.Scatter(x=[pair.x.value], y=[pair.y.value], mode="markers", name=html.escape(label), marker={"size": 13, "color": COLORS.get(pair.x.geography, "#697780")}, customdata=[[pair.x.observation_id, pair.y.observation_id, pair.x.publisher, f"{pair.x.period_start} to {pair.x.period_end}", pair.x.definition, pair.y.definition]], hovertemplate=html.escape(x_label) + ": %{x:,.2~f} " + html.escape(chart.x_unit) + "<br>" + html.escape(y_label) + ": %{y:,.2~f} " + html.escape(chart.y_unit) + "<br>%{customdata[3]}<br>%{customdata[2]}<br>%{customdata[4]}<br>%{customdata[5]}<br>%{customdata[0]} / %{customdata[1]}<extra>%{fullData.name}</extra>"))
    else:
        figure.add_trace(go.Table(header={"values": ["Submarket", "Measure", "Reference period", "Value", "Unit"]}, cells={"values": [[html.escape(point.geography) for point in chart.points], [html.escape(measure_label(point.metric)) for point in chart.points], [f"{point.period_start} to {point.period_end}" for point in chart.points], [point.value for point in chart.points], [html.escape(point.unit) for point in chart.points]]}))
    figure.update_layout(xaxis_title=html.escape(f"{x_label} ({chart.x_unit})") if chart.chart_type == "relation" else None, yaxis_title=html.escape(f"{y_label} ({chart.y_unit})") if chart.chart_type == "relation" else html.escape(chart.unit), margin=dict(l=55, r=20, t=45, b=55), height=360, template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", font=dict(family="Arial, sans-serif", size=12, color="#344c58"), legend=dict(orientation="h", y=1.18, x=0, font={"size": 11}), hovermode="closest" if chart.chart_type == "relation" else "x unified")
    if chart.chart_type != "table":
        figure.update_yaxes(gridcolor="#dce5e8", zerolinecolor="#94aab3", rangemode="tozero")
        figure.update_xaxes(showgrid=False, linecolor="#b8c9cf")
    if chart.chart_type in ("line", "multi_line"):
        periods = sorted({point.period_end for point in chart.points})
        quarterly = all(point.period_start.day == 1 and point.period_start.month in (1, 4, 7, 10) and (point.period_end - point.period_start).days in (89, 90, 91) for point in chart.points)
        figure.update_xaxes(tickmode="array", tickvals=periods, ticktext=[f"Q{(period.month - 1) // 3 + 1} {period.year}" if quarterly else period.strftime("%b %Y") for period in periods])
    if chart.chart_type in ("bar", "total", "proportion"):
        figure.update_traces(texttemplate="%{y:,.2~f}", textposition="outside", cliponaxis=False)
    if theme == "dark":
        figure.update_layout(template="plotly_dark", paper_bgcolor="#172b3e", plot_bgcolor="#172b3e", font_color="#edf3f7")
        figure.update_yaxes(gridcolor="#3b5266", zerolinecolor="#bdccd7")
        figure.update_xaxes(linecolor="#3b5266")
        dark_colors = {"#173a59": "#86bde5", "#168582": "#79d1cb", "#bb782c": "#e7b46f", "#7c65a3": "#bba8df", "#65834d": "#acd091", "#697780": "#bdccd7"}
        for trace in figure.data:
            if trace.type == "scatter":
                trace.line.color = dark_colors.get(trace.line.color, trace.line.color)
                trace.marker.color = dark_colors.get(trace.marker.color, trace.marker.color)
            elif trace.type == "bar":
                trace.marker.color = [dark_colors.get(color, color) for color in trace.marker.color]
    return figure


def export_report_html(result: RunResult, evidence: list[Evidence], observations: list[Observation], *, title: str = "London office market briefing") -> str:
    escape = lambda value: html.escape(str(value), quote=True)
    source_numbers = {source.id: index + 1 for index, source in enumerate(evidence)}
    def citation(evidence_id):
        if evidence_id not in source_numbers:
            return f'<span>Unavailable source {escape(evidence_id)}</span>'
        return f'<a href="#evidence-{escape(evidence_id)}">[{source_numbers[evidence_id]}]</a>'
    periods = sorted({(item.period_start, item.period_end) for item in observations})
    period_label = f'{periods[0][0]} to {max(end for _, end in periods)}' if periods else "No reviewed measurement period"
    scope = f'Evidence reference periods {period_label}. {len(set(str(item.source_url) for item in evidence))} source publications.'
    header_status = '' if result.weekly else f' &nbsp; / &nbsp; <span class="status">{escape(STATUS_LABELS[result.status])}</span>'
    header_scope = '' if result.weekly else f'<p class="scope">{escape(scope)} Read the dated findings as market context for leasing discussions. Coverage gaps remain explicit.</p>'
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">', f'<title>{escape(title)}</title>', '<style>' + REPORT_CSS + '</style></head><body>', '<header>', f'<h1>{escape(title)}</h1>', f'<p class="metadata">Briefing date {escape(result.briefing_date or "Unknown")}{header_status}</p>{header_scope}</header>']
    if any(item.mode == "fixture" for item in evidence) or "SYNTHETIC" in result.answer:
        parts.append('<p class="synthetic">SYNTHETIC DEMONSTRATION DATA. Fictional observations and named examples; scripted model behavior. These are not live market findings.</p>')
    if "Editor-reviewed:" in result.reason:
        parts.append('<p class="status">Editor-reviewed briefing. Run history retains the original draft and documented corrections.</p>')
    key_figures = report_key_figures(result, evidence, observations)
    if result.weekly:
        copy = weekly_summary_copy(result)
        parts.append(f'<section id="this-week"><h2>This week</h2><p class="week-window">{escape(copy["window"])}</p><p class="metadata">{escape(copy["cutoff"])}<br>{escape(copy["status"])}<br>{escape(copy["measurements"])}</p>')
        if result.weekly.week_over_week:
            parts.append('<h3>Week-over-week indicators</h3>')
        for heading, rows in weekly_indicator_groups(result, observations):
            parts.append(f'<h3>{escape(heading)}</h3><div class="weekly-indicators"><table><thead><tr>' + ''.join(f'<th>{escape(field)}</th>' for field in rows[0]) + '</tr></thead><tbody>')
            for row in rows:
                parts.append('<tr>' + ''.join(f'<td>{escape(value)}</td>' for value in row.values()) + '</tr>')
            parts.append('</tbody></table></div>')
        parts.append('</section>')
    if key_figures:
        if result.weekly:
            parts.append('<details class="context-figures"><summary>Market context</summary><p>Latest supported measurements at their source reference periods. These are not weekly changes.</p>')
        parts.append('<div class="key-figures" aria-label="Dated scoped key figures">')
        for item, label in key_figures:
            parts.append(f'<div class="key-figure"><div class="key-value">{item.value:g}<small>{escape(item.unit)}</small></div><p>{escape(label)} · {escape(observation_scope(item))}</p><small>{item.period_start} to {item.period_end}<br>{escape(item.publisher)} ' + ' '.join(citation(eid) for eid in item.evidence_ids) + f'<br>{escape(item.definition)}</small></div>')
        parts.append('</div>')
        if result.weekly:
            parts.append('</details>')
    cited_ids = set(result.evidence_ids) | {eid for claim in result.claims for eid in claim.evidence_ids} | {eid for chart in result.charts for eid in chart.source_refs}
    missing_ids = sorted(cited_ids - source_numbers.keys())
    if missing_ids:
        parts.append('<p role="alert" class="synthetic">Some cited sources are unavailable. Treat affected claims as unsupported and review the missing-source manifest below.</p>')
    groups = list(report_groups(result))
    chart_indexes_by_id = {chart_reference_id(chart): index for index, chart in enumerate(result.charts)}
    parts.append('<nav aria-label="Report sections">' + ''.join(f'<a href="#{kind}">{escape(heading)}</a>' for kind, heading, _, _ in groups) + '<a href="#source-appendix">Source appendix</a></nav>')
    scripts = []
    def claim_html(index):
        claim = result.claims[index]
        citations = ' '.join(citation(eid) for eid in claim.evidence_ids)
        citations += ' ' + ' '.join(f'<a href="#exhibit-{chart_indexes_by_id[chart_id]}">Exhibit {chart_indexes_by_id[chart_id] + 1}</a>' for chart_id in claim.chart_ids if chart_id in chart_indexes_by_id)
        unsupported = claim.kind in ("fact", "forecast") and any(eid not in source_numbers for eid in saved_claim_source_ids(claim, result))
        label = "Unsupported " + claim.kind if unsupported else claim_label(claim)
        if claim.kind == "fact" and result.weekly is None:
            label += " · Signal unavailable" if unsupported or claim.signal is None else " · " + claim.signal.capitalize()
        headline = f'<h3>{escape(claim.headline)}</h3>' if claim.headline else ''
        metadata = ' · '.join(claim_metadata(claim, result, evidence, observations))
        tags = ' '.join('#' + tag for tag in claim.topic_tags)
        return f'<div id="claim-{index}" class="claim {claim.kind}"><span class="claim-label">{escape(label)}</span>{headline}<p>{escape(claim.text)} <span class="citations">{citations}</span></p><small class="timing">{escape(metadata)}</small><div class="topic-tags">{escape(tags)}</div></div>'
    coverage_rows = report_coverage_rows(result, evidence, observations)
    for kind, heading, claim_indexes, chart_indexes in groups:
        parts.append(f'<section id="{kind}"><h2>{escape(heading)}</h2>')
        if kind in ("coverage_topics", "coverage_areas"):
            parts.append(coverage_html(coverage_rows, "topic" if kind == "coverage_topics" else "area", chart_indexes=chart_indexes_by_id) + '</section>')
            continue
        if kind == "market_facts":
            parts.append('<p class="metadata">From a landlord’s leasing perspective</p>')
            for signal, label, indexes in market_evidence_groups(result, set(missing_ids)):
                parts.append(f'<h3 class="signal-heading signal-{signal or "unavailable"}">{escape(label)} · {len(indexes)}</h3>')
                current, retained = partition_market_evidence(result, indexes)
                if current:
                    parts.append('<div class="findings">' + ''.join(claim_html(index) for index in current) + '</div>')
                if retained:
                    parts.append(f'<details class="retained-context"><summary>Retained context · {len(retained)}</summary><div class="findings">' + ''.join(claim_html(index) for index in retained) + '</div></details>')
                if not indexes:
                    parts.append('<p class="metadata">No observed facts in this group.</p>')
            claims_html = []
        else:
            claims_html = ['<div class="findings">', *(claim_html(index) for index in claim_indexes), '</div>']
        parts.append('<div class="exhibits">')
        for index in chart_indexes:
            chart = result.charts[index]
            chart_id = f'chart-{kind}-{index}'
            parts.append(f'<figure id="exhibit-{index}"><h3>{escape(chart.title)}</h3><p>{escape(chart.purpose)}</p>')
            if chart.points and chart.chart_type == "table":
                parts.append('<table class="chart-table"><thead><tr><th>Geography / measure</th><th>Reference period</th><th>Value</th></tr></thead><tbody>')
                for point in chart.points:
                    parts.append(f'<tr><td>{escape(point.geography)}<br><small>{escape(measure_label(point.metric))}</small></td><td>{point.period_start}<br>{point.period_end}</td><td>{point.value:,.2f}<small>{escape(point.unit)}</small></td></tr>')
                parts.append('</tbody></table>')
            elif chart.points or chart.pairs:
                parts.append(f'<div class="chart" id="{chart_id}"></div>')
                payload = json.loads(chart_figure(chart).to_json())
                encoded = json.dumps(payload).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
                scripts.append(f'{{const f={encoded};Plotly.newPlot("{chart_id}",f.data,f.layout,{{responsive:true,displaylogo:false}});}}')
            else:
                parts.append('<p>Insufficient comparable observations for a chart. No values have been fabricated.</p>')
            measurements = chart_points(chart)
            total = displayed_flow_total(chart)
            if total:
                parts.append('<p>' + escape(total) + '</p>')
            definitions = sorted({point.definition for point in measurements})
            publishers = sorted({point.publisher for point in measurements})
            periods = sorted({f'{point.period_start} to {point.period_end}' for point in measurements})
            parts.append('<figcaption>' + escape('Source: ' + ', '.join(publishers)) + ' ' + ' '.join(citation(eid) for eid in chart.source_refs) + '<br>' + escape('Reference periods: ' + '; '.join(periods)) + '<br>' + escape('; '.join(chart.caveats)) + '<details><summary>Measurement definitions and observation IDs</summary><p>' + escape('; '.join(definitions)) + '</p><p>' + escape('; '.join(point.observation_id for point in measurements)) + '</p></details></figcaption></figure>')
        parts.append('</div>' + ''.join(claims_html) + '</section>')
    if not result.claims:
        parts.append(f'<p>{escape(result.answer)}</p>')
    parts.append('<section id="source-appendix"><h2>Sources and coverage</h2>')
    if result.weekly:
        parts.append(f'<details class="coverage-detail"><summary>Evidence coverage and source concentration</summary><p>{escape(copy["coverage"])}</p><p>{escape(copy["concentration"])}</p><table><thead><tr><th>Publisher</th><th>Domain</th><th>Publications</th><th>Excerpts</th></tr></thead><tbody>')
        for item in result.weekly.source_concentration:
            parts.append(f'<tr><td>{escape(item.publisher)}</td><td>{escape(item.domain)}</td><td>{item.publication_count}</td><td>{item.excerpt_count}</td></tr>')
        parts.append(f'</tbody></table><p>{len(result.weekly.source_attempts)} recorded source checks · {len(result.weekly.fresh_searches)} fresh searches</p>')
        parts.append(f'<p>{escape(copy["prior"])}</p><p>{escape(copy["activity"])}</p>')
        if result.weekly.indicator_gaps:
            parts.append('<h3>Weekly measurement gaps</h3><ul>' + ''.join(f'<li>{escape(gap)}</li>' for gap in result.weekly.indicator_gaps) + '</ul>')
        for item in result.weekly.source_attempts:
            parts.append(f'<p>{escape(SOURCE_CHECK_LABELS[item.status])} · {escape(item.at)} · <a href="{escape(item.url)}">Source</a> {escape(item.error or "")}</p>')
        for item in result.weekly.fresh_searches:
            parts.append(f'<p>{escape(item.query)} · {escape(item.at)} · {escape(item.outcome)} · Publication window {escape(item.start_date or "Unspecified")} to {escape(item.end_date or "Unspecified")}</p>')
        parts.append('</details>')
    if result.coverage is None:
        parts.append("<p>Report-specific topic and area coverage record unavailable for this saved report.</p>")
    catalogue = source_catalogue_rows(evidence, observations)
    if catalogue:
        parts.append('<table><thead><tr><th>Publisher / role</th><th>Topics / geography</th><th>Publication / measured period</th><th>Source</th></tr></thead><tbody>')
        for row in catalogue:
            parts.append(f'<tr><td>{escape(row["Publisher"])}<small>{escape(row["Role"])}</small></td><td>{escape(row["Topics"])}<small>{escape(row["Source geography"])} · Coverage grouping {escape(row["Coverage grouping"])}</small></td><td>{escape(row["Publication date"])}<small>{escape(row["Latest measured period"])}</small></td><td><a href="{escape(row["Original publication"])}" rel="noreferrer">Original publication</a></td></tr>')
        parts.append('</tbody></table>')
    gaps = [f'{gap.geography}: {gap.reason}' for gap in result.coverage_gaps] + result.topic_gaps
    parts.append('<ul>' + ''.join('<li>' + escape(gap) + '</li>' for gap in gaps) + '</ul>')
    if '\n\nLimitations: ' in result.answer:
        notes = result.answer.rsplit('\n\nLimitations: ', 1)[1].split('; ')
        parts.append('<details><summary>Evidence limitations</summary><ul>' + ''.join('<li>' + escape(note) + '</li>' for note in notes) + '</ul></details>')
    if missing_ids:
        parts.append('<h3>Missing-source manifest</h3><ul>' + ''.join(f'<li>{escape(eid)}. Citation cannot currently be inspected.</li>' for eid in missing_ids) + '</ul>')
    for source in evidence:
        parts.append(f'<article id="evidence-{escape(source.id)}"><h3>[{source_numbers[source.id]}] {escape(source.mapped_geography)} · {escape(measure_label(source.category))}</h3><p>Source <a href="{escape(source.source_url)}" rel="noreferrer">{escape(source.source_url)}</a></p><blockquote>{escape(source.excerpt)}</blockquote><small>Publication date {escape(source.published_at or "Unknown")} · Event date {escape(source.event_date or "Unknown")} · Retrieved at {escape(source.retrieved_at)} · Source geography {escape(source.native_geography)} · Coverage grouping {escape(source.mapped_geography)} · {escape(MONITORING_LABELS[source.validation_status])} · Statement {escape(LABELS.get(source.statement_kind, "Unclassified"))}</small><details><summary>Source record details</summary><p>{escape(source.validation_reason or MONITORING_LABELS[source.validation_status])}</p><p>Evidence ID {escape(source.id)} · Monitoring status {escape(source.validation_status)}. Monitoring review concerns use as a recurring source; it does not independently verify every statement.</p></details>')
        for observation in observations:
            if source.id in observation.evidence_ids:
                parts.append(f'<p>{escape(measure_label(observation.metric))} · {escape(observation.value)} {escape(observation.unit)} · {escape(observation.period_start)} to {escape(observation.period_end)}<br>Definition {escape(observation.definition)}<br>Publisher {escape(observation.publisher)}<small>{escape(observation.id)}</small></p>')
        parts.append('</article>')
    parts.append('</section><footer>London office market research · Market-level evidence and interpretation. This report does not assess the value or rental terms of a specific asset. Publication dates, observation periods and retrieval dates have distinct meanings.</footer><script>' + get_plotlyjs() + '</script><script>' + ''.join(scripts) + '</script></body></html>')
    return ''.join(parts)
