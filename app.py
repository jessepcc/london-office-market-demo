"""Streamlit frontend; research and on-demand tasks share the bounded backend runner."""

from __future__ import annotations

import html
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo
import streamlit as st
from pydantic import ValidationError

from london_market.config import Settings
from london_market.domain import ChatMessage, ChartPayload, Evidence, Observation, RunResult, Task, TraceEvent, SUBMARKETS, utc_now
from london_market.fixtures import coverage_gaps
from london_market.report import coverage_html, report_coverage_rows, chart_figure, chart_points, chart_reference_id, displayed_flow_total, export_report_html, observation_scope, report_groups, report_key_figures, source_catalogue_rows
from london_market.presentation import CLAIM_LABELS as LABELS, VIEW_LABELS, STATUS_LABELS, MONITORING_LABELS, SOURCE_CHECK_LABELS, action_label, claim_label, claim_metadata, claim_timing, market_evidence_groups, measure_label, partition_market_evidence, research_pipeline_rows, runtime_skill_rows, weekly_indicator_groups, weekly_summary_copy
from london_market.runner import Runner
from london_market.jobs import ManualJobs
from london_market.store import Store
from london_market.weekly import saved_claim_source_ids
from london_market.web_session import SessionData, session_settings

VIEWS = ("Weekly reports", "Research", "Controls", "Sources")

APP_STYLE = """
<style>
[data-testid="stApp"]{--market-surface:light-dark(#ffffff,#172b3e);--market-soft:light-dark(#f3f7fa,#172b3e);--market-ink:light-dark(#173a59,#edf3f7);--market-muted:light-dark(#526675,#bdccd7);--market-rule:light-dark(#dce5ed,#3b5266);--market-accent:light-dark(#116b68,#79d1cb)}
[data-testid="stApp"]{--market-bullish:light-dark(#17633e,#9ddbb4);--market-bearish:light-dark(#963535,#f0afaf)}
[data-testid="stMainBlockContainer"]{max-width:1280px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{font-family:Arial,sans-serif!important;font-weight:700!important}
h1{letter-spacing:-.03em;font-size:2.2rem!important}h2{font-size:1.45rem!important;margin-top:1.25rem!important}h3{font-size:1.15rem!important}
[data-testid="stCaptionContainer"]{opacity:1}[data-testid="stChatInput"] textarea::placeholder{color:inherit;opacity:.8}
[data-testid="stTable"] th{color:inherit}
.market-claim,.market-key-figure,.market-prompt,.market-coverage{background:var(--market-surface);color:var(--market-ink);border:1px solid var(--market-rule);border-radius:12px;padding:18px;margin:6px 0 12px;line-height:1.55;overflow-wrap:anywhere}
.market-claim{font-size:15px}.market-claim h4{font-size:16px;margin:5px 0 10px}.market-label{font:700 12px/1.5 Arial,sans-serif;color:var(--market-accent);display:block;margin-bottom:6px}
.market-timing,.market-tags{display:block;font:12px/1.6 Arial,sans-serif;color:var(--market-muted);margin-top:12px}.market-tags{margin-top:8px}.market-signal-heading{margin:24px 0 10px!important;font:700 18px/1.4 Arial,sans-serif!important}.market-signal-bullish{color:var(--market-bullish)}.market-signal-bearish{color:var(--market-bearish)}.market-signal-neutral,.market-signal-unavailable{color:var(--market-muted)}
.coverage-nav{display:flex;gap:8px;flex-wrap:wrap;font-size:13px;margin:12px 0 20px}.coverage-nav a{color:var(--market-accent);text-decoration:underline}.coverage-nav a:focus-visible{outline:2px solid var(--market-accent);outline-offset:3px}.coverage-lane{padding:12px 0;border-bottom:1px solid var(--market-rule);max-width:90ch}.coverage-lane summary{font-size:14px;line-height:1.55;color:var(--market-ink);cursor:pointer}.coverage-lane summary:focus-visible{outline:2px solid var(--market-accent);outline-offset:3px}.coverage-lane p{margin:7px 0;font-size:12px}.coverage-lane small,.coverage-gap{color:var(--market-muted);font-size:12px;line-height:1.6}.coverage-links a{color:var(--market-accent);text-decoration:underline;margin-right:8px}.market-citation{font-size:12px;color:var(--market-muted)}.market-claim a{color:var(--market-accent);text-decoration:underline;text-underline-offset:3px}.market-key-value{font:700 30px/1.15 Arial,sans-serif;font-variant-numeric:tabular-nums}.market-key-figure p{font-size:14px;margin:10px 0 7px}.market-key-figure small{display:block;font:12px/1.55 Arial,sans-serif;color:var(--market-muted)}
.market-prompt{font-size:14px;background:var(--market-soft)}.market-prompt strong{display:block;font-size:15px;margin:0 0 8px}.market-coverage{padding:14px;font-size:14px}.market-coverage strong{display:block;margin-bottom:6px}.market-coverage small{font-size:12px;color:var(--market-muted)}
[data-testid="stPlotlyChart"]{border:1px solid var(--market-rule);border-radius:12px;background:var(--market-surface);overflow:hidden}
@media(max-width:700px){[data-testid="stMainBlockContainer"]{padding-top:1.5rem}.market-claim,.market-prompt,.market-key-figure{padding:16px}h1{font-size:1.9rem!important}}
</style>
"""


def display_date(value) -> str:
    return value.isoformat() if value is not None else "Unknown"


def display_run_time(value) -> str:
    return value.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC") if value is not None else "Unknown"


def observation_rows(observations: list[Observation]) -> list[dict]:
    return [{"Coverage area": item.mapped_geography, "Source geography": item.native_geography, "Measure": measure_label(item.metric), "Value": item.value, "Unit": item.unit, "Reference period": f"{item.period_start} to {item.period_end}", "Publisher": item.publisher, "Definition": item.definition, "Topic": measure_label(item.category), "Observation ID": item.id, "Conflict/revision link": item.revision_of or ""} for item in observations]


def render_source(evidence: Evidence, observations: list[Observation]) -> None:
    if evidence.mode == "fixture":
        st.caption(f"Synthetic source reference: {evidence.source_url}")
    else:
        st.link_button("Open original source", str(evidence.source_url))
    st.text(evidence.excerpt)
    if evidence.statement_kind:
        st.caption("Source statement: " + LABELS[evidence.statement_kind])
    st.table([{"Field": field, "Value": value} for field, value in (("Publication date", display_date(evidence.published_at)), ("Event date", display_date(evidence.event_date)), ("Retrieved at", display_date(evidence.retrieved_at)), ("Source geography", evidence.native_geography), ("Coverage grouping", evidence.mapped_geography), ("Topic", measure_label(evidence.category)))])
    st.caption(MONITORING_LABELS[evidence.validation_status])
    with st.expander("Source record details"):
        st.caption(f"Evidence ID: {evidence.id}")
        st.caption(f"Monitoring status: {evidence.validation_status}")
        if evidence.validation_reason:
            st.caption(evidence.validation_reason)
        st.caption("Monitoring review concerns use as a recurring source; it does not independently verify every statement.")
    linked = [item for item in observations if evidence.id in item.evidence_ids]
    if linked:
        st.dataframe(observation_rows(linked), hide_index=True, width="stretch")
    if evidence.mapped_geography in ("Central London", "Whole London", "Unmapped"):
        st.caption("Aggregate source evidence does not establish coverage of each London submarket.")


def render_chart(chart: ChartPayload, key: str) -> None:
    st.markdown(f"**{chart.title}**")
    st.caption(chart.purpose)
    if chart.chart_type == "insufficient_data" or not (chart.points or chart.pairs):
        st.info("Insufficient comparable observations for a chart. No values have been fabricated.")
    elif chart.chart_type == "table":
        st.dataframe([{"Submarket": point.geography, "Measure": measure_label(point.metric), "Value": point.value, "Unit": point.unit, "Reference period": f"{point.period_start} to {point.period_end}", "Definition": point.definition, "Publisher": point.publisher, "Observation ID": point.observation_id} for point in chart.points], hide_index=True, width="stretch")
    else:
        figure = chart_figure(chart, theme=st.context.theme.type)
        st.plotly_chart(figure, key=key, use_container_width=True, config={"displaylogo": False})
        if chart.chart_type in ("line", "multi_line"):
            st.caption("Source measurements at the stated reference periods; no interpolated observations.")
    for caveat in chart.caveats:
        st.caption(caveat)
    total = displayed_flow_total(chart)
    if total:
        st.caption(total)
    measurements = chart_points(chart)
    if measurements:
        st.caption("Source: " + ", ".join(sorted({point.publisher for point in measurements})))
        st.caption("Reference periods: " + "; ".join(sorted({f"{point.period_start} to {point.period_end}" for point in measurements})))
        with st.expander("Measurement definitions and chart data"):
            st.dataframe([{"Label": point.label, "Geography": point.geography, "Source geography": point.native_geography or point.geography, "Segment": point.segment or "", "Value": point.value, "Unit": point.unit, "Reference period": f"{point.period_start} to {point.period_end}", "Publisher": point.publisher, "Definition": point.definition, "Observation ID": point.observation_id, "Evidence IDs": ", ".join(point.evidence_ids)} for point in measurements], hide_index=True, width="stretch")


def render_research_log(run_id: str, settings: Settings, *, label: str = "Run log") -> None:
    with Store(settings.database_path) as store:
        traces = store.list_traces(run_id)
        run = store.get_run(run_id)
    render_research_pipeline(traces, run.task.entry_skill if run else None)
    with st.expander(label):
        st.caption(f"Run {run_id}. Recorded actions and outcomes.")
        st.dataframe([{"Time": event.at.strftime("%H:%M:%S"), "Action": action_label(event.name), "Outcome": event.outcome} for event in traces if event.kind in ("tool", "router", "skill", "termination")], hide_index=True, width="stretch")


def render_research_pipeline(traces: list[TraceEvent], entry_skill: str | None) -> None:
    rows = research_pipeline_rows(traces)
    if rows:
        st.markdown('#### Research pipeline')
        if entry_skill:
            st.caption('Task: ' + action_label(entry_skill))
        for row in rows:
            st.markdown(f'**{row["Work"]}** · {row["Recorded outcomes"]}')
            st.caption(row['Recorded actions'])
    skills = runtime_skill_rows(traces)
    if skills:
        with st.expander('Runtime skills'):
            st.table(skills)


def progress_callback(status):
    def on_event(event: TraceEvent) -> None:
        action = "Check question relevance" if event.kind == "router" else action_label(event.name)
        if event.kind == "progress":
            status.update(label=f"{action}…")
        elif event.kind in ("tool", "router", "skill", "termination"):
            if event.kind == "termination":
                status.write(STATUS_LABELS.get(event.outcome, event.outcome))
            elif event.outcome == "failed":
                status.write(f"{action}: failed. See Run history for details.")
            elif event.outcome in ("success", "loaded", "classified"):
                status.write(f"{action}: done")
    return on_event


def render_result(result: RunResult, settings: Settings, key: str, *, allow_export: bool) -> None:
    with Store(settings.database_path) as store:
        observation_ids = set(result.observation_ids)
        if result.weekly:
            observation_ids.update(oid for item in result.weekly.week_over_week for oid in (item.from_observation_id, item.to_observation_id))
        snapshot = [store.get_observation(observation_id, settings.mode) for observation_id in sorted(observation_ids)]
        snapshot = [item for item in snapshot if item is not None]
        cited_ids = sorted(set(result.evidence_ids) | {eid for claim in result.claims for eid in claim.evidence_ids} | {eid for chart in result.charts for eid in chart.source_refs})
        evidence = {evidence_id: store.get_evidence(evidence_id, settings.mode) for evidence_id in cited_ids}
        all_observations = store.list_observations(settings.mode)
    source_numbers = {evidence_id: index + 1 for index, evidence_id in enumerate(cited_ids)}
    chart_indexes_by_id = {chart_reference_id(chart): index for index, chart in enumerate(result.charts)}
    key_figures = report_key_figures(result, [item for item in evidence.values() if item], snapshot)
    def render_figures():
        for position in range(0, len(key_figures), 2):
            for column, (item, label) in zip(st.columns(min(2, len(key_figures) - position)), key_figures[position:position + 2], strict=True):
                citations = " ".join(f"[{source_numbers[evidence_id]}]" for evidence_id in item.evidence_ids)
                column.markdown(f'<div class="market-key-figure"><div class="market-key-value">{item.value:g}</div><small>{html.escape(item.unit)}</small><p>{html.escape(label)} · {html.escape(observation_scope(item))}</p><small>{item.period_start} to {item.period_end}<br>{html.escape(item.publisher)} {citations}<br>{html.escape(item.definition)}</small></div>', unsafe_allow_html=True)
    if result.weekly:
        copy = weekly_summary_copy(result)
        st.subheader("This week")
        st.markdown(f'**{copy["window"]}**  \n{copy["cutoff"]}  \n{copy["status"]}  \n{copy["measurements"]}')
        if result.weekly.week_over_week:
            st.markdown("**Week-over-week indicators**")
        for heading, rows in weekly_indicator_groups(result, snapshot):
            st.markdown(f"**{heading}**")
            st.dataframe(rows, hide_index=True, width="stretch")
    elif key_figures and result.status not in ("failed", "stopped"):
        render_figures()

    def render_claims(indexes):
        for position in range(0, len(indexes), 2):
            columns = st.columns(min(2, len(indexes) - position))
            for column, index in zip(columns, indexes[position:position + 2], strict=True):
                claim = result.claims[index]
                unsupported = claim.kind in ("fact", "forecast") and any(evidence.get(eid) is None for eid in saved_claim_source_ids(claim, result))
                label = "Unsupported " + claim.kind if unsupported else claim_label(claim)
                if claim.kind == "fact" and result.weekly is None:
                    label += " · Signal unavailable" if unsupported or claim.signal is None else " · " + claim.signal.capitalize()
                citations = " ".join(f'<a href="{html.escape(str(evidence[eid].source_url), quote=True)}" target="_blank" rel="noreferrer">[{source_numbers[eid]}]</a>' if evidence.get(eid) else f"[{source_numbers[eid]}]" for eid in claim.evidence_ids)
                citations += " " + " ".join(f'<a href="#{key}-exhibit-{chart_indexes_by_id[chart_id]}">Exhibit {chart_indexes_by_id[chart_id] + 1}</a>' for chart_id in claim.chart_ids if chart_id in chart_indexes_by_id)
                headline = f'<h4>{html.escape(claim.headline)}</h4>' if claim.headline else ""
                metadata = " · ".join(claim_metadata(claim, result, [item for item in evidence.values() if item], snapshot))
                tags = " ".join("#" + tag for tag in claim.topic_tags)
                with column:
                    if unsupported:
                        st.warning("This claim's source is unavailable; treat it as unsupported.")
                    st.markdown(f'<div id="{key}-claim-{index}" class="market-claim"><span class="market-label">{html.escape(label)}</span>{headline}{html.escape(claim.text)} <span class="market-citation">{citations}</span><small class="market-timing">{html.escape(metadata)}</small><span class="market-tags">{html.escape(tags)}</span></div>', unsafe_allow_html=True)
    if result.status in ("failed", "stopped"):
        st.warning(result.answer)
        st.caption(result.reason)
    else:
        if result.weekly is None:
            st.caption(STATUS_LABELS[result.status] + (". Review the coverage and topic gaps below." if result.status == "partial" else ""))
        if "Editor-reviewed:" in result.reason:
            st.caption("Editor-reviewed briefing. Run history retains the original draft and documented corrections.")
        if not result.claims:
            st.write(result.answer)
    groups = list(report_groups(result))
    if result.coverage is not None:
        st.markdown('<nav class="coverage-nav" aria-label="Report sections">' + ' · '.join(f'<a href="#{html.escape(key)}-{section_key}">{html.escape(title)}</a>' for section_key, title, _, _ in groups) + '</nav>', unsafe_allow_html=True)
    coverage_rows = report_coverage_rows(result, [item for item in evidence.values() if item], snapshot)
    for section_key, title, claim_indexes, chart_indexes in groups:
        if result.sections or result.briefing_date or result.weekly or result.coverage:
            st.subheader(title, anchor=f"{key}-{section_key}")
        if section_key in ("coverage_topics", "coverage_areas"):
            st.markdown(coverage_html(coverage_rows, "topic" if section_key == "coverage_topics" else "area", prefix=key + "-", chart_indexes=chart_indexes_by_id, link_findings=False), unsafe_allow_html=True)
            continue
        if section_key == "market_facts":
            st.caption("From a landlord’s leasing perspective")
            missing = {eid for eid, source in evidence.items() if source is None}
            for signal, label, indexes in market_evidence_groups(result, missing):
                st.markdown(f'<h3 class="market-signal-heading market-signal-{signal or "unavailable"}">{label} · {len(indexes)}</h3>', unsafe_allow_html=True)
                current, retained = partition_market_evidence(result, indexes)
                render_claims(current)
                if retained:
                    with st.expander(f"Retained context · {len(retained)}", expanded=False):
                        render_claims(retained)
                if not indexes:
                    st.caption("No observed facts in this group.")
            continue
        if chart_indexes:
            columns = st.columns(min(len(chart_indexes), 2))
            for position, index in enumerate(chart_indexes):
                with columns[position % len(columns)]:
                    st.markdown(f'<div id="{key}-exhibit-{index}"></div>', unsafe_allow_html=True)
                    render_chart(result.charts[index], f"{key}-{section_key}-chart-{index}")
        if result.status not in ("failed", "stopped"):
            render_claims(claim_indexes)
    if result.weekly and key_figures and result.status not in ("failed", "stopped"):
        with st.expander("Market context"):
            st.caption("Latest supported measurements at their source reference periods. These are not weekly changes.")
            render_figures()
    if result.weekly:
        with st.expander("Evidence coverage and source concentration"):
            st.write(copy["coverage"])
            st.caption(copy["concentration"])
            st.caption(copy["prior"])
            st.caption(copy["activity"])
            if result.weekly.indicator_gaps:
                st.markdown("**Weekly measurement gaps**\n\n" + "\n".join("- " + gap for gap in result.weekly.indicator_gaps))
            if result.weekly.source_concentration:
                st.table([{"Publisher": item.publisher, "Domain": item.domain, "Publications": item.publication_count, "Excerpts": item.excerpt_count} for item in result.weekly.source_concentration])
            st.caption(f"{len(result.weekly.source_attempts)} recorded source checks · {len(result.weekly.fresh_searches)} fresh searches")
            failed_checks = [item for item in result.weekly.source_attempts if item.status in ("failed", "blocked")]
            if failed_checks:
                st.markdown("\n".join(f"- {SOURCE_CHECK_LABELS[item.status]} · {item.url} · {item.error or 'No further error detail recorded'}" for item in failed_checks))
            if result.weekly.source_attempts:
                st.dataframe([{"Source": str(item.url), "Checked": display_date(item.at), "Status": SOURCE_CHECK_LABELS[item.status], "Error": item.error or ""} for item in result.weekly.source_attempts], hide_index=True, width="stretch")
            if result.weekly.fresh_searches:
                st.dataframe([{"Query": item.query, "Searched": display_date(item.at), "Outcome": item.outcome, "Publication window": f"{item.start_date or 'Unspecified'} to {item.end_date or 'Unspecified'}"} for item in result.weekly.fresh_searches], hide_index=True, width="stretch")
    if allow_export:
        st.download_button("Download interactive report", data=export_report_html(result, [item for item in evidence.values() if item], snapshot), file_name="london-office-report.html", mime="text/html", key=f"{key}-download-html")
    if snapshot:
        with st.expander("Dated market snapshot", expanded=False):
            st.dataframe(observation_rows(snapshot), hide_index=True, width="stretch")
    if result.coverage is None and (result.weekly or result.briefing_date):
        st.caption("Report-specific topic and area coverage record unavailable for this saved report.")
    if result.coverage_gaps:
        st.markdown("**Coverage gaps:** " + "; ".join(gap.geography for gap in result.coverage_gaps))
    if result.topic_gaps:
        st.markdown("**Topic gaps**\n\n" + "\n".join("- " + gap for gap in result.topic_gaps))
    if "\n\nLimitations: " in result.answer:
        with st.expander("Evidence limitations"):
            notes = result.answer.rsplit("\n\nLimitations: ", 1)[1].split("; ")
            st.markdown("\n".join("- " + note for note in notes))
    if evidence:
        with st.expander("Inspect cited sources"):
            for evidence_id, item in evidence.items():
                if item is None:
                    st.warning(f"Evidence unavailable: {evidence_id}. This citation cannot currently be inspected.")
                else:
                    st.markdown(f"**[{source_numbers[evidence_id]}] {item.mapped_geography} · {measure_label(item.category)}**")
                    render_source(item, all_observations)
                    st.divider()
    with st.expander("Run details and limitations"):
        if result.weekly:
            context = result.weekly.context
            st.caption(f"Briefing date {context.briefing_date} · filed under week of {context.report_week_start}")
        st.write(result.reason)
        if not result.claims:
            st.write(result.answer)
        st.caption(f"Usage: {result.usage.input_tokens + result.usage.output_tokens:,} tokens; ${result.usage.spend_usd:.4f} {'provider reported' if result.usage.cost_known else 'includes conservative estimates'}")


def research_view(settings: Settings) -> None:
    st.subheader("Chat over London offices")
    st.caption("Ask a question about London offices or choose a starting point.")
    conversations = st.session_state.setdefault("conversations", {"fixture": [], "live": []})
    conversation = conversations[settings.mode]
    if not conversation:
        prompt_cards = (("Leasing conditions", "What do the latest rents and vacancy evidence show across London submarkets?"), ("Competing supply", "Which office developments, refurbishments and pre-lets deserve landlord attention?"), ("Occupier demand", "What evidence supports flight-to-quality, sustainability and AI occupier demand?"))
        selected_prompt = None
        for column, (label, prompt) in zip(st.columns(3), prompt_cards, strict=True):
            column.markdown(f'<div class="market-prompt"><strong>{html.escape(label)}</strong>{html.escape(prompt)}</div>', unsafe_allow_html=True)
            if column.button("Ask this question", key=f"prompt-{settings.mode}-{label}", disabled=not live_ready(settings)):
                selected_prompt = prompt
        with Store(settings.database_path) as store:
            observations = store.list_observations(settings.mode)
            evidence = store.list_evidence(settings.mode)
        st.markdown("**Market coverage**")
        for column, geography in zip(st.columns(len(SUBMARKETS)), SUBMARKETS, strict=True):
            metrics = [item for item in observations if item.mapped_geography == geography]
            qualitative = [item for item in evidence if item.mapped_geography == geography]
            label = f"{len({item.metric for item in metrics})} measures" if metrics else "Qualitative evidence only" if qualitative else "No saved evidence"
            period = f"Latest period ends {max(item.period_end for item in metrics):%d %b %Y}" if metrics else "No reviewed measurements"
            if metrics and geography in ("Midtown and Fringe", "Wider London"):
                native = sorted({item.native_geography for item in metrics})
                label = "Samples: " + ", ".join(native)
                if geography == "Midtown and Fringe" and all("Midtown" not in item for item in native):
                    label += "; Midtown gap"
            column.markdown(f'<div class="market-coverage"><strong>{html.escape(geography)}</strong>{label}<br><small>{period}</small></div>', unsafe_allow_html=True)
        st.caption("Central London aggregates do not establish coverage of every submarket or wider London.")
    else:
        selected_prompt = None
    for message in conversation:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("result"):
                render_result(RunResult.model_validate(message["result"]), settings, message["run_id"], allow_export=True)
                render_research_log(message["run_id"], settings, label="Chat log")
            else:
                st.write(message["content"])
    question = st.chat_input("Ask a London office market question", disabled=not live_ready(settings), key=f"question-{settings.mode}") or selected_prompt
    if question:
        history = [ChatMessage(role=message["role"], content=message["content"]) for message in conversation[-6:]]
        conversation.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.status("Working on your question…", expanded=True) as status:
            with Store(settings.database_path) as store:
                run = Runner(settings, store).run(Task(objective=question, entry_skill="research_market", mode=settings.mode, trigger="chat", history=history, limits=settings.limits), on_event=progress_callback(status))
            status.update(label=f"Answer: {STATUS_LABELS[run.status]}", state="error" if run.status in ("failed", "stopped") else "complete", expanded=False)
        conversation.append({"role": "assistant", "content": run.result.answer, "run_id": run.id, "result": run.result.model_dump(mode="json")})
        with st.chat_message("assistant"):
            render_result(run.result, settings, run.id, allow_export=True)
            render_research_log(run.id, settings, label="Chat log")


def navigate(view: str, *, selection_key: str | None = None, selection_value: str | None = None) -> None:
    if selection_key is not None:
        st.session_state[selection_key] = selection_value
    if view == "Activity":
        st.session_state["show-run-history"] = True
    else:
        st.session_state["show-run-history"] = False
        st.session_state["view"] = view


def close_run_history() -> None:
    st.session_state["show-run-history"] = False


def controls_view(settings: Settings) -> None:
    st.subheader("Controls and tasks")
    st.caption("Run collection or prepare a briefing now. Ask ad-hoc questions in Chat.")
    st.button("Open Chat", key=f"controls-chat-{settings.mode}", on_click=navigate, args=("Research",))
    today = utc_now().astimezone(ZoneInfo(settings.timezone)).date()
    week_start = today - timedelta(days=today.weekday())
    collect_column, briefing_column = st.columns(2)
    disabled = not live_ready(settings)
    with collect_column:
        st.markdown("**Collect market evidence**")
        st.write("Check monitoring sources and save new evidence and measurements. This task does not create a briefing.")
        collect_clicked = st.button("Collect market evidence", key="collect", disabled=disabled)
    with briefing_column:
        st.markdown("**Prepare weekly briefing**")
        st.write("Use saved evidence and check fresh sources for important gaps.")
        st.caption(f"Dated today, {today:%d %b %Y} in {settings.timezone}. Filed under the week of Monday {week_start:%d %b %Y}.")
        briefing_clicked = st.button("Prepare weekly briefing", key="briefing", disabled=disabled)
    st.caption("The briefing separates its seven-day reporting window from retained evidence with earlier reference periods.")
    st.caption("The first saved briefing for each week is retained. Reruns remain in Run history; their new output is previewed here.")
    action = "collection" if collect_clicked else "briefing" if briefing_clicked else None
    if action:
        with st.status("Starting source collection…" if action == "collection" else "Starting briefing…", expanded=True) as status:
            with Store(settings.database_path) as store:
                run, _ = ManualJobs(settings, store).run_manual(action, on_event=progress_callback(status))
            status.update(label=f"{action.capitalize()}: {STATUS_LABELS[run.status]}", state="error" if run.status in ("failed", "stopped") else "complete", expanded=False)
    with Store(settings.database_path) as store:
        last = {label: store.last_success(settings.mode, skill) for label, skill in (("Collection", "collect_market_evidence"), ("Briefing", "prepare_weekly_briefing"))}
        latest = next((run for run in store.list_runs(settings.mode, limit=30) if run.task.trigger == "manual" and run.task.entry_skill in ("collect_market_evidence", "prepare_weekly_briefing")), None)
        reports = store.list_reports(settings.mode)
        attempts = store.list_source_attempts(latest.id) if latest else []
    with st.expander("Last completed tasks"):
        st.caption(f"Tasks run on demand. Reports use {settings.timezone} dates.")
        for label, run in last.items():
            st.write(f"**Last completed {label.lower()}:** {display_run_time(run.finished_at) + ' (' + STATUS_LABELS[run.status] + ')' if run else 'None recorded'}")
    if latest is None:
        st.info("No manual collection or briefing in recent run history for this data mode. Choose a task above.")
        return
    kind = "collection" if latest.task.entry_skill == "collect_market_evidence" else "briefing"
    st.subheader(f"Latest manual {kind}")
    st.caption(f"{display_run_time(latest.started_at)} · {STATUS_LABELS[latest.status]}")
    st.button("Inspect run", key=f"controls-run-{settings.mode}", on_click=navigate, args=("Activity",), kwargs={"selection_key": f"run-select-{settings.mode}", "selection_value": latest.id})
    if latest.result is None:
        st.info("This task is still in progress. Inspect its run for recorded actions.")
        return
    saved = next((report for report in reports if report["week_start"] == str(latest.task.report_week_start)), None) if kind == "briefing" else None
    if saved:
        if saved["run_id"] != latest.id:
            st.info("This week's first saved report was retained. The new run output is shown below and remains in Run history.")
        else:
            st.info("Briefing saved to Weekly briefings.")
        st.button("Open saved briefing", key=f"controls-report-{settings.mode}", on_click=navigate, args=("Weekly reports",), kwargs={"selection_key": f"report-select-{settings.mode}", "selection_value": saved["id"]})
    with st.expander("Task outcome details", expanded=latest.status in ("failed", "stopped") or any(attempt.status in ("failed", "blocked") for attempt in attempts)):
        st.write(latest.result.reason)
        if latest.result.coverage_gaps:
            st.table([{"Coverage area": gap.geography, "Gap": gap.reason} for gap in latest.result.coverage_gaps])
        if attempts:
            st.dataframe([{"Source": str(attempt.url), "Checked": display_date(attempt.at), "Status": SOURCE_CHECK_LABELS[attempt.status], "Error": attempt.error or ""} for attempt in attempts], hide_index=True, width="stretch")
    if kind == "collection":
        count = latest.result.collection_counts.get("evidence", 0)
        st.info(f"{count} new source excerpts saved. Collection does not create a briefing.")
        if latest.status in ("failed", "stopped"):
            st.warning(latest.result.answer)
        else:
            st.write(latest.result.answer)
        if count == 0:
            st.info("No new evidence was stored. Retained older observations do not establish unchanged market conditions.")
    else:
        render_result(latest.result, settings, f"controls-{latest.id}", allow_export=True)
    render_research_log(latest.id, settings)


def reports_view(settings: Settings) -> None:
    st.subheader("Weekly landlord briefings")
    with Store(settings.database_path) as store:
        reports = store.list_reports(settings.mode)
    if not reports:
        st.info("No weekly briefings saved. Use Controls to collect evidence and prepare a briefing, or ask a question in Chat.")
        st.button("Open Controls", key=f"reports-controls-{settings.mode}", on_click=navigate, args=("Controls",))
        st.button("Open Chat", key=f"reports-chat-{settings.mode}", on_click=navigate, args=("Research",))
        return
    if len(reports) == 1:
        report = reports[0]
    else:
        by_id = {report["id"]: report for report in reports}
        selected = st.selectbox("Filing week", list(by_id), format_func=lambda report_id: f"Filed under week of {by_id[report_id]['week_start']} · briefing dated {by_id[report_id]['result'].briefing_date or 'date unknown'}", key=f"report-select-{settings.mode}")
        report = by_id[selected]
    if report["result"].weekly is None:
        st.caption(f"Briefing date {report['result'].briefing_date or 'Unknown'} · filed under week of {report['week_start']} · saved {report['created_at'].split('T')[0]} · {STATUS_LABELS[report['result'].status]}")
        with Store(settings.database_path) as store:
            observations = [store.get_observation(oid, settings.mode) for oid in report["result"].observation_ids]
        observations = [item for item in observations if item]
        if observations:
            latest = max(item.period_end for item in observations)
            st.caption(f"Latest measured period ends {latest}. Historical measurements retain their source periods.")
    render_result(report["result"], settings, report["id"], allow_export=True)


def sources_view(settings: Settings) -> None:
    st.subheader("Sources and coverage")
    with Store(settings.database_path) as store:
        evidence = store.list_evidence(settings.mode)
        observations = store.list_observations(settings.mode)
        gaps = coverage_gaps(store, settings.mode)
    st.caption("Publication date, market event date, reference period and retrieval time are distinct. Unknown dates remain unknown.")
    st.caption("Midtown and Fringe includes City Fringe samples; Midtown itself remains a gap. Wider London evidence covers selected districts.")
    if settings.mode == "live" and not settings.cw_permission:
        with st.expander("Source access limits"):
            st.caption("Cushman monitoring is disabled because applicable reuse permission has not been recorded.")
    if gaps:
        st.write("Areas without reviewed measurements: " + ", ".join(gap.geography for gap in gaps))
    if not evidence:
        st.info("No sources saved in this mode. Start a chat or collect market evidence.")
        return
    catalogue = source_catalogue_rows(evidence, observations)
    columns = st.columns(3)
    columns[0].metric("Source publications", len(catalogue))
    columns[1].metric("Measures", len({item.metric for item in observations}))
    columns[2].metric("Areas without measurements", len(gaps))
    primary = sum(item["Role"].startswith("Primary") for item in catalogue)
    secondary = sum(item["Role"] == "Secondary news" for item in catalogue)
    candidate = sum(item["Role"] == "Candidate research" for item in catalogue)
    commentary = sum(item["Role"] == "Research commentary" for item in catalogue)
    st.caption(f"{primary} primary publications · {secondary} secondary · {commentary} research commentary · {candidate} research candidates" if settings.mode == "live" else "Synthetic demonstration publications")
    st.dataframe(catalogue, hide_index=True, width="stretch", column_order=["Original publication", "Publisher", "Role", "Source geography", "Coverage grouping", "Topics", "Publication date", "Latest measured period", "Measurements"], column_config={"Original publication": st.column_config.LinkColumn("Source", display_text="Open source", width="small"), "Publisher": st.column_config.TextColumn(width="medium"), "Role": st.column_config.TextColumn(width="medium")})
    st.caption("Primary publications and secondary news retain their roles. Research candidates await monitoring review. Qualitative sources do not establish measured trends.")
    publishers = {row["Original publication"]: row["Publisher"] for row in catalogue}
    by_id = {item.id: item for item in evidence}
    selected = st.selectbox("Inspect a source excerpt", list(by_id), format_func=lambda evidence_id: f"{publishers[str(by_id[evidence_id].source_url)]} · {by_id[evidence_id].native_geography} · {measure_label(by_id[evidence_id].category)} · published {display_date(by_id[evidence_id].published_at)} · {by_id[evidence_id].excerpt[:70]}", key=f"source-select-{settings.mode}")
    render_source(by_id[selected], observations)


def activity_view(settings: Settings) -> None:
    st.button("Back", key="activity-back", on_click=close_run_history)
    st.subheader("Run history")
    with Store(settings.database_path) as store:
        last = {label: store.last_success(settings.mode, skill) for label, skill in (("Collection", "collect_market_evidence"), ("Briefing", "prepare_weekly_briefing"))}
        runs = store.list_runs(settings.mode, limit=30)
        gaps = coverage_gaps(store, settings.mode)
    for label, run in last.items():
        st.write(f"**Last completed {label.lower()}:** {display_run_time(run.finished_at) + ' (' + STATUS_LABELS[run.status] + ')' if run else 'None recorded'}")
    st.caption("Runs completed with gaps retain their coverage limits and source failures.")
    if gaps:
        st.write("Coverage gaps: " + ", ".join(gap.geography for gap in gaps))
    if not runs:
        st.info("No runs recorded. Start a chat or prepare a weekly briefing.")
        return
    st.dataframe([{"Run": position, "Started": display_run_time(run.started_at), "Task": action_label(run.task.entry_skill), "Started by": {"chat": "Chat", "manual": "Manual action", "fixture": "Demonstration"}.get(run.task.trigger, run.task.trigger), "Status": STATUS_LABELS[run.status], "Reason": run.result.reason if run.result else "In progress"} for position, run in enumerate(runs, 1)], hide_index=True, width="stretch")
    by_id = {run.id: run for run in runs}
    selected = st.selectbox("Inspect a run", list(by_id), format_func=lambda run_id: f"Run {list(by_id).index(run_id) + 1} · {display_run_time(by_id[run_id].started_at)} · {STATUS_LABELS[by_id[run_id].status]} · {action_label(by_id[run_id].task.entry_skill)}", key=f"run-select-{settings.mode}")
    run = by_id[selected]
    with Store(settings.database_path) as store:
        attempts = store.list_source_attempts(run.id)
        traces = store.list_traces(run.id)
    if run.result:
        st.write(f"**Outcome:** {STATUS_LABELS[run.status]}. {run.result.reason}")
    render_research_pipeline(traces, run.task.entry_skill)
    with st.expander("Source checks", expanded=any(attempt.status in ("failed", "blocked") for attempt in attempts)):
        if attempts:
            st.dataframe([{"Source": str(attempt.url), "Checked": display_date(attempt.at), "Status": SOURCE_CHECK_LABELS[attempt.status], "Error": attempt.error or ""} for attempt in attempts], hide_index=True, width="stretch")
        else:
            st.caption("No source checks recorded for this run.")
    with st.expander("Observable actions and usage"):
        st.caption(f"Run {run.id} · main model {run.main_model} · router {run.router_model} · prompt {run.prompt_version}")
        st.caption(f"Started at {display_date(run.started_at)} · Finished at {display_date(run.finished_at)}")
        st.json(run.usage.model_dump(mode="json"))
        for trace in traces:
            st.write(f"{trace.at.isoformat()} · {trace.kind} · {trace.name} · {trace.outcome}")
            st.json(trace.data)
    if run.result and st.checkbox("Show saved answer and exhibits", key=f"saved-result-{run.id}"):
        render_result(run.result, settings, f"activity-{run.id}", allow_export=run.task.trigger != "chat")


def live_ready(settings: Settings) -> bool:
    return settings.mode == "fixture" or bool(settings.openrouter_api_key and settings.tavily_api_key)


def clear_session() -> None:
    data = st.session_state.get("session-data")
    if data is not None:
        data.close()
    st.session_state.clear()


def sidebar_settings() -> Settings:
    data = st.session_state.get("session-data")
    if data is None:
        data = SessionData()
        st.session_state["session-data"] = data
    mode = st.sidebar.selectbox("Mode", ("fixture", "live"), key="data-mode", format_func=lambda value: "Synthetic demo" if value == "fixture" else "Live with your keys")
    if mode == "live":
        st.sidebar.text_input("OpenRouter API key", type="password", key="openrouter-key")
        st.sidebar.text_input("Tavily API key", type="password", key="tavily-key")
        with st.sidebar.expander("Model settings"):
            st.text_input("Research model", value=Settings.model_fields["main_model"].default, key="main-model")
            st.text_input("Router model", value=Settings.model_fields["router_model"].default, key="router-model")
        st.sidebar.caption("Each run is limited to 180 seconds, 96 steps and $0.25 of model usage. Search charges may be separate.")
        st.sidebar.caption("Your keys pass through this demo server to OpenRouter and Tavily. They remain in session memory and are excluded from saved records and downloads. Provider charges apply to your accounts.")
    st.sidebar.button("Clear session", key="clear-session", on_click=clear_session)
    try:
        settings = session_settings(data, mode=mode, openrouter_key=st.session_state.get("openrouter-key", ""), tavily_key=st.session_state.get("tavily-key", ""), main_model=st.session_state.get("main-model"), router_model=st.session_state.get("router-model"))
    except (ValidationError, ValueError):
        st.error("Enter valid model settings to continue.")
        st.stop()
    if mode == "fixture":
        data.initialize_demo()
    return settings


def main() -> None:
    st.set_page_config(page_title="London office market", layout="wide")
    st.markdown(APP_STYLE, unsafe_allow_html=True)
    settings = sidebar_settings()
    st.title("London office market")
    mode = settings.mode
    selected_view = st.sidebar.radio("View", VIEWS, index=1, key="view", format_func=VIEW_LABELS.__getitem__, on_change=close_run_history)
    with st.sidebar.expander("Run history"):
        st.button("Inspect runs", key="open-run-history", on_click=navigate, args=("Activity",), use_container_width=True)
    view = "Activity" if st.session_state.get("show-run-history", False) else selected_view
    if view != "Weekly reports":
        st.caption("Market evidence for landlord and asset-management discussions")
    st.sidebar.caption("Research, collection and briefings run on demand.")
    st.sidebar.caption("This session's data is temporary. Download reports to keep them. Data is removed when the server releases the session or restarts.")
    if mode == "fixture":
        st.warning("SYNTHETIC DEMONSTRATION DATA · scripted model behavior. These are not live market findings.")
    else:
        st.caption("Live market sources" if view == "Weekly reports" else "Live market sources. Measurements retain their original reference periods; coverage may be incomplete.")
        missing = [name for name, value in (("OpenRouter model access", settings.openrouter_api_key), ("Tavily fresh search", settings.tavily_api_key)) if not value]
        if missing:
            st.warning("Unavailable services: " + ", ".join(missing) + ". Enter your keys in the sidebar to enable live actions.")
    {"Research": research_view, "Weekly reports": reports_view, "Controls": controls_view, "Sources": sources_view, "Activity": activity_view}[view](settings)


if __name__ == "__main__":
    main()
