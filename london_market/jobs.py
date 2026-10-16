"""On-demand collection and briefings share the bounded research runner."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable, Literal
from zoneinfo import ZoneInfo

from .config import Settings
from .domain import RunRecord, Task, TraceEvent, utc_now
from .runner import Runner
from .store import RunBusy, Store

JobKind = Literal["collection", "briefing"]


def job_task(settings: Settings, kind: JobKind, *, trigger: Literal["manual"] = "manual", now: datetime) -> Task:
    day = now.astimezone(ZoneInfo(settings.timezone)).date()
    briefing_day = now.astimezone(ZoneInfo(settings.timezone)).date()
    if kind == "collection":
        objective = "Collect new London-office monitoring evidence for rents, vacancy/availability, take-up, supply/pipeline, submarkets and relevant macro/occupier/news. Record source checks, preserve dates, expose failed sources and missing coverage. No new evidence never proves unchanged market conditions."
        skill = "collect_market_evidence"
    else:
        objective = f"Prepare a concise professional landlord briefing dated {briefing_day}. Begin with the pinned publication-window query required by the weekly skill. Address material current releases, attributed commentary and announcements before historical measurements or exhibits. Cover rents, leasing/vacancy, supply and prelets, City/West End/Canary Wharf/Midtown-Fringe/Wider London, macro/employment, occupier quality/ESG/hybrid drivers and material news. Use primary and secondary evidence with exact citations. Resolve material stale or missing coverage with bounded fresh source checks. Include supported landlord interpretations and observable watchpoints. Prepare useful exhibits through choose_visualization and prepare_visualization_data after addressing current evidence. Omit repeated historical baselines unless they explain a current update or a watchpoint. Preserve source dates and reference periods. Never fabricate weekly measurements or imply unchanged conditions from no new evidence."
        skill = "prepare_weekly_briefing"
    return Task(objective=objective, entry_skill=skill, mode=settings.mode, trigger=trigger, limits=settings.limits, briefing_date=briefing_day if kind == "briefing" else None, report_week_start=day - timedelta(days=day.weekday()) if kind == "briefing" else None)



class ManualJobs:
    def __init__(self, settings: Settings, store: Store, runner: Runner | None = None):
        self.settings = settings
        self.store = store
        self.runner = runner or Runner(settings, store)

    def run_manual(self, kind: JobKind, now: datetime | None = None, *, on_event: Callable[[TraceEvent], None] | None = None) -> tuple[RunRecord, str | None]:
        now = now or utc_now()
        task = job_task(self.settings, kind, trigger="manual", now=now)
        try:
            with self.store.execution_lock():
                self.store.reconcile_abandoned_runs()
                run = self.runner.run(task, on_event=on_event) if on_event is not None else self.runner.run(task)
                report_id = self.store.save_report(run.id, task.report_week_start, run.result) if kind == "briefing" and run.status in ("completed", "partial") else None
                return run, report_id
        except RunBusy:
            return self.runner._busy_run(task), None
