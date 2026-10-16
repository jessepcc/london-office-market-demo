"""Small local entry points; fixture ingestion does not call external services."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from .config import load_settings
from .domain import ChatMessage, Task
from .fixtures import FIXTURE_LABEL, coverage_gaps, load_fixtures
from .store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description="London office market PoC")
    parser.add_argument("command", choices=("init-fixtures", "inspect", "research", "collect", "briefing", "reports", "status", "report-demo", "export-report"))
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--context", action="append", default=[], help="Recent conversation context; repeat for multiple messages")
    parser.add_argument("--now", help="Timezone-aware ISO timestamp for a manual briefing")
    parser.add_argument("--run-id", help="Inspect a saved run and observable trace")
    parser.add_argument("--db", type=Path, help="Override the configured SQLite path")
    parser.add_argument("--mode", choices=("fixture", "live"), help="Select the evidence partition for inspection")
    parser.add_argument("--report-id", help="Saved report to export; defaults to latest in selected mode")
    parser.add_argument("--output", type=Path, help="Explicit standalone HTML output path")
    parser.add_argument("--max-steps", type=int, default=None, help="Report-demo action cap, between 1 and 192")
    args = parser.parse_args()
    try:
        settings = load_settings()
    except (ValidationError, ValueError) as error:
        parser.error(str(error))
    mode = args.mode or settings.mode
    settings = settings.model_copy(update={"mode": mode, "database_path": args.db or settings.database_path})
    now = None
    if args.now:
        try:
            now = datetime.fromisoformat(args.now)
            if now.tzinfo is None:
                raise ValueError("timestamp timezone required")
        except ValueError:
            parser.error("--now must be an ISO timestamp with a timezone")
    with Store(args.db or settings.database_path) as store:
        if args.command == "report-demo":
            if mode != "fixture" or not args.db:
                parser.error("report-demo requires --mode fixture and an explicit isolated --db")
            if args.max_steps is not None and not 1 <= args.max_steps <= 192:
                parser.error("--max-steps must be between 1 and 192")
            from .report_demo import generate_report_demo
            run, report_id = generate_report_demo(settings, store, max_steps=args.max_steps)
            print(json.dumps({"run": run.model_dump(mode="json"), "report_id": report_id}, indent=2))
        elif args.command == "export-report":
            if not args.output:
                parser.error("export-report requires --output")
            reports = store.list_reports(mode)
            report = next((item for item in reports if args.report_id is None or item["id"] == args.report_id), None)
            if report is None:
                parser.error("no saved report matches the selected mode and report ID")
            from .report import export_report_html
            result = report["result"]
            cited_ids = set(result.evidence_ids) | {eid for claim in result.claims for eid in claim.evidence_ids} | {eid for chart in result.charts for eid in chart.source_refs}
            evidence = [store.get_evidence(eid, mode) for eid in sorted(cited_ids)]
            observations = [store.get_observation(oid, mode) for oid in result.observation_ids]
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(export_report_html(result, [item for item in evidence if item], [item for item in observations if item]))
            print(json.dumps({"report_id": report["id"], "output": str(args.output), "mode": mode}))
        elif args.command == "init-fixtures":
            print(FIXTURE_LABEL)
            print(json.dumps({"mode": "fixture", "new": load_fixtures(store), "stored": store.counts("fixture")}, indent=2))
        elif args.command in ("collect", "briefing"):
            from .jobs import ManualJobs
            run, report_id = ManualJobs(settings, store).run_manual("collection" if args.command == "collect" else "briefing", now)
            if mode == "fixture":
                print(FIXTURE_LABEL)
            print(json.dumps({"run": run.model_dump(mode="json"), "report_id": report_id}, indent=2))
        elif args.command == "reports":
            if mode == "fixture":
                print(FIXTURE_LABEL)
            reports = [{**report, "result": report["result"].model_dump(mode="json")} for report in store.list_reports(mode)]
            print(json.dumps({"mode": mode, "reports": reports}, indent=2))
        elif args.command == "status":
            if mode == "fixture":
                print(FIXTURE_LABEL)
            last_success = {kind: store.last_success(mode, skill) for kind, skill in (("collection", "collect_market_evidence"), ("briefing", "prepare_weekly_briefing"))}
            print(json.dumps({"mode": mode, "last_success": {kind: run.model_dump(mode="json") if run else None for kind, run in last_success.items()}, "recent_runs": [{"id": run.id, "entry_skill": run.task.entry_skill, "trigger": run.task.trigger, "started_at": run.started_at.isoformat(), "status": run.status, "reason": run.result.reason if run.result else None} for run in store.list_runs(mode, limit=20)]}, indent=2))
        elif args.command == "research":
            if not args.question:
                parser.error("research requires a question")
            from .runner import Runner
            task = Task(objective=args.question, entry_skill="research_market", trigger="chat", mode=mode, history=[ChatMessage(role="user", content=item) for item in args.context], limits=settings.limits)
            run = Runner(settings, store).run(task)
            if mode == "fixture":
                print(FIXTURE_LABEL)
            print(run.model_dump_json(indent=2))
        elif args.run_id:
            run = store.get_run(args.run_id)
            if run is None or run.task.mode != mode:
                parser.error("run ID does not exist in the selected mode")
            if mode == "fixture":
                print(FIXTURE_LABEL)
            print(json.dumps({"run": run.model_dump(mode="json"), "traces": [event.model_dump(mode="json") for event in store.list_traces(run.id)], "source_attempts": [attempt.model_dump(mode="json") for attempt in store.list_source_attempts(run.id)]}, indent=2))
        else:
            if mode == "fixture":
                print(FIXTURE_LABEL)
            print(json.dumps({"mode": mode, "stored": store.counts(mode), "observations": [item.model_dump(mode="json") for item in store.list_observations(mode)], "evidence": [item.model_dump(mode="json") for item in store.list_evidence(mode)], "coverage_gaps": [item.model_dump(mode="json") for item in coverage_gaps(store, mode)]}, indent=2))


if __name__ == "__main__":
    main()
