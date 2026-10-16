"""Manual jobs retain reports, overlap protection and interruption recovery."""
import json
import subprocess
import sys
from datetime import date, datetime

import pytest

from london_market.config import Settings
from london_market.jobs import ManualJobs, job_task
from london_market.store import Store

NOW = datetime.fromisoformat("2026-10-12T09:00:00+01:00")


def cli(db, *arguments):
    result = subprocess.run([sys.executable, "-m", "london_market", *arguments, "--mode", "fixture", "--db", str(db)], capture_output=True, text=True, check=True)
    assert result.stdout.startswith("Demo mode uses synthetic sample data")
    return json.loads(result.stdout.split("\n", 1)[1])



def test_manual_collect_repeat_and_week_unique_briefing(tmp_path):
    path = tmp_path / "db"
    first = cli(path, "collect")
    second = cli(path, "collect")
    assert first["run"]["result"]["collection_counts"] == {"documents": 5, "evidence": 6, "observations": 4}
    assert second["run"]["result"]["collection_counts"] == {"documents": 0, "evidence": 0, "observations": 0}
    answer = second["run"]["result"]["answer"]
    assert "No new evidence was stored" in answer
    assert "do not establish unchanged market conditions" in answer
    assert "2026-03-31" in answer
    first_report = cli(path, "briefing", "--now", NOW.isoformat())
    second_report = cli(path, "briefing", "--now", NOW.isoformat())
    assert first_report["report_id"] == second_report["report_id"]
    assert first_report["run"]["task"]["trigger"] == "manual"
    assert "Weekly landlord briefing · 2026-10-12" in first_report["run"]["result"]["answer"]
    with Store(path) as store:
        assert len(store.list_reports("fixture")) == 1
        assert len(store.list_observations("fixture")) == 4
        assert {attempt.status for attempt in store.list_source_attempts(second["run"]["id"])} == {"unchanged"}



BLOCKING_SCRIPT = '''
import json, sys
from london_market.config import Settings
from london_market.domain import Task, Usage
from london_market.providers import ModelReply, ToolCall
from london_market.runner import Runner
from london_market.store import Store
class BlockingModel:
    def complete(self, messages, tools, max_output_tokens, timeout):
        print("MODEL_ENTERED", flush=True)
        sys.stdin.readline()
        return ModelReply(model="synthetic-blocker", provider="synthetic", calls=[ToolCall(id="finish",name="finish",arguments=json.dumps({"claims":[{"kind":"watchpoint","text":"Synthetic bounded test","evidence_ids":[]}]}))],usage=Usage())
with Store(sys.argv[1]) as store:
    run=Runner(Settings(),store,model=BlockingModel()).run(Task(objective="Synthetic lock check",entry_skill="research_market",mode=sys.argv[2],trigger="manual"))
    print(run.model_dump_json(),flush=True)
'''



def start_blocker(path, mode="fixture"):
    process = subprocess.Popen([sys.executable, "-c", BLOCKING_SCRIPT, str(path), mode], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "MODEL_ENTERED"
    return process



def test_two_actual_processes_cannot_overlap_or_recover_active_run(tmp_path):
    path = tmp_path / "market.sqlite3"
    first = start_blocker(path)
    try:
        alias = tmp_path / "same-database.sqlite3"
        alias.symlink_to(path)
        second = cli(alias, "research", "What is City office vacancy?")
        assert second["status"] == "stopped"
        assert "execution lock" in second["result"]["reason"]
        with Store(path) as store:
            running = [run for run in store.list_runs("fixture") if run.status == "running"]
            assert len(running) == 1
            assert not any(event.kind == "recovery" for event in store.list_traces(running[0].id))
        first.stdin.write("release\n")
        first.stdin.flush()
        stdout, stderr = first.communicate(timeout=5)
        assert first.returncode == 0, stderr
        assert json.loads(stdout)["status"] == "partial"
        with Store(path) as store:
            assert len(store.list_runs("fixture")) == 2
            stopped = store.get_run(second["id"])
            assert not any(event.kind == "model" for event in store.list_traces(stopped.id))
    finally:
        if first.poll() is None:
            first.kill()
            first.communicate(timeout=5)



def test_process_death_releases_lock_and_recovery_preserves_pending_usage(tmp_path):
    path = tmp_path / "market.sqlite3"
    first = start_blocker(path, mode="live")
    with Store(path) as store:
        abandoned = store.list_runs("live")[0]
        assert abandoned.status == "running"
    first.kill()
    first.communicate(timeout=5)
    with Store(path) as store:
        ManualJobs(Settings(), store).run_manual("collection", NOW)
        recovered = store.get_run(abandoned.id)
        assert recovered.status == "failed" and recovered.finished_at
        assert "Interrupted process" in recovered.result.reason
        assert recovered.result.usage.spend_usd > 0 and not recovered.result.usage.cost_known
        assert any(event.kind == "recovery" for event in store.list_traces(abandoned.id))



def test_recovery_requires_exclusive_execution_lock(tmp_path):
    with Store(tmp_path / "db") as store:
        with pytest.raises(RuntimeError, match="execution lock"):
            store.reconcile_abandoned_runs()



def test_manual_briefing_uses_current_local_date_and_filing_week():
    task = job_task(Settings(), "briefing", now=datetime.fromisoformat("2026-10-09T23:30:00-04:00"))
    assert task.briefing_date == date(2026, 10, 10)
    assert task.report_week_start == date(2026, 10, 5)
    assert task.trigger == "manual"
