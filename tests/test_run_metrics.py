"""Run summary collector: schema, aggregation, and no-op safety."""

from __future__ import annotations

import json
from pathlib import Path

from orchestrator import run_metrics
from orchestrator.cli import parse_args


def _sample_run(tmp_path: Path) -> run_metrics.RunMetrics:
    metrics = run_metrics.start_run(
        model="night-runner:latest",
        num_ctx=24576,
        num_predict=8192,
        from_id="GRAPH-047.1",
        to_id="GRAPH-047.1",
        workspace_root=tmp_path,
        max_tool_steps=80,
        max_subtask_attempts=5,
        prompt_variant="baseline",
    )
    run_metrics.begin_attempt("GRAPH-047", "GRAPH-047.1", "test_writer", 1)
    run_metrics.record_chat(step=1, chat_seconds=2.5, prompt_eval_count=1000, eval_count=50)
    run_metrics.record_tool("read_file", "ok", {"path": "src/a.py"})
    run_metrics.record_chat(step=2, chat_seconds=3.0, prompt_eval_count=1200, eval_count=80)
    run_metrics.record_tool("read_file", "ok", {"path": "src/a.py"})
    run_metrics.record_tool("write_file", "error", {"path": "tests/t.py", "content": "x"})
    run_metrics.record_handover()
    run_metrics.end_attempt(ok=False, stop_reason="tool_step_budget")

    run_metrics.begin_attempt("GRAPH-047", "GRAPH-047.1", "implementer", 1)
    run_metrics.record_chat(step=1, chat_seconds=1.0, prompt_eval_count=900, eval_count=20)
    run_metrics.record_tool("run_command", "ok", {"command": "pytest -q"})
    run_metrics.end_attempt(ok=True, stop_reason="ok")
    run_metrics.record_verifier(
        "GRAPH-047",
        "GRAPH-047.1",
        passed=True,
        commands=[{"command": "pytest -q", "passed": True}],
    )
    metrics.finish(0)
    return metrics


def test_run_summary_schema_and_totals(tmp_path: Path) -> None:
    try:
        metrics = _sample_run(tmp_path)
        path = metrics.write(tmp_path / "out" / "run_summary.json")
    finally:
        run_metrics.clear()

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema"] == run_metrics.SCHEMA_VERSION
    assert data["model"] == "night-runner:latest"
    assert data["promptVariant"] == "baseline"
    assert data["exitCode"] == 0
    assert data["range"] == {"fromId": "GRAPH-047.1", "toId": "GRAPH-047.1"}

    totals = data["totals"]
    assert totals["attempts"] == 2
    assert totals["steps"] == 3
    assert totals["toolCalls"] == 4
    assert totals["toolErrors"] == 1
    assert totals["repeatedIdenticalCalls"] == 1  # read_file src/a.py twice
    assert totals["promptTokens"] == 3100
    assert totals["completionTokens"] == 150
    assert totals["handovers"] == 1

    first, second = data["attempts"]
    assert first["role"] == "test_writer"
    assert first["ok"] is False
    assert first["stopReason"] == "tool_step_budget"
    assert first["maxRepeatOfOneCall"] == 2
    assert first["wallSeconds"] is not None
    assert len(first["stepLog"]) == 2
    assert first["stepLog"][1]["tools"][1]["status"] == "error"
    assert second["ok"] is True

    assert data["verifierResults"][0]["passed"] is True


def test_record_calls_are_noops_without_active_run() -> None:
    run_metrics.clear()
    run_metrics.begin_attempt("X-1", "X-1.1", "implementer", 1)
    run_metrics.record_chat(step=1, chat_seconds=1.0, prompt_eval_count=1, eval_count=1)
    run_metrics.record_tool("read_file", "ok", {})
    run_metrics.record_handover()
    run_metrics.end_attempt(ok=True, stop_reason="ok")
    run_metrics.record_verifier("X-1", "X-1.1", passed=True, commands=[])
    assert run_metrics.active() is None


def test_prompt_variant_defaults_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_AGENT_PROMPT_VARIANT", "v2")
    try:
        metrics = run_metrics.start_run(
            model="m",
            num_ctx=1,
            num_predict=1,
            from_id="A-1",
            to_id="A-1",
            workspace_root=tmp_path,
            max_tool_steps=1,
            max_subtask_attempts=1,
        )
        assert metrics.prompt_variant == "v2"
        assert metrics.workspace_head == ""  # tmp_path is not a git repo
    finally:
        run_metrics.clear()


def test_cli_accepts_run_summary_flag() -> None:
    args = parse_args(["--run-summary", "out/run.json", "--from", "GRAPH-047.1", "--to", "GRAPH-047.1"])
    assert args.run_summary == "out/run.json"
    default_args = parse_args([])
    assert default_args.run_summary == ""
