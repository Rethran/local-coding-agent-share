"""Code-only graders for one isolated TDD-coder sample.

These functions are independent of Inspect AI so unit tests and the fallback
runner can call them. The Inspect wrappers in tdd_coder_task.py are thin.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from evals.criteria import PROCESS_BUDGET
from evals.isolation import IsolatedRun, load_json, overlay_reference_tests

PYTEST_TIMEOUT = 180


def subtask_status(state: dict[str, Any], subtask_id: str) -> str:
    for item in state.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return str(item.get("status") or "")
    return str(state.get("status") or "")


def score_subtask_done(state: dict[str, Any], subtask_id: str) -> tuple[bool, str]:
    status = subtask_status(state, subtask_id)
    if status == "DONE":
        return True, f"Subtask {subtask_id} is DONE."
    reason = state.get("lastFailureReason") or state.get("summary") or "no failure reason"
    return False, f"Subtask {subtask_id} status={status or 'missing'}. {reason}"


def score_process_budget(
    summary: dict[str, Any],
    thresholds: dict[str, Any] | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    limits = thresholds or PROCESS_BUDGET
    totals = summary.get("totals") or {}
    attempts = int(totals.get("attempts") or 0)
    steps = int(totals.get("steps") or 0)
    tool_calls = int(totals.get("toolCalls") or 0)
    tool_errors = int(totals.get("toolErrors") or 0)
    max_repeat = 1
    for attempt in summary.get("attempts") or []:
        max_repeat = max(max_repeat, int(attempt.get("maxRepeatOfOneCall") or 1))
    error_rate = (tool_errors / tool_calls) if tool_calls else 0.0
    checks = {
        "attempts": {
            "value": attempts,
            "limit": limits["max_attempts"],
            "ok": attempts <= limits["max_attempts"],
        },
        "steps": {
            "value": steps,
            "limit": limits["max_steps"],
            "ok": steps <= limits["max_steps"],
        },
        "maxRepeatOfOneCall": {
            "value": max_repeat,
            "limit": limits["max_repeat_of_one_call"],
            "ok": max_repeat <= limits["max_repeat_of_one_call"],
        },
        "toolErrorRate": {
            "value": round(error_rate, 3),
            "limit": limits["max_tool_error_rate"],
            "ok": error_rate <= limits["max_tool_error_rate"],
        },
    }
    failed = [name for name, row in checks.items() if not row["ok"]]
    if not failed:
        return True, "Process budget held: " + ", ".join(
            f"{name}={row['value']}/{row['limit']}" for name, row in checks.items()
        ), checks
    explanation = "Process budget failed: " + "; ".join(
        f"{name} {checks[name]['value']} > {checks[name]['limit']}" for name in failed
    )
    return False, explanation, checks


def score_reference_tests(
    isolated: IsolatedRun,
    *,
    repo: Path,
    accepted_commit: str,
    reference_tests: list[str],
) -> tuple[bool, str]:
    if not reference_tests:
        return False, "No tests/ files in the accepted commit; cannot score reference tests."
    overlaid = overlay_reference_tests(
        isolated.worktree, repo, accepted_commit, reference_tests
    )
    if not overlaid:
        return False, f"Could not overlay gold tests from {accepted_commit}."
    test_paths = [path for path in overlaid if path.endswith(".py")]
    if not test_paths:
        return False, f"Accepted commit changed fixtures only: {overlaid}"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(isolated.worktree), str(isolated.worktree / "src"), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", *test_paths],
        cwd=isolated.worktree,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=PYTEST_TIMEOUT,
        check=False,
    )
    output = ((completed.stdout or "") + "\n" + (completed.stderr or ""))[-2000:]
    if completed.returncode == 0:
        return True, f"Gold tests passed ({len(test_paths)} files).\n{output}"
    return False, f"Gold tests failed (exit {completed.returncode}).\n{output}"


def score_isolated_run(
    isolated: IsolatedRun,
    *,
    subtask_id: str,
    repo: Path,
    accepted_commit: str,
    reference_tests: list[str],
) -> dict[str, dict[str, Any]]:
    state = load_json(isolated.state_path)
    summary = load_json(isolated.run_summary_path)
    done_ok, done_why = score_subtask_done(state, subtask_id)
    budget_ok, budget_why, budget_checks = score_process_budget(summary)
    if isolated.timed_out:
        done_ok = False
        done_why = f"Timed out after {isolated.wall_seconds:.0f}s. {done_why}"
    try:
        tests_ok, tests_why = score_reference_tests(
            isolated,
            repo=repo,
            accepted_commit=accepted_commit,
            reference_tests=reference_tests,
        )
    except subprocess.TimeoutExpired:
        tests_ok, tests_why = False, "Gold pytest timed out."
    tokens = int((summary.get("totals") or {}).get("promptTokens") or 0) + int(
        (summary.get("totals") or {}).get("completionTokens") or 0
    )
    return {
        "subtask_done": {"ok": done_ok, "explanation": done_why},
        "reference_tests_pass": {"ok": tests_ok, "explanation": tests_why},
        "process_budget": {
            "ok": budget_ok,
            "explanation": budget_why,
            "checks": budget_checks,
        },
        "metadata": {
            "wallSeconds": isolated.wall_seconds,
            "exitCode": isolated.exit_code,
            "timedOut": isolated.timed_out,
            "tokens": tokens,
            "promptVariant": summary.get("promptVariant"),
            "model": summary.get("model"),
            "totals": summary.get("totals") or {},
        },
    }
