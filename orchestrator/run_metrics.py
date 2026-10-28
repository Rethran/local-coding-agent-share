"""Process-wide run summary collector.

The orchestrator already logs steps, token counts and latencies to stdout.
Evals need those numbers in a machine-readable file so that scorers can read
them without scraping logs. This module collects them in memory and writes
one JSON document at the end of a run when ``--run-summary PATH`` is given.

The collector is deliberately passive: when no run is active every ``record_*``
call is a no-op, so the orchestrator behaves exactly as before.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.prompt_variants import active_variant_name

SCHEMA_VERSION = "run-summary/v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tool_fingerprint(name: str, arguments: dict[str, Any] | None) -> str:
    """Stable identity for one tool call, used to detect repeated identical calls."""
    try:
        encoded = json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False)
    except TypeError:
        encoded = str(arguments)
    return f"{name}:{encoded}"


@dataclass
class StepRecord:
    step: int
    chat_seconds: float
    prompt_eval_count: int | None
    eval_count: int | None
    tools: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "chatSeconds": round(self.chat_seconds, 3),
            "promptEvalCount": self.prompt_eval_count,
            "evalCount": self.eval_count,
            "tools": self.tools,
        }


@dataclass
class AttemptRecord:
    task_id: str
    subtask_id: str
    role: str
    attempt: int
    started_at: str = field(default_factory=_utc_now_iso)
    started_perf: float = field(default_factory=time.perf_counter)
    finished_at: str | None = None
    wall_seconds: float | None = None
    ok: bool | None = None
    stop_reason: str = ""
    handovers: int = 0
    steps: list[StepRecord] = field(default_factory=list)

    def finish(self, *, ok: bool, stop_reason: str) -> None:
        self.finished_at = _utc_now_iso()
        self.wall_seconds = time.perf_counter() - self.started_perf
        self.ok = ok
        self.stop_reason = stop_reason

    def summary(self) -> dict[str, Any]:
        fingerprints: dict[str, int] = {}
        tool_calls = 0
        tool_errors = 0
        for step in self.steps:
            for tool in step.tools:
                tool_calls += 1
                if tool.get("status") != "ok":
                    tool_errors += 1
                fingerprint = tool.get("fingerprint") or ""
                fingerprints[fingerprint] = fingerprints.get(fingerprint, 0) + 1
        repeated = {fp: n for fp, n in fingerprints.items() if n > 1}
        return {
            "steps": len(self.steps),
            "toolCalls": tool_calls,
            "toolErrors": tool_errors,
            "repeatedIdenticalCalls": sum(n - 1 for n in repeated.values()),
            "maxRepeatOfOneCall": max(repeated.values(), default=1),
            "promptTokens": sum(s.prompt_eval_count or 0 for s in self.steps),
            "completionTokens": sum(s.eval_count or 0 for s in self.steps),
            "chatSeconds": round(sum(s.chat_seconds for s in self.steps), 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "subtaskId": self.subtask_id,
            "role": self.role,
            "attempt": self.attempt,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "wallSeconds": round(self.wall_seconds, 3) if self.wall_seconds is not None else None,
            "ok": self.ok,
            "stopReason": self.stop_reason,
            "handovers": self.handovers,
            **self.summary(),
            "stepLog": [step.to_dict() for step in self.steps],
        }


@dataclass
class VerifierRecord:
    task_id: str
    subtask_id: str
    passed: bool
    commands: list[dict[str, Any]]
    recorded_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "taskId": self.task_id,
            "subtaskId": self.subtask_id,
            "passed": self.passed,
            "commands": self.commands,
            "recordedAt": self.recorded_at,
        }


@dataclass
class RunMetrics:
    model: str
    num_ctx: int
    num_predict: int
    prompt_variant: str
    from_id: str
    to_id: str
    workspace_root: str
    workspace_head: str
    max_tool_steps: int
    max_subtask_attempts: int
    started_at: str = field(default_factory=_utc_now_iso)
    started_perf: float = field(default_factory=time.perf_counter)
    finished_at: str | None = None
    exit_code: int | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    verifier_results: list[VerifierRecord] = field(default_factory=list)
    _current: AttemptRecord | None = None
    _current_step: StepRecord | None = None

    def begin_attempt(self, task_id: str, subtask_id: str, role: str, attempt: int) -> None:
        self._current = AttemptRecord(task_id, subtask_id, role, attempt)
        self._current_step = None
        self.attempts.append(self._current)

    def record_chat(
        self,
        *,
        step: int,
        chat_seconds: float,
        prompt_eval_count: int | None,
        eval_count: int | None,
    ) -> None:
        if self._current is None:
            return
        self._current_step = StepRecord(step, chat_seconds, prompt_eval_count, eval_count)
        self._current.steps.append(self._current_step)

    def record_tool(self, name: str, status: str, arguments: dict[str, Any] | None) -> None:
        if self._current is None or self._current_step is None:
            return
        self._current_step.tools.append({
            "name": name,
            "status": status,
            "fingerprint": tool_fingerprint(name, arguments),
        })

    def record_handover(self) -> None:
        if self._current is not None:
            self._current.handovers += 1

    def end_attempt(self, *, ok: bool, stop_reason: str) -> None:
        if self._current is None:
            return
        self._current.finish(ok=ok, stop_reason=stop_reason)
        self._current = None
        self._current_step = None

    def record_verifier(
        self,
        task_id: str,
        subtask_id: str,
        *,
        passed: bool,
        commands: list[dict[str, Any]],
    ) -> None:
        self.verifier_results.append(VerifierRecord(task_id, subtask_id, passed, commands))

    def finish(self, exit_code: int) -> None:
        self.finished_at = _utc_now_iso()
        self.exit_code = exit_code

    def totals(self) -> dict[str, Any]:
        summaries = [attempt.summary() for attempt in self.attempts]
        return {
            "attempts": len(self.attempts),
            "steps": sum(s["steps"] for s in summaries),
            "toolCalls": sum(s["toolCalls"] for s in summaries),
            "toolErrors": sum(s["toolErrors"] for s in summaries),
            "repeatedIdenticalCalls": sum(s["repeatedIdenticalCalls"] for s in summaries),
            "promptTokens": sum(s["promptTokens"] for s in summaries),
            "completionTokens": sum(s["completionTokens"] for s in summaries),
            "chatSeconds": round(sum(s["chatSeconds"] for s in summaries), 3),
            "handovers": sum(a.handovers for a in self.attempts),
            "wallSeconds": round(time.perf_counter() - self.started_perf, 3),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "model": self.model,
            "promptVariant": self.prompt_variant,
            "numCtx": self.num_ctx,
            "numPredict": self.num_predict,
            "maxToolSteps": self.max_tool_steps,
            "maxSubtaskAttempts": self.max_subtask_attempts,
            "range": {"fromId": self.from_id, "toId": self.to_id},
            "workspaceRoot": self.workspace_root,
            "workspaceHead": self.workspace_head,
            "startedAt": self.started_at,
            "finishedAt": self.finished_at,
            "exitCode": self.exit_code,
            "totals": self.totals(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "verifierResults": [result.to_dict() for result in self.verifier_results],
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


_ACTIVE: RunMetrics | None = None


def git_head(workspace_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def start_run(
    *,
    model: str,
    num_ctx: int,
    num_predict: int,
    from_id: str,
    to_id: str,
    workspace_root: Path,
    max_tool_steps: int,
    max_subtask_attempts: int,
    prompt_variant: str | None = None,
) -> RunMetrics:
    global _ACTIVE
    _ACTIVE = RunMetrics(
        model=model,
        num_ctx=num_ctx,
        num_predict=num_predict,
        prompt_variant=prompt_variant or active_variant_name(),
        from_id=from_id,
        to_id=to_id,
        workspace_root=str(workspace_root),
        workspace_head=git_head(workspace_root),
        max_tool_steps=max_tool_steps,
        max_subtask_attempts=max_subtask_attempts,
    )
    return _ACTIVE


def active() -> RunMetrics | None:
    return _ACTIVE


def clear() -> None:
    global _ACTIVE
    _ACTIVE = None


# Convenience no-op-safe wrappers used by worker/loop so call sites stay short.


def begin_attempt(task_id: str, subtask_id: str, role: str, attempt: int) -> None:
    if _ACTIVE is not None:
        _ACTIVE.begin_attempt(task_id, subtask_id, role, attempt)


def record_chat(
    *,
    step: int,
    chat_seconds: float,
    prompt_eval_count: int | None,
    eval_count: int | None,
) -> None:
    if _ACTIVE is not None:
        _ACTIVE.record_chat(
            step=step,
            chat_seconds=chat_seconds,
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )


def record_tool(name: str, status: str, arguments: dict[str, Any] | None) -> None:
    if _ACTIVE is not None:
        _ACTIVE.record_tool(name, status, arguments)


def record_handover() -> None:
    if _ACTIVE is not None:
        _ACTIVE.record_handover()


def end_attempt(*, ok: bool, stop_reason: str) -> None:
    if _ACTIVE is not None:
        _ACTIVE.end_attempt(ok=ok, stop_reason=stop_reason)


def record_verifier(
    task_id: str,
    subtask_id: str,
    *,
    passed: bool,
    commands: list[dict[str, Any]],
) -> None:
    if _ACTIVE is not None:
        _ACTIVE.record_verifier(task_id, subtask_id, passed=passed, commands=commands)
