"""Per-attempt worker context for orchestrator guardrails."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

_current_task_id: ContextVar[str | None] = ContextVar("worker_task_id", default=None)


@dataclass
class WorkerAttemptContext:
    task_id: str
    role: str = "implementer"
    merged_subtask: dict[str, Any] | None = None
    seen_searches: set[tuple[str, str]] = field(default_factory=set)
    text_only_retries: int = 0
    syntax_write_retries: int = 0


_current_attempt: ContextVar[WorkerAttemptContext | None] = ContextVar(
    "worker_attempt_context",
    default=None,
)


def set_worker_task_id(task_id: str) -> None:
    _current_task_id.set(task_id)


def get_worker_task_id() -> str | None:
    return _current_task_id.get()


def begin_worker_attempt(
    task_id: str,
    *,
    role: str = "implementer",
    merged_subtask: dict[str, Any] | None = None,
) -> WorkerAttemptContext:
    ctx = WorkerAttemptContext(
        task_id=task_id,
        role=role,
        merged_subtask=merged_subtask,
    )
    set_worker_task_id(task_id)
    _current_attempt.set(ctx)
    return ctx


def get_worker_attempt() -> WorkerAttemptContext | None:
    return _current_attempt.get()


def clear_worker_context() -> None:
    _current_task_id.set(None)
    _current_attempt.set(None)
