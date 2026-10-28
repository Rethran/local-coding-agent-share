"""Deterministic worker tool allowlists.

The worker case is already known from the role, the context pack, and session
flags. Schemas sent to Ollama are a subset of ``TOOLS``. ``TOOL_FUNCTIONS``
stays complete so a stale call can still dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orchestrator.tools_registry import TOOLS

EDIT_TOOL_NAMES: tuple[str, ...] = (
    "read_file",
    "replace_in_file",
    "write_file",
    "run_command",
    "update_subtask_state",
)

TEST_WRITE_TOOL_NAMES: tuple[str, ...] = EDIT_TOOL_NAMES + ("list_directory",)

EXPLORE_TOOL_NAMES: tuple[str, ...] = EDIT_TOOL_NAMES + (
    "search_files",
    "find_symbol",
    "extract_symbol",
    "list_directory",
    "get_subtask_context",
)

_CASE_TOOL_NAMES = {
    "edit": EDIT_TOOL_NAMES,
    "test_write": TEST_WRITE_TOOL_NAMES,
    "explore": EXPLORE_TOOL_NAMES,
}

_TOOLS_BY_NAME = {tool["function"]["name"]: tool for tool in TOOLS}


@dataclass(frozen=True)
class WorkerSessionFlags:
    exploration_budget_exhausted: bool = False
    has_successful_write: bool = False
    pending_truncated_handoff: bool = False


def session_flags_from(session: Any, task_id: str) -> WorkerSessionFlags:
    """Read the flags that can switch explore to edit between worker steps."""
    if session is None:
        return WorkerSessionFlags()
    task = session._task(task_id)
    pending = bool(set(task.truncated_handoff_ids) - set(task.expanded_handoff_ids))
    exhausted = (
        not task.has_successful_write
        and task.read_only_tools_used >= task.max_read_only_before_write
    )
    return WorkerSessionFlags(
        exploration_budget_exhausted=exhausted,
        has_successful_write=bool(task.has_successful_write),
        pending_truncated_handoff=pending,
    )


def select_worker_case(
    role: str,
    context_pack: dict[str, Any] | None,
    session_flags: WorkerSessionFlags | None = None,
) -> str:
    flags = session_flags or WorkerSessionFlags()
    if role == "test_writer":
        return "test_write"
    policy: dict[str, Any] = {}
    if isinstance(context_pack, dict):
        raw_policy = context_pack.get("explorationPolicy") or {}
        if isinstance(raw_policy, dict):
            policy = raw_policy
    preflight_satisfied = bool(policy.get("preflightSatisfied"))
    if (
        not preflight_satisfied
        and not flags.exploration_budget_exhausted
        and not flags.has_successful_write
    ):
        return "explore"
    return "edit"


def select_worker_tools(
    role: str,
    context_pack: dict[str, Any] | None,
    session_flags: WorkerSessionFlags | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(case_name, tool_schemas)`` for one worker step."""
    flags = session_flags or WorkerSessionFlags()
    case_name = select_worker_case(role, context_pack, flags)
    names = list(_CASE_TOOL_NAMES[case_name])
    if flags.pending_truncated_handoff and "read_handoff" not in names:
        names.append("read_handoff")
    schemas = [_TOOLS_BY_NAME[name] for name in names if name in _TOOLS_BY_NAME]
    return case_name, schemas
