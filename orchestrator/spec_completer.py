"""Deterministic repair of weak subtask verification commands."""

from __future__ import annotations

import json
from typing import Any

from orchestrator.planner import load_plan_sidecar, save_plan_sidecar
from orchestrator.verification_spec import (
    add_slice_selector,
    executable_commands,
    has_slice_selector,
    is_executable_command,
    shared_test_file_siblings,
    sibling_subtasks,
    slice_token,
    verification_lists_match,
)
from tools import _tasks


def complete_subtask_verification(
    task: dict[str, Any],
    subtask_id: str,
    merged_subtask: dict[str, Any],
) -> list[str] | None:
    """Return a repaired verification list when uniqueness or slice selectors are missing."""
    current = [
        str(item).strip()
        for item in merged_subtask.get("verification") or []
        if isinstance(item, str) and str(item).strip()
    ]
    commands = executable_commands(current)
    if not commands:
        return None

    token = slice_token(subtask_id)
    needs_slice = shared_test_file_siblings(task, merged_subtask, subtask_id) and not all(
        has_slice_selector(command) for command in commands
    )
    matches_sibling = False
    for sibling in sibling_subtasks(task, subtask_id):
        if verification_lists_match(commands, executable_commands(sibling.get("verification"))):
            matches_sibling = True
            break

    if not needs_slice and not matches_sibling:
        return None

    repaired = [
        add_slice_selector(item, token) if is_executable_command(item) else item
        for item in current
    ]
    if repaired == current:
        return None
    return repaired


def persist_verification_repair(
    task_id: str,
    subtask_id: str,
    verification: list[str],
) -> str:
    """Save sidecar enrichments and write verification back into the requirement JSON."""
    existing = load_plan_sidecar(task_id, subtask_id) or {}
    enrichments = dict(existing.get("enrichments") or {})
    enrichments["verification"] = verification
    sidecar_path = save_plan_sidecar(
        {
            "taskId": task_id,
            "subtaskId": subtask_id,
            "plannerRound": existing.get("plannerRound", 0),
            "enrichments": enrichments,
            "addressedRuleIds": ["GR-SUB-004", "GR-SUB-006"],
            "source": "spec_completer",
        }
    )
    write_subtask_verification(task_id, subtask_id, verification)
    return sidecar_path


def write_subtask_verification(
    task_id: str,
    subtask_id: str,
    verification: list[str],
) -> None:
    """Patch only the subtask verification array in the requirement JSON."""
    path = _tasks.task_path(task_id)
    task = json.loads(path.read_text(encoding="utf-8"))
    changed = False
    for item in task.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            item["verification"] = list(verification)
            changed = True
            break
    if not changed:
        return
    path.write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
