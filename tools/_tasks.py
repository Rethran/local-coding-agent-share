"""Task graph loading, state I/O, and READY/blocked computation for HEAT tasks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.project import PROJECT_PROFILE

REQUIREMENTS_DIR = PROJECT_PROFILE.requirements_dir
TASK_PREFIX = PROJECT_PROFILE.task_prefix

VALID_STATUSES = frozenset({
    "PENDING",
    "READY",
    "IN_PROGRESS",
    "VERIFYING",
    "DONE",
    "FAILED",
    "BLOCKED",
})

VALID_SUBTASK_STATUSES = frozenset({
    "PENDING",
    "IN_PROGRESS",
    "DONE",
    "FAILED",
    "BLOCKED",
})
def _task_id_pattern() -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(TASK_PREFIX)}-(\d+)$")


def _task_ref_pattern() -> re.Pattern[str]:
    return re.compile(rf"^({re.escape(TASK_PREFIX)}-(\d+))(?:\.(\d+))?$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_heat_ref(value: str) -> dict[str, str | None]:
    """Parse TASK-003 or TASK-003.4 into parent task + optional subtask id."""
    raw = (value or "").strip()
    match = _task_ref_pattern().match(raw)
    if not match:
        raise ValueError(
            f"Invalid {TASK_PREFIX} id: {value!r}. "
            f"Expected {TASK_PREFIX}-003 or {TASK_PREFIX}-003.4."
        )
    task_id = match.group(1)
    suffix = match.group(3)
    subtask_id = f"{task_id}.{suffix}" if suffix is not None else None
    return {
        "raw": raw,
        "taskId": task_id,
        "subtaskId": subtask_id,
    }


def parent_task_id(value: str) -> str:
    """Return the parent task id for a task or subtask ref."""
    return str(parse_heat_ref(value)["taskId"])


def parse_task_number(task_id: str) -> int:
    """Numeric sort key for TASK-003 or TASK-003.4 (uses parent task number)."""
    parent = parent_task_id(task_id)
    match = _task_id_pattern().match(parent)
    if not match:
        raise ValueError(f"Invalid task id: {task_id}")
    return int(match.group(1))


def requirements_dir() -> Path:
    return REQUIREMENTS_DIR


def state_path(task_id: str) -> Path:
    return REQUIREMENTS_DIR / f"{task_id}.state.json"


def task_path(task_id: str) -> Path:
    return REQUIREMENTS_DIR / f"{task_id}.json"


def load_task_definitions() -> dict[str, dict[str, Any]]:
    """Load all configured task definitions (excludes *.state.json)."""
    tasks: dict[str, dict[str, Any]] = {}
    if not REQUIREMENTS_DIR.exists():
        return tasks

    for path in sorted(REQUIREMENTS_DIR.glob(f"{TASK_PREFIX}-*.json")):
        if path.name.endswith(".state.json"):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        task_id = data.get("id") or path.stem
        tasks[task_id] = data

    return tasks


def load_task_state(task_id: str) -> dict[str, Any] | None:
    path = state_path(task_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_states(task_ids: list[str]) -> dict[str, dict[str, Any]]:
    return {
        task_id: state
        for task_id in task_ids
        if (state := load_task_state(task_id)) is not None
    }


def effective_status(task_id: str, states: dict[str, dict[str, Any]]) -> str:
    state = states.get(task_id)
    if state is None:
        return "PENDING"
    status = state.get("status", "PENDING")
    return status if status in VALID_STATUSES else "PENDING"


def deps_satisfied(task: dict[str, Any], states: dict[str, dict[str, Any]]) -> bool:
    for dep_id in task.get("dependsOn", []):
        if effective_status(dep_id, states) != "DONE":
            return False
    return True


def filter_task_range(
    tasks: dict[str, dict[str, Any]],
    from_id: str | None,
    to_id: str | None,
) -> dict[str, dict[str, Any]]:
    """Filter by parent task number. Subtask refs use their parent task id."""
    if from_id is None and to_id is None:
        return tasks

    from_num = parse_task_number(from_id) if from_id else None
    to_num = parse_task_number(to_id) if to_id else None

    filtered: dict[str, dict[str, Any]] = {}
    for task_id, task in tasks.items():
        num = parse_task_number(task_id)
        if from_num is not None and num < from_num:
            continue
        if to_num is not None and num > to_num:
            continue
        filtered[task_id] = task

    return filtered


def _subtask_index(subtask_id: str) -> int:
    ref = parse_heat_ref(subtask_id)
    if ref["subtaskId"] is None:
        raise ValueError(f"Expected subtask id with suffix: {subtask_id}")
    return int(str(ref["subtaskId"]).rsplit(".", 1)[1])


def ensure_subtask_exists(task_id: str, subtask_id: str) -> None:
    packet = get_task_packet(task_id)
    derived = derive_subtasks(task_id, packet["task"])
    known = {item["id"] for item in derived}
    if subtask_id not in known:
        known_list = ", ".join(sorted(known)) or "(none)"
        raise ValueError(
            f"Subtask {subtask_id} not found on {task_id}. Known: {known_list}"
        )


def pin_from_subtask(task_id: str, subtask_id: str) -> dict[str, Any]:
    """Force active work to start at subtask_id (earlier subtasks marked DONE)."""
    ensure_subtask_exists(task_id, subtask_id)
    packet = get_task_packet(task_id)
    existing = packet["state"] or {}
    normalized = normalize_subtask_state(task_id, packet["task"], existing)
    target_index = _subtask_index(subtask_id)

    subtasks: list[dict[str, Any]] = []
    for item in normalized["subtasks"]:
        updated = dict(item)
        item_index = _subtask_index(str(updated["id"]))
        if item_index < target_index:
            if updated.get("status") != "DONE":
                updated["status"] = "DONE"
                if not updated.get("summary"):
                    updated["summary"] = (
                        f"Marked DONE by orchestrator pin (--from {subtask_id})."
                    )
        elif updated["id"] == subtask_id:
            updated["status"] = "IN_PROGRESS"
            if not updated.get("summary"):
                updated["summary"] = f"Pinned active by orchestrator (--from {subtask_id})."
        subtasks.append(updated)

    return update_task_state(
        task_id,
        "IN_PROGRESS",
        summary=f"Orchestrator pinned active subtask to {subtask_id}.",
        subtasks=subtasks,
        active_subtask_id=subtask_id,
        progress_summary=f"Active subtask is {subtask_id}.",
        recommended_resume_step=f"Resume from {subtask_id}.",
    )


def skip_subtasks_after(task_id: str, to_subtask_id: str) -> dict[str, Any]:
    """Mark subtasks after to_subtask_id as DONE so --to HEAT-N.M stops there."""
    ensure_subtask_exists(task_id, to_subtask_id)
    packet = get_task_packet(task_id)
    existing = packet["state"] or {}
    normalized = normalize_subtask_state(task_id, packet["task"], existing)
    stop_index = _subtask_index(to_subtask_id)

    subtasks: list[dict[str, Any]] = []
    for item in normalized["subtasks"]:
        updated = dict(item)
        item_index = _subtask_index(str(updated["id"]))
        if item_index > stop_index and updated.get("status") != "DONE":
            updated["status"] = "DONE"
            updated["summary"] = (
                f"Skipped by orchestrator (--to {to_subtask_id}); outside selected range."
            )
        subtasks.append(updated)

    active = next((item for item in subtasks if item["status"] != "DONE"), None)
    current = effective_status(task_id, {task_id: existing})
    if active is None:
        next_status = "DONE" if current == "DONE" else current
    elif current in ("DONE", "PENDING", "READY"):
        next_status = "IN_PROGRESS"
    else:
        next_status = current

    return update_task_state(
        task_id,
        next_status,
        summary=existing.get("summary") or f"Range limited with --to {to_subtask_id}.",
        subtasks=subtasks,
        active_subtask_id=active["id"] if active else existing.get("activeSubtaskId"),
    )


def apply_cli_heat_range(
    from_id: str | None,
    to_id: str | None,
) -> dict[str, Any]:
    """
    Normalize --from/--to that may include subtask refs.

    Examples:
      --from TASK-003.4 --to TASK-004
      --from TASK-003.4 --to TASK-003.6
    """
    from_ref = parse_heat_ref(from_id) if from_id else None
    to_ref = parse_heat_ref(to_id) if to_id else None

    result: dict[str, Any] = {
        "from": from_ref,
        "to": to_ref,
        "fromTaskId": from_ref["taskId"] if from_ref else None,
        "toTaskId": to_ref["taskId"] if to_ref else None,
        "fromSubtaskId": from_ref["subtaskId"] if from_ref else None,
        "toSubtaskId": to_ref["subtaskId"] if to_ref else None,
        "pinned": None,
        "skipped": None,
    }

    if from_ref and from_ref["subtaskId"]:
        result["pinned"] = pin_from_subtask(
            str(from_ref["taskId"]),
            str(from_ref["subtaskId"]),
        )

    if to_ref and to_ref["subtaskId"]:
        result["skipped"] = skip_subtasks_after(
            str(to_ref["taskId"]),
            str(to_ref["subtaskId"]),
        )

    return result


def build_task_graph(
    from_id: str | None = None,
    to_id: str | None = None,
) -> dict[str, Any]:
    """Build task list with computed ready/done/blocked sets."""
    all_tasks = load_task_definitions()
    tasks = filter_task_range(all_tasks, from_id, to_id)
    task_ids = sorted(tasks.keys(), key=parse_task_number)
    all_task_ids = sorted(all_tasks.keys(), key=parse_task_number)
    states = load_all_states(all_task_ids)

    task_entries = []
    ready: list[str] = []
    done: list[str] = []
    blocked: list[str] = []

    for task_id in task_ids:
        task = tasks[task_id]
        status = effective_status(task_id, states)

        if status == "DONE":
            done.append(task_id)
        elif status in ("IN_PROGRESS", "VERIFYING"):
            pass
        elif deps_satisfied(task, states):
            ready.append(task_id)
            if status == "PENDING":
                status = "READY"
        else:
            blocked.append(task_id)

        task_entries.append({
            "id": task_id,
            "status": status,
            "dependsOn": task.get("dependsOn", []),
            "summary": task.get("summary", ""),
        })

    next_action = None
    if ready:
        next_action = {
            "tool": "local_agent_get_task",
            "arguments": {
                "task_id": ready[0],
            },
            "instruction": (
                f"Call local_agent_get_task natively for {ready[0]} only. "
                "Wait for the result before calling any other task tool."
            ),
        }

    return {
        "tasks": task_entries,
        "ready": ready,
        "done": done,
        "blocked": blocked,
        "nextAction": next_action,
    }


def select_next_task_id(
    from_id: str | None = None,
    to_id: str | None = None,
) -> dict[str, Any]:
    """Prefer in-progress work, otherwise choose the first READY task."""
    graph = build_task_graph(from_id, to_id)

    active = [
        task["id"]
        for task in graph["tasks"]
        if task["status"] in ("IN_PROGRESS", "VERIFYING")
    ]
    if active:
        return {
            "taskId": active[0],
            "reason": "active",
            "graph": graph,
        }

    if graph["ready"]:
        return {
            "taskId": graph["ready"][0],
            "reason": "ready",
            "graph": graph,
        }

    return {
        "taskId": None,
        "reason": "none",
        "graph": graph,
    }


def get_task_packet(task_id: str) -> dict[str, Any]:
    path = task_path(task_id)
    if not path.exists():
        raise ValueError(f"Task not found: {task_id}")

    task = json.loads(path.read_text(encoding="utf-8"))
    state = load_task_state(task_id)
    return {
        "task": task,
        "state": state,
    }


def has_explicit_subtasks(task: dict[str, Any]) -> bool:
    explicit = task.get("subtasks")
    return isinstance(explicit, list) and bool(explicit)


def get_subtask_spec_definition(task_id: str, subtask_id: str) -> dict[str, Any] | None:
    packet = get_task_packet(task_id)
    task = packet["task"]
    for item in task.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return item
    for item in derive_subtasks(task_id, task):
        if item.get("id") == subtask_id:
            scope_line = item.get("title") or item.get("summary") or ""
            return {
                **item,
                "scope": [scope_line] if scope_line else [],
            }
    return None


def derive_subtasks(task_id: str, task: dict[str, Any]) -> list[dict[str, Any]]:
    """Build stable default subtasks from explicit task data or scope items."""
    explicit = task.get("subtasks")
    items = explicit if isinstance(explicit, list) and explicit else task.get("scope", [])

    subtasks: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        default_id = f"{task_id}.{index}"
        if isinstance(item, dict):
            subtask_id = str(item.get("id") or default_id)
            title = str(item.get("title") or item.get("summary") or subtask_id)
            summary = str(item.get("summary") or "")
        else:
            subtask_id = default_id
            title = str(item)
            summary = ""

        subtasks.append({
            "id": subtask_id,
            "title": title,
            "status": "PENDING",
            "summary": summary,
        })

    return subtasks


def normalize_subtask_state(
    task_id: str,
    task: dict[str, Any],
    state: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge persisted subtask progress onto subtasks derived from task data."""
    state = state or {}
    existing = {
        str(item.get("id")): item
        for item in state.get("subtasks", [])
        if isinstance(item, dict) and item.get("id")
    }

    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for subtask in derive_subtasks(task_id, task):
        subtask_id = subtask["id"]
        persisted = existing.get(subtask_id, {})
        status = persisted.get("status", subtask["status"])
        if status not in VALID_SUBTASK_STATUSES:
            status = subtask["status"]
        normalized.append({
            **subtask,
            "status": status,
            "summary": persisted.get("summary") or subtask["summary"],
        })
        seen_ids.add(subtask_id)

    for subtask_id, persisted in existing.items():
        if subtask_id in seen_ids:
            continue
        status = persisted.get("status", "PENDING")
        if status not in VALID_SUBTASK_STATUSES:
            status = "PENDING"
        normalized.append({
            "id": subtask_id,
            "title": str(persisted.get("title") or subtask_id),
            "status": status,
            "summary": persisted.get("summary", ""),
        })

    active_subtask_id = state.get("activeSubtaskId")
    active = next((item for item in normalized if item["id"] == active_subtask_id), None)
    if active is None or active["status"] == "DONE":
        active = next((item for item in normalized if item["status"] != "DONE"), None)
        active_subtask_id = active["id"] if active else None

    task_status = state.get("status", "PENDING")
    if task_status in ("IN_PROGRESS", "VERIFYING") and active_subtask_id:
        for subtask in normalized:
            if subtask["id"] != active_subtask_id:
                continue
            if subtask["status"] == "PENDING":
                subtask["status"] = "IN_PROGRESS"
            active = subtask
            break

    return {
        "subtasks": normalized,
        "activeSubtaskId": active_subtask_id,
        "activeSubtask": active,
        "allSubtasksDone": bool(normalized) and all(
            item["status"] == "DONE" for item in normalized
        ),
    }


def ensure_subtask_state(task_id: str) -> dict[str, Any]:
    packet = get_task_packet(task_id)
    task = packet["task"]
    existing = packet["state"] or {}
    normalized = normalize_subtask_state(task_id, task, existing)

    if (
        existing.get("subtasks") != normalized["subtasks"]
        or existing.get("activeSubtaskId") != normalized["activeSubtaskId"]
    ):
        update_task_state(
            task_id,
            effective_status(task_id, {task_id: existing}),
            subtasks=normalized["subtasks"],
            active_subtask_id=normalized["activeSubtaskId"],
        )

    return normalized


def get_active_subtask(task_id: str) -> dict[str, Any]:
    normalized = ensure_subtask_state(task_id)
    return {
        "taskId": task_id,
        **normalized,
    }


def update_subtask_state(
    task_id: str,
    subtask_id: str,
    status: str,
    *,
    summary: str | None = None,
    completion_kind: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_SUBTASK_STATUSES:
        raise ValueError(
            f"Invalid subtask status: {status}. Must be one of: {sorted(VALID_SUBTASK_STATUSES)}"
        )

    packet = get_task_packet(task_id)
    existing = packet["state"] or {}
    normalized = normalize_subtask_state(task_id, packet["task"], existing)
    subtasks = normalized["subtasks"]

    for subtask in subtasks:
        if subtask["id"] == subtask_id:
            subtask["status"] = status
            if summary is not None:
                subtask["summary"] = summary
            if completion_kind is not None:
                subtask["completionKind"] = completion_kind
            break
    else:
        raise ValueError(f"Subtask not found for {task_id}: {subtask_id}")

    active_subtask_id = normalized["activeSubtaskId"]
    if status == "DONE" and active_subtask_id == subtask_id:
        next_active = next((item for item in subtasks if item["status"] != "DONE"), None)
        active_subtask_id = next_active["id"] if next_active else None
    elif status in ("IN_PROGRESS", "FAILED", "BLOCKED"):
        active_subtask_id = subtask_id

    state = update_task_state(
        task_id,
        effective_status(task_id, {task_id: existing}),
        subtasks=subtasks,
        active_subtask_id=active_subtask_id,
    )

    return {
        "taskId": task_id,
        "subtasks": state.get("subtasks", []),
        "activeSubtaskId": state.get("activeSubtaskId"),
        "activeSubtask": next(
            (item for item in state.get("subtasks", []) if item["id"] == state.get("activeSubtaskId")),
            None,
        ),
        "allSubtasksDone": bool(state.get("subtasks")) and all(
            item["status"] == "DONE" for item in state.get("subtasks", [])
        ),
    }


def build_handoff_prompt(
    *,
    task_id: str,
    handoff_id: str,
    handoff_path: str,
    action: str = "implement",
) -> str:
    packet = get_task_packet(task_id)
    task = packet["task"]

    if action == "read":
        return (
            f"Continue from handoff {handoff_id}. "
            f"Call local_agent_read_handoff once with handoff_id {handoff_id}, "
            f"then follow the nextAction. Handoff path: {handoff_path}."
        )

    if action == "context_continue":
        brief = build_resume_brief(task_id, last_handoff_id=handoff_id or None)
        return (
            f"Context was reset because the previous chat approached the context budget. "
            f"Continue {task_id} from handoff {handoff_id} without redoing verified work. "
            f"Call read_handoff once if you need the cached result, then continue the active "
            f"subtask. Resume brief:\n"
            f"{json.dumps(brief, indent=2, ensure_ascii=False)}"
        )

    scope = "; ".join(task.get("scope", []))
    constraints = "; ".join(task.get("constraints", []))
    verification = "; ".join(task.get("verification", []))
    return (
        f"Continue from handoff {handoff_id}. Implement {task_id} only. "
        f"Use local_agent_read_handoff once if you need the cached handoff. "
        f"Task summary: {task.get('summary', '')}. "
        f"Scope: {scope}. "
        f"Constraints: {constraints}. "
        f"Acceptance criteria: {task.get('acceptanceCriteria', '')}. "
        f"Verification: {verification}. "
        f"After successful verification, update task state to DONE and call "
        f"local_agent_delete_handoff_chain for chain {task_id}."
    )


def update_task_state(
    task_id: str,
    status: str,
    *,
    summary: str | None = None,
    files_touched: list[str] | None = None,
    blockers: list[str] | None = None,
    verification_result: str | None = None,
    last_failure_reason: str | None = None,
    last_recovery_action: str | None = None,
    changed_files_since_attempt_start: list[str] | None = None,
    progress_summary: str | None = None,
    completed_scope_items: list[str] | None = None,
    remaining_scope_items: list[str] | None = None,
    verified_behaviors: list[str] | None = None,
    partial_files_touched: list[str] | None = None,
    recommended_resume_step: str | None = None,
    subtasks: list[dict[str, Any]] | None = None,
    active_subtask_id: str | None = None,
    preflight_status: str | None = None,
    preflight_blockers: list[str] | None = None,
    plan_sidecar_path: str | None = None,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}. Must be one of: {sorted(VALID_STATUSES)}")

    if not task_path(task_id).exists():
        raise ValueError(f"Task not found: {task_id}")

    existing = load_task_state(task_id) or {}
    task = json.loads(task_path(task_id).read_text(encoding="utf-8"))

    if status == "DONE":
        preview = {
            **existing,
            "status": status,
            "subtasks": subtasks if subtasks is not None else existing.get("subtasks", []),
            "activeSubtaskId": (
                active_subtask_id
                if active_subtask_id is not None
                else existing.get("activeSubtaskId")
            ),
        }
        normalized = normalize_subtask_state(task_id, task, preview)
        incomplete = [
            item["id"]
            for item in normalized["subtasks"]
            if item["status"] != "DONE"
        ]
        if incomplete:
            raise ValueError(
                f"Cannot mark {task_id} DONE while subtasks remain incomplete: "
                f"{', '.join(incomplete)}. Call update_subtask_state to finish "
                "each subtask first, then mark the task DONE."
            )

    now = _utc_now_iso()

    state: dict[str, Any] = {
        "taskId": task_id,
        "status": status,
        "attempt": existing.get("attempt", 0) + (1 if status == "IN_PROGRESS" and existing.get("status") != "IN_PROGRESS" else 0),
        "startedAt": existing.get("startedAt") or now,
        "updatedAt": now,
        "summary": summary if summary is not None else existing.get("summary", ""),
        "filesTouched": files_touched if files_touched is not None else existing.get("filesTouched", []),
        "blockers": blockers if blockers is not None else existing.get("blockers", []),
        "verificationResult": verification_result if verification_result is not None else existing.get("verificationResult"),
        "lastFailureReason": last_failure_reason if last_failure_reason is not None else existing.get("lastFailureReason"),
        "lastRecoveryAction": last_recovery_action if last_recovery_action is not None else existing.get("lastRecoveryAction"),
        "changedFilesSinceAttemptStart": (
            changed_files_since_attempt_start
            if changed_files_since_attempt_start is not None
            else existing.get("changedFilesSinceAttemptStart", [])
        ),
        "progressSummary": progress_summary if progress_summary is not None else existing.get("progressSummary", ""),
        "completedScopeItems": (
            completed_scope_items
            if completed_scope_items is not None
            else existing.get("completedScopeItems", [])
        ),
        "remainingScopeItems": (
            remaining_scope_items
            if remaining_scope_items is not None
            else existing.get("remainingScopeItems", [])
        ),
        "verifiedBehaviors": (
            verified_behaviors
            if verified_behaviors is not None
            else existing.get("verifiedBehaviors", [])
        ),
        "partialFilesTouched": (
            partial_files_touched
            if partial_files_touched is not None
            else existing.get("partialFilesTouched", [])
        ),
        "recommendedResumeStep": (
            recommended_resume_step
            if recommended_resume_step is not None
            else existing.get("recommendedResumeStep")
        ),
        "subtasks": subtasks if subtasks is not None else existing.get("subtasks", []),
        "activeSubtaskId": (
            active_subtask_id
            if active_subtask_id is not None
            else existing.get("activeSubtaskId")
        ),
        "preflightStatus": (
            preflight_status
            if preflight_status is not None
            else existing.get("preflightStatus")
        ),
        "preflightBlockers": (
            preflight_blockers
            if preflight_blockers is not None
            else existing.get("preflightBlockers", [])
        ),
        "planSidecarPath": (
            plan_sidecar_path
            if plan_sidecar_path is not None
            else existing.get("planSidecarPath")
        ),
    }

    if status == "IN_PROGRESS" and not existing.get("startedAt"):
        state["startedAt"] = now

    if status in ("IN_PROGRESS", "VERIFYING"):
        normalized = normalize_subtask_state(task_id, task, state)
        state["subtasks"] = normalized["subtasks"]
        state["activeSubtaskId"] = normalized["activeSubtaskId"]

    state_path(task_id).write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return state


def suggest_verification_commands(task: dict[str, Any]) -> list[str]:
    """Heuristic command suggestions from verification steps (not auto-run)."""
    verification = task.get("verification", [])
    combined = " ".join(verification).lower()
    suggestions: list[str] = []

    if "pytest" in combined or "test suite" in combined or "automated" in combined:
        if "price" in combined:
            suggestions.append("pytest tests/ -k price -v")
        else:
            suggestions.append("pytest tests/ -v")

    if "get /" in combined or "request get" in combined:
        suggestions.append(
            "python -c \"from fastapi.testclient import TestClient; "
            "from app.main import app; print(TestClient(app).get('/').status_code)\""
        )

    if "start the application" in combined or "start application" in combined:
        suggestions.append("python -c \"from app.main import app; print('app import ok')\"")

    if combined.startswith("import") or "import the" in combined:
        module_hint = _guess_domain_module(task)
        if module_hint:
            suggestions.append(f'python -c "import {module_hint}; print({module_hint!r})"')

    if "provide 96 price slots" in combined or "96 price slots" in combined:
        suggestions.append('python -c "from app.domain.prices import *; print(\'prices module ok\')"')

    if "generate a schedule" in combined:
        suggestions.append('python -c "from app.domain.schedule import *; print(\'schedule module ok\')"')

    if "set the variables" in combined or "environment" in combined:
        suggestions.append("python -c \"from app.config import *; print('config module ok')\"")

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cmd in suggestions:
        if cmd not in seen:
            seen.add(cmd)
            unique.append(cmd)
    return unique


def _guess_domain_module(task: dict[str, Any]) -> str | None:
    from orchestrator import config

    for item in task.get("scope", []):
        normalized = item.replace("\\", "/")
        if item.endswith(".py") and any(
            prefix in normalized for prefix in config.IMPLEMENTATION_PREFIXES
        ):
            return item.replace("\\", "/").removesuffix(".py").replace("/", ".")
    return None


def build_verification_checklist(task_id: str) -> dict[str, Any]:
    packet = get_task_packet(task_id)
    task = packet["task"]
    return {
        "taskId": task_id,
        "acceptanceCriteria": task.get("acceptanceCriteria", ""),
        "verification": task.get("verification", []),
    }


def build_resume_brief(
    task_id: str,
    *,
    last_handoff_id: str | None = None,
) -> dict[str, Any]:
    from orchestrator.continuity import load_continuity_packet

    packet = get_task_packet(task_id)
    task = packet["task"]
    state = packet["state"] or {"status": "PENDING", "summary": "", "filesTouched": []}

    status = state.get("status", "PENDING")
    summary = state.get("summary", "")
    files_touched = state.get("filesTouched", [])
    blockers = state.get("blockers", [])
    progress_summary = state.get("progressSummary", "")
    recommended_resume_step = state.get("recommendedResumeStep")
    resolved_handoff_id = last_handoff_id or state.get("lastHandoffId") or ""

    active_subtask_id = state.get("activeSubtaskId")
    subtask = get_subtask_spec_definition(task_id, str(active_subtask_id)) if active_subtask_id else None
    continuity = (
        load_continuity_packet(task_id, str(active_subtask_id))
        if active_subtask_id
        else None
    )

    scope_source = (subtask or {}).get("scope") or task.get("scope", [])
    verification_source = (subtask or {}).get("verification") or task.get("verification", [])

    if continuity:
        recommended_resume_step = continuity.get("nextStep") or recommended_resume_step
        blockers = continuity.get("blockers") or blockers
        progress_summary = (
            f"{continuity.get('subtaskId')}: {continuity.get('stopReason')}. "
            f"Next: {continuity.get('nextStep')}"
        )
        resolved_handoff_id = continuity.get("lastHandoffId") or resolved_handoff_id

    next_steps: list[str] = []
    if continuity:
        next_steps.append(str(continuity.get("nextStep") or "Continue from continuityPacket"))
        for gap in (continuity.get("gaps") or [])[:3]:
            if gap not in next_steps:
                next_steps.append(gap)
    elif status in ("IN_PROGRESS", "BLOCKED", "FAILED", "VERIFYING"):
        remaining_scope = list(state.get("remainingScopeItems") or [])
        if not remaining_scope:
            remaining_scope = [s for s in scope_source if s not in summary]
        next_steps.extend(remaining_scope[:3])
        if not next_steps:
            next_steps.append("Continue implementation within active subtask scope")
        if verification_source:
            next_steps.append(f"Run verification: {verification_source[0]}")
    elif status == "PENDING":
        next_steps.append("Start implementation from task scope")
        next_steps.append("Call local_agent_update_task_state with IN_PROGRESS")
    else:
        next_steps.append("Task may already be complete; verify state or pick next ready task")

    if blockers:
        next_steps.insert(0, f"Resolve blockers: {'; '.join(blockers)}")

    if continuity:
        prompt = (
            f"Resume {task_id}.{active_subtask_id} using continuityPacket. "
            f"Obey doNotRepeat and execute nextStep. "
            f"Last handoff: {resolved_handoff_id or 'none'}."
        )
    elif resolved_handoff_id:
        prompt = (
            f"Resume {task_id} from handoff {resolved_handoff_id}. "
            f"Call local_agent_get_resume_brief('{task_id}') and "
            f"local_agent_read_handoff once if needed, then continue implementation."
        )
    else:
        prompt = (
            f"Resume {task_id}. Call local_agent_get_resume_brief('{task_id}') "
            f"and local_agent_get_task('{task_id}'), then continue implementation."
        )

    result = {
        "taskId": task_id,
        "activeSubtaskId": active_subtask_id,
        "status": status,
        "summary": summary,
        "filesTouched": files_touched,
        "blockers": blockers,
        "progressSummary": progress_summary,
        "completedScopeItems": state.get("completedScopeItems", []),
        "remainingScopeItems": state.get("remainingScopeItems", []) or list(scope_source),
        "verifiedBehaviors": state.get("verifiedBehaviors", []),
        "partialFilesTouched": state.get("partialFilesTouched", []),
        "recommendedResumeStep": recommended_resume_step,
        "lastHandoffId": resolved_handoff_id,
        "scope": scope_source,
        "verification": verification_source,
        "constraints": task.get("constraints", []),
        "nextSteps": next_steps,
        "promptForNewChat": prompt,
    }
    if continuity is not None:
        result["continuityPacket"] = continuity
    return result
