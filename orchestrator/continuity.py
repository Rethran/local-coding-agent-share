"""Subtask Continuity Protocol (SCP): deterministic semantic resume artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.config import ATTEMPT_REPORTS_DIR
from orchestrator.read_guard import normalize_relative_path
from orchestrator.session import SessionStore
from tools import _handoffs, _tasks
from tools._common import safe_path

SCP_VERSION = "scp/v1"
CONTINUITY_DIR = ATTEMPT_REPORTS_DIR / "continuity"
SCHEMA_PATH = Path(__file__).resolve().parent / "scp_schema.json"

CAT_FILE_COMMAND_PATTERN = re.compile(
    r"(?:"
    r"\b(?:cat|type|more)\b"
    r"|\bGet-Content\b"
    r"|open\s*\("
    r"|read\s*\("
    r"|print\s*\(\s*open\s*\("
    r")",
    re.IGNORECASE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def continuity_dir(task_id: str) -> Path:
    return CONTINUITY_DIR / task_id


def latest_packet_path(task_id: str, subtask_id: str) -> Path:
    return continuity_dir(task_id) / f"{subtask_id}.latest.json"


def attempt_packet_path(task_id: str, subtask_id: str, attempt: int) -> Path:
    return continuity_dir(task_id) / f"{subtask_id}.attempt-{attempt}.json"


def _load_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def validate_continuity_packet(packet: dict[str, Any]) -> list[str]:
    """Return validation error messages; empty list means valid."""
    errors: list[str] = []

    if packet.get("schemaVersion") != SCP_VERSION:
        errors.append(f"schemaVersion must be {SCP_VERSION!r}.")

    required = _load_schema().get("required", [])
    for key in required:
        if key not in packet:
            errors.append(f"Missing required field: {key}.")

    status = packet.get("status")
    if status in ("failed", "blocked", "handover") and not str(packet.get("nextStep", "")).strip():
        errors.append("nextStep must be non-empty when status is failed, blocked, or handover.")

    if status in ("failed", "blocked") and not packet.get("blockers"):
        errors.append("blockers must be non-empty when status is failed or blocked.")

    return errors


def load_continuity_packet(task_id: str, subtask_id: str) -> dict[str, Any] | None:
    path = latest_packet_path(task_id, subtask_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def persist_continuity_packet(
    task_id: str,
    subtask_id: str,
    packet: dict[str, Any],
) -> Path:
    errors = validate_continuity_packet(packet)
    if errors:
        raise ValueError("Invalid continuity packet: " + "; ".join(errors))

    directory = continuity_dir(task_id)
    directory.mkdir(parents=True, exist_ok=True)

    latest = latest_packet_path(task_id, subtask_id)
    attempt_path = attempt_packet_path(
        task_id,
        subtask_id,
        int(packet.get("attempt", 1)),
    )
    payload = json.dumps(packet, indent=2, ensure_ascii=False) + "\n"
    latest.write_text(payload, encoding="utf-8")
    attempt_path.write_text(payload, encoding="utf-8")
    return latest


def list_handoff_summaries(task_id: str) -> list[dict[str, Any]]:
    directory = _handoffs.chain_dir(task_id)
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "handoffId": record.get("handoffId", ""),
                "tool": record.get("tool", ""),
                "summary": record.get("summary", ""),
                "nextAction": record.get("nextAction"),
            }
        )
    return rows


def _artifact_conclusion(path: str, *, function: str | None = None) -> str:
    file_path = safe_path(path)
    if not file_path.exists():
        return "missing"
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return "unreadable"
    if function:
        if function not in content:
            return f"exists; {function} not implemented"
        if "NotImplementedError" in content:
            return f"exists; {function} stub only"
        return f"exists; {function} present"
    line_count = len(content.splitlines())
    return f"exists; {line_count} lines"


def _build_do_not_repeat(
    session: SessionStore | None,
    task_id: str,
) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if session is None:
        return items

    task_state = session._task(task_id)
    for fingerprint, record in task_state.tool_records.items():
        if record.get("status") != "ok":
            continue
        tool = record.get("tool", "")
        args = record.get("args") or {}
        handoff_id = record.get("handoffId", "")

        if tool == "read_file":
            path = normalize_relative_path(str(args.get("path", "")))
            if path:
                items.append(
                    {
                        "action": "read_file",
                        "target": path,
                        "reason": f"already reviewed (handoff {handoff_id})",
                    }
                )
        elif tool == "extract_symbol":
            name = str(args.get("name", ""))
            path = normalize_relative_path(str(args.get("path", ".")))
            items.append(
                {
                    "action": "extract_symbol",
                    "target": f"{name}@{path}",
                    "reason": f"already extracted (handoff {handoff_id})",
                }
            )
        elif tool == "get_subtask_context":
            items.append(
                {
                    "action": "get_subtask_context",
                    "target": task_id,
                    "reason": f"cached in session (handoff {handoff_id})",
                }
            )
        elif tool == "search_files":
            query = str(args.get("query", ""))
            path = normalize_relative_path(str(args.get("path", ".")))
            items.append(
                {
                    "action": "search_files",
                    "target": f"{query}@{path}",
                    "reason": f"already searched (handoff {handoff_id})",
                }
            )
        elif tool == "read_handoff":
            handoff_target = str(args.get("handoff_id") or args.get("handoffId") or "")
            items.append(
                {
                    "action": "read_handoff",
                    "target": handoff_target,
                    "reason": f"already read (handoff {handoff_id})",
                }
            )
        elif tool == "list_directory":
            path = normalize_relative_path(str(args.get("path", ".")))
            items.append(
                {
                    "action": "list_directory",
                    "target": path,
                    "reason": f"already listed (handoff {handoff_id})",
                }
            )

    for lesson in task_state.lessons:
        if lesson.get("kind") != "do_not_repeat":
            continue
        text = str(lesson.get("text", "")).strip()
        if text:
            items.append(
                {
                    "action": "lesson",
                    "target": text[:200],
                    "reason": str(lesson.get("source", "session")),
                }
            )

    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item["action"], item["target"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _build_evidence(handoff_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for row in handoff_summaries[-20:]:
        handoff_id = row.get("handoffId", "")
        tool = row.get("tool", "")
        summary = row.get("summary", "")
        if tool == "run_command" and "Exit code:" in summary:
            exit_match = re.search(r"Exit code:\s*(\d+)", summary)
            command_match = re.search(r"Command result:\s*(.+?)\.\s*Output", summary)
            evidence.append(
                {
                    "kind": "test",
                    "command": command_match.group(1) if command_match else summary[:120],
                    "exitCode": int(exit_match.group(1)) if exit_match else -1,
                    "summary": summary,
                }
            )
        elif handoff_id:
            evidence.append(
                {
                    "kind": "handoff",
                    "id": handoff_id,
                    "summary": summary,
                }
            )
    return evidence


def _build_gaps(
    merged_subtask: dict[str, Any] | None,
    session: SessionStore | None,
    task_id: str,
) -> list[str]:
    gaps: list[str] = []
    if session is not None:
        for lesson in session._task(task_id).lessons:
            if lesson.get("kind") == "gap":
                text = str(lesson.get("text", "")).strip()
                if text and text not in gaps:
                    gaps.append(text)

    target = (merged_subtask or {}).get("targetArtifacts") or {}
    if isinstance(target, dict):
        module = target.get("module")
        function = target.get("function")
        if module and function:
            conclusion = _artifact_conclusion(str(module), function=str(function))
            if "not implemented" in conclusion or "stub only" in conclusion:
                gaps.append(f"{function} missing in {module}")
            elif conclusion == "missing":
                gaps.append(f"{module} missing")

    if not gaps and merged_subtask:
        for scope_item in merged_subtask.get("scope") or []:
            text = str(scope_item).strip()
            if text and text not in gaps:
                gaps.append(f"Remaining: {text}")
                if len(gaps) >= 5:
                    break
    return gaps[:10]


def _derive_next_step(
    merged_subtask: dict[str, Any] | None,
    gaps: list[str],
    *,
    recommended: str = "",
) -> str:
    if recommended.strip():
        return recommended.strip()

    instructions = list((merged_subtask or {}).get("instructions") or [])
    if instructions:
        first_instruction = str(instructions[0]).strip()
        if first_instruction:
            return first_instruction

    target = (merged_subtask or {}).get("targetArtifacts") or {}
    if isinstance(target, dict):
        module = target.get("module")
        function = target.get("function")
        test_path = target.get("test")
        if module and function:
            conclusion = _artifact_conclusion(str(module), function=str(function))
            if "not implemented" in conclusion or "stub only" in conclusion or conclusion == "missing":
                return f"Implement {function} in {module}, then run subtask verification."
        if module and test_path and gaps:
            return f"Implement remaining scope in {module}; add or update tests in {test_path}."

    if gaps:
        return gaps[0].removeprefix("Remaining: ")

    title = (merged_subtask or {}).get("title") or "active subtask"
    return f"Continue {title}: implement scope and run verification."


def _build_artifacts_reviewed(
    session: SessionStore | None,
    task_id: str,
    merged_subtask: dict[str, Any] | None,
) -> list[dict[str, str]]:
    reviewed: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    if session is not None:
        for fingerprint, record in session._task(task_id).tool_records.items():
            if record.get("tool") != "read_file" or record.get("status") != "ok":
                continue
            path = normalize_relative_path(str((record.get("args") or {}).get("path", "")))
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            reviewed.append({"path": path, "conclusion": _artifact_conclusion(path)})

    target = (merged_subtask or {}).get("targetArtifacts") or {}
    if isinstance(target, dict):
        for key in ("module", "test"):
            path = target.get(key)
            if not path:
                continue
            normalized = normalize_relative_path(str(path))
            if normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            function = str(target.get("function")) if key == "module" and target.get("function") else None
            reviewed.append(
                {
                    "path": normalized,
                    "conclusion": _artifact_conclusion(normalized, function=function),
                }
            )

    return reviewed


def _verified_behaviors(session: SessionStore | None, task_id: str) -> list[str]:
    if session is None:
        return []
    return [
        str(lesson.get("text", "")).strip()
        for lesson in session._task(task_id).lessons
        if lesson.get("kind") == "verified" and str(lesson.get("text", "")).strip()
    ]


def build_continuity_packet(
    *,
    task_id: str,
    subtask_id: str,
    attempt: int,
    status: str,
    merged_subtask: dict[str, Any] | None = None,
    session: SessionStore | None = None,
    stop_reason: str = "unknown",
    blockers: list[str] | None = None,
    last_handoff_id: str = "",
    changed_files: list[str] | None = None,
    recommended_resume_step: str = "",
) -> dict[str, Any]:
    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    subtask = merged_subtask
    if subtask is None:
        subtask = _tasks.get_subtask_spec_definition(task_id, subtask_id)

    handoff_summaries = list_handoff_summaries(task_id)
    gaps = _build_gaps(subtask, session, task_id)
    next_step = _derive_next_step(
        subtask,
        gaps,
        recommended=recommended_resume_step,
    )
    blocker_list = list(blockers or [])
    normalized_status = status.lower()
    if normalized_status not in ("in_progress", "failed", "handover", "blocked"):
        normalized_status = "failed"

    exploration_policy: dict[str, Any] = {}
    read_only_tools_used = 0
    if session is not None:
        read_only_tools_used = session.get_read_only_tools_used(task_id)
        task_state = session._task(task_id)
        exploration_policy = {
            "maxReadOnlyBeforeWrite": task_state.max_read_only_before_write,
            "preflightSatisfied": task_state.context_pack_preflight_satisfied,
            "readOnlyToolsUsed": read_only_tools_used,
            "readHandoffRule": "only when compact excerpt ends with [truncated; full output stored in handoff]",
        }
    elif isinstance(subtask, dict) and subtask.get("explorationPolicy"):
        exploration_policy = dict(subtask["explorationPolicy"])

    packet_out: dict[str, Any] = {
        "schemaVersion": SCP_VERSION,
        "taskId": task_id,
        "subtaskId": subtask_id,
        "attempt": max(int(attempt), 1),
        "status": normalized_status,
        "objective": (subtask or {}).get("title") or task.get("summary", task_id),
        "acceptance": list((subtask or {}).get("verification") or task.get("verification") or []),
        "constraints": list(task.get("constraints") or []),
        "decisions": [],
        "artifactsReviewed": _build_artifacts_reviewed(session, task_id, subtask),
        "verifiedBehaviors": _verified_behaviors(session, task_id),
        "gaps": gaps,
        "doNotRepeat": _build_do_not_repeat(session, task_id),
        "nextStep": next_step,
        "explorationPolicy": exploration_policy,
        "readOnlyToolsUsed": read_only_tools_used,
        "evidence": _build_evidence(handoff_summaries),
        "blockers": blocker_list,
        "stopReason": stop_reason,
        "lastHandoffId": last_handoff_id or "",
        "changedFiles": list(changed_files or []),
        "recommendedResumeStep": next_step,
        "createdAt": utc_now_iso(),
    }

    errors = validate_continuity_packet(packet_out)
    if errors:
        if not packet_out["blockers"]:
            packet_out["blockers"] = [errors[0]]
        if normalized_status == "failed":
            packet_out["stopReason"] = "validation_gap"

    return packet_out


def sync_task_state_from_packet(task_id: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Write SCP fields into HEAT task state for human-readable resume."""
    remaining = packet.get("gaps") or []
    completed: list[str] = []
    for item in packet.get("artifactsReviewed") or []:
        path = item.get("path", "")
        conclusion = item.get("conclusion", "")
        if path and "present" in conclusion:
            completed.append(f"Reviewed {path}: {conclusion}")

    verified = list(packet.get("verifiedBehaviors") or [])
    partial = list(packet.get("changedFiles") or [])
    progress = (
        f"{packet.get('subtaskId')}: {packet.get('stopReason')}. "
        f"Next: {packet.get('nextStep')}"
    )

    state = _tasks.load_task_state(task_id) or {}
    return _tasks.update_task_state(
        task_id,
        state.get("status", "IN_PROGRESS"),
        progress_summary=progress,
        completed_scope_items=completed,
        remaining_scope_items=remaining,
        verified_behaviors=verified,
        partial_files_touched=partial,
        recommended_resume_step=str(packet.get("recommendedResumeStep") or packet.get("nextStep") or ""),
        blockers=list(packet.get("blockers") or []),
        last_failure_reason=(
            packet.get("blockers", [""])[0]
            if packet.get("status") == "failed" and packet.get("blockers")
            else state.get("lastFailureReason")
        ),
    )


def persist_and_sync(
    *,
    task_id: str,
    subtask_id: str,
    attempt: int,
    status: str,
    merged_subtask: dict[str, Any] | None = None,
    session: SessionStore | None = None,
    stop_reason: str = "unknown",
    blockers: list[str] | None = None,
    last_handoff_id: str = "",
    changed_files: list[str] | None = None,
    recommended_resume_step: str = "",
) -> dict[str, Any]:
    packet = build_continuity_packet(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt=attempt,
        status=status,
        merged_subtask=merged_subtask,
        session=session,
        stop_reason=stop_reason,
        blockers=blockers,
        last_handoff_id=last_handoff_id,
        changed_files=changed_files,
        recommended_resume_step=recommended_resume_step,
    )
    persist_continuity_packet(task_id, subtask_id, packet)
    sync_task_state_from_packet(task_id, packet)
    return packet


def _duplicate_contract_lesson(
    attempt: int,
    blockers: list[str],
    recommended_resume_step: str,
) -> dict[str, str]:
    reason = blockers[0] if blockers else "the duplicate contract was not proven"
    text = (
        f"Do not submit another green test as done. "
        f"Attempt {attempt} failed because {reason} "
        f"Next: {recommended_resume_step}"
    )
    return {
        "action": "lesson",
        "target": text[:500],
        "reason": "duplicate_contract",
    }


def persist_duplicate_failure(
    *,
    task_id: str,
    subtask_id: str,
    attempt: int,
    merged_subtask: dict[str, Any] | None,
    session: SessionStore | None,
    blockers: list[str],
    recommended_resume_step: str,
) -> dict[str, Any]:
    """Replace an already-satisfied packet when the duplicate gate rejects the subtask."""
    state = _tasks.load_task_state(task_id) or {}
    previous = load_continuity_packet(task_id, subtask_id) or {}
    packet = build_continuity_packet(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt=attempt,
        status="failed",
        merged_subtask=merged_subtask,
        session=session,
        stop_reason="duplicate_contract",
        blockers=blockers,
        last_handoff_id=str(state.get("lastHandoffId") or previous.get("lastHandoffId") or ""),
        changed_files=list(state.get("changedFilesSinceAttemptStart") or previous.get("changedFiles") or []),
        recommended_resume_step=recommended_resume_step,
    )
    if not packet.get("verifiedBehaviors"):
        packet["verifiedBehaviors"] = list(
            state.get("verifiedBehaviors") or previous.get("verifiedBehaviors") or []
        )
    if not packet.get("evidence"):
        packet["evidence"] = list(previous.get("evidence") or [])
    if not packet.get("doNotRepeat"):
        packet["doNotRepeat"] = list(previous.get("doNotRepeat") or [])
    packet["doNotRepeat"] = [
        item
        for item in packet["doNotRepeat"]
        if item.get("reason") != "duplicate_contract"
    ]
    packet["doNotRepeat"].append(
        _duplicate_contract_lesson(attempt, blockers, recommended_resume_step)
    )
    persist_continuity_packet(task_id, subtask_id, packet)
    sync_task_state_from_packet(task_id, packet)
    return packet


def is_cat_file_command(command: str, known_read_paths: set[str]) -> str | None:
    """Return matched path if a command reads a guarded or already-read file."""
    if not CAT_FILE_COMMAND_PATTERN.search(command):
        return None
    normalized_command = command.replace("\\", "/").lower()
    for path in known_read_paths:
        normalized = normalize_relative_path(path)
        if normalized.lower() in normalized_command:
            return normalized
    return None


def infer_tool_next_action(
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    raw_result: str,
    *,
    merged_subtask: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if status != "ok":
        if tool_name == "read_file" and any(
            marker in raw_result
            for marker in (
                "contextPack",
                "Duplicate read_file",
                "Repeated read_file",
                "Refusing read_file",
            )
        ):
            return {
                "tool": "replace_in_file",
                "hint": "use preloaded or previously returned context; do not re-read",
            }
        from orchestrator.python_write_gate import (
            SYNTAX_WRITE_RETRY_HINT,
            is_python_syntax_write_error,
        )

        if tool_name in {"write_file", "replace_in_file"} and is_python_syntax_write_error(
            raw_result
        ):
            action: dict[str, Any] = {
                "tool": tool_name,
                "hint": SYNTAX_WRITE_RETRY_HINT,
            }
            path = str(arguments.get("path") or "").strip()
            if path:
                action["path"] = path
            return action
        return {"tool": tool_name, "hint": "fix error and retry or choose another approach"}

    from orchestrator.session import get_session

    active_session = get_session()
    if active_session is not None:
        task_id = str(arguments.get("task_id") or "")
        if not task_id and merged_subtask:
            task_id = str(merged_subtask.get("taskId") or "")
        if task_id:
            warning = active_session.exploration_budget_warning(task_id)
            if warning and tool_name in {
                "read_file",
                "search_files",
                "extract_symbol",
                "find_symbol",
                "list_directory",
                "read_handoff",
                "get_subtask_context",
            }:
                return {"tool": "replace_in_file", "hint": warning}

    target = (merged_subtask or {}).get("targetArtifacts") or {}
    module = target.get("module") if isinstance(target, dict) else None
    function = target.get("function") if isinstance(target, dict) else None

    if tool_name == "read_file":
        return {"tool": "replace_in_file", "hint": f"implement using context from {arguments.get('path')}"}
    if tool_name == "extract_symbol":
        hint = f"port {arguments.get('name')} to {module or 'target module'}"
        return {"tool": "replace_in_file", "hint": hint}
    if tool_name == "get_subtask_context":
        return {"tool": "replace_in_file", "hint": "implement using legacy context already loaded"}
    if tool_name == "run_command":
        first_line = raw_result.splitlines()[0] if raw_result.splitlines() else ""
        if "Exit code: 0" in first_line:
            command = str(arguments.get("command") or "").strip()
            declared = [
                str(item).strip()
                for item in (merged_subtask or {}).get("verification") or []
                if isinstance(item, str) and str(item).strip()
            ]
            if declared and command not in declared:
                return {
                    "tool": "run_command",
                    "hint": f"run exact declared verification: {declared[0]}",
                }
            return {
                "tool": "update_subtask_state",
                "hint": "declared verification passed; coordinator will confirm completion",
            }
        if "Exit code: 1" in first_line or "Exit code: 2" in first_line:
            return {"tool": "replace_in_file", "hint": "fix failing tests then re-run verification"}
    if tool_name in {"write_file", "replace_in_file"} and module and function:
        return {"tool": "run_command", "hint": f"run subtask verification after updating {function}"}
    return None
