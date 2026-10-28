from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import _handoffs, _recovery, _tasks

from orchestrator import config
from orchestrator.continuity import load_continuity_packet
from orchestrator.session import SessionStore, get_session


def attempt_report_path(task_id: str) -> Path:
    return config.ATTEMPT_REPORTS_DIR / f"{task_id}.attempts.jsonl"


def append_attempt_report(report: dict[str, Any]) -> dict[str, Any]:
    config.ATTEMPT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = attempt_report_path(str(report["taskId"]))
    with path.open("a", encoding="utf-8") as report_file:
        report_file.write(json.dumps(report, ensure_ascii=False) + "\n")
    return report


def load_recent_attempt_reports(task_id: str, limit: int = 3) -> list[dict[str, Any]]:
    path = attempt_report_path(task_id)
    if not path.exists():
        return []

    reports: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            reports.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return reports[-limit:]


def infer_likely_cause_and_fix(failure_reason: str) -> tuple[str, str]:
    reason = failure_reason.lower()
    if "missing module file" in reason or "does not call" in reason:
        return (
            "The declared tests passed without the target module or function.",
            "Rewrite the declared test so it calls the declared function and fails until that module exists.",
        )
    if "cannot load" in reason or "cannot stub missing function" in reason:
        return (
            "The declared target function is not in the module, so a green test of another function is not proof.",
            "Rewrite the declared test so it calls the declared function and fails until that function exists.",
        )
    if "still pass when" in reason and "stub" in reason:
        return (
            "The declared tests pass even when the target function raises NotImplementedError.",
            "Rewrite the declared test so it fails when that function is stubbed.",
        )
    if "not executable by the coordinator oracle" in reason or "dataexamples" in reason:
        return (
            "The subtask dataExamples cannot be executed against the target function.",
            "Give the examples an input the function accepts, or satisfy the fields they assert.",
        )
    if "invalid tool call" in reason or "unexpected end of json" in reason:
        return (
            "Ollama returned malformed tool-call JSON.",
            "Retry with one small replace_in_file payload instead of a full-file write.",
        )
    if "tool step budget" in reason:
        return (
            "The worker exhausted its tool-step budget.",
            "Use the active subtask and recent handoff summary to continue with fewer exploratory calls.",
        )
    if "stopped without tool call" in reason or "assistant stopped without tool call" in reason:
        return (
            "The model stopped without completing the task state transition.",
            "Resume from the active subtask and call update_task_state only after verification.",
        )
    if "stopped before marking task done" in reason:
        return (
            "The model stopped without completing the task state transition.",
            "Resume from the active subtask and call update_task_state only after verification.",
        )
    return (
        "The worker failed before completing the task.",
        "Inspect the recent attempt report, then continue from the active subtask.",
    )


def finalize_task(task_id: str, state: dict[str, Any]) -> bool:
    status = state.get("status")
    if status == "DONE":
        print(f"{task_id}: DONE")
        cleanup = _handoffs.delete_or_consume_chain(task_id, mode="consume")
        print(f"{task_id}: consumed {cleanup['count']} handoff(s)")
        return True

    print(f"{task_id}: {status}")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    return False


def all_tasks_done(graph: dict[str, Any]) -> bool:
    tasks = graph.get("tasks", [])
    done = set(graph.get("done", []))
    return bool(tasks) and all(task["id"] in done for task in tasks)


def build_failure_progress(
    *,
    task_id: str,
    state: dict[str, Any],
    changed_files: list[str],
    previous_handoff_id: str,
) -> dict[str, Any]:
    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    subtask_id = state.get("activeSubtaskId")
    subtask = None
    if subtask_id:
        subtask = _tasks.get_subtask_spec_definition(task_id, str(subtask_id))

    completed = state.get("completedScopeItems", [])
    remaining = state.get("remainingScopeItems", [])
    if not remaining:
        scope_source = (subtask or {}).get("scope") or task.get("scope", [])
        remaining = [item for item in scope_source if item not in completed]

    progress_summary = state.get("progressSummary") or state.get("summary") or "No progress summary was provided by the worker."
    recommended_resume_step = (
        state.get("recommendedResumeStep")
        or "Inspect changed files and verification results, keep correct partial work, then continue remaining scope."
    )
    return {
        "taskId": task_id,
        "status": state.get("status", "FAILED"),
        "attempt": state.get("attempt"),
        "progressSummary": progress_summary,
        "completedScopeItems": completed,
        "remainingScopeItems": remaining,
        "verifiedBehaviors": state.get("verifiedBehaviors", []),
        "partialFilesTouched": state.get("partialFilesTouched") or changed_files,
        "changedFilesSinceAttemptStart": changed_files,
        "blockers": state.get("blockers", []),
        "lastFailureReason": state.get("lastFailureReason") or progress_summary,
        "previousHandoffId": previous_handoff_id,
        "recommendedResumeStep": recommended_resume_step,
        "activeSubtaskId": state.get("activeSubtaskId"),
    }


def describe_restore_result(result: _recovery.RestoreResult | None) -> str:
    if result is None:
        return "Rollback skipped by configuration."
    return (
        "Rollback completed. "
        f"Restored: {result.restored}; "
        f"Deleted: {result.deleted}; "
        f"Skipped: {result.skipped}."
    )


def update_failed_attempt_state(
    *,
    task_id: str,
    state: dict[str, Any],
    failure_progress: dict[str, Any],
    recovery_action: str,
) -> dict[str, Any]:
    return _tasks.update_task_state(
        task_id,
        state.get("status", "FAILED"),
        summary=state.get("summary") or failure_progress["progressSummary"],
        files_touched=state.get("filesTouched", []),
        blockers=state.get("blockers", []),
        verification_result=state.get("verificationResult"),
        last_failure_reason=failure_progress["lastFailureReason"],
        last_recovery_action=recovery_action,
        changed_files_since_attempt_start=failure_progress["changedFilesSinceAttemptStart"],
        progress_summary=failure_progress["progressSummary"],
        completed_scope_items=failure_progress["completedScopeItems"],
        remaining_scope_items=failure_progress["remainingScopeItems"],
        verified_behaviors=failure_progress["verifiedBehaviors"],
        partial_files_touched=failure_progress["partialFilesTouched"],
        recommended_resume_step=failure_progress["recommendedResumeStep"],
    )


def build_attempt_report(
    *,
    task_id: str,
    attempt_number: int,
    updated_state: dict[str, Any],
    failure_progress: dict[str, Any],
    recovery_action: str,
    restore_result: _recovery.RestoreResult | None,
) -> dict[str, Any]:
    last_failure_reason = updated_state.get("lastFailureReason") or failure_progress["lastFailureReason"]
    likely_cause, possible_fix = infer_likely_cause_and_fix(last_failure_reason)
    rolled_back_files: list[str] = []
    skipped_rollback_files: list[str] = []
    if restore_result is not None:
        rolled_back_files = [*restore_result.restored, *restore_result.deleted]
        skipped_rollback_files = restore_result.skipped

    return {
        "taskId": task_id,
        "attempt": attempt_number,
        "status": updated_state.get("status", "FAILED"),
        "failureReason": last_failure_reason,
        "likelyCause": likely_cause,
        "possibleFix": possible_fix,
        "changedFiles": failure_progress["changedFilesSinceAttemptStart"],
        "rolledBackFiles": rolled_back_files,
        "skippedRollbackFiles": skipped_rollback_files,
        "lastSuccessfulHandoffId": failure_progress.get("previousHandoffId", ""),
        "activeSubtaskId": updated_state.get("activeSubtaskId"),
        "recoveryAction": recovery_action,
    }


def build_retry_context(
    *,
    task_id: str,
    failure_progress: dict[str, Any],
    recovery_action: str,
    session: SessionStore | None = None,
    include_cached_context: bool = True,
) -> dict[str, Any]:
    active_session = session or get_session()
    subtask_id = failure_progress.get("activeSubtaskId") or ""
    continuity = load_continuity_packet(task_id, str(subtask_id)) if subtask_id else None

    if active_session is not None:
        state = _tasks.load_task_state(task_id) or {}
        active_session.extract_lessons_on_failure(task_id, state)
        active_session.persist()
        session_payload = active_session.get_session_payload_for_retry(
            task_id,
            include_cached_context=include_cached_context,
        )
    else:
        session_payload = {}

    instruction = (
        "This is a fresh retry for the same HEAT subtask. "
        "continuityPacket is authoritative when present: obey doNotRepeat, "
        "execute nextStep, and use evidence handoff IDs only if needed. "
        "sessionLessons are supplementary. Use cachedSubtaskContext instead "
        "of calling get_subtask_context again. Use recentAttemptReports for "
        "the previous failure context."
    )

    context: dict[str, Any] = {
        **failure_progress,
        **session_payload,
        "lastRecoveryAction": recovery_action,
        "recentAttemptReports": [
            {
                key: report.get(key)
                for key in (
                    "attempt",
                    "failureReason",
                    "likelyCause",
                    "possibleFix",
                    "changedFiles",
                    "rolledBackFiles",
                    "lastSuccessfulHandoffId",
                    "recoveryAction",
                )
                if report.get(key) not in (None, "", [])
            }
            for report in load_recent_attempt_reports(task_id, limit=2)
        ],
        "activeSubtask": _tasks.get_active_subtask(task_id).get("activeSubtask"),
        "instruction": instruction,
    }
    if continuity is not None:
        context["continuityPacket"] = continuity
    return context


def should_retry_task(status: str, attempt_number: int, max_attempts: int) -> bool:
    return status == "FAILED" and attempt_number < max(max_attempts, 1)
