from __future__ import annotations

import json
import re
from typing import Any

from tools import _handoffs, _tasks
from tools.read_file import read_file
from tools.run_command import run_command
from tools.search_files import search_files

from orchestrator.config import FORBIDDEN_COMMAND_PATTERNS
from orchestrator.read_guard import validate_read_file
from orchestrator.search_guard import cap_search_results, validate_search_files
from orchestrator.subtask_context import get_subtask_context_json


def update_task_state_tool(
    task_id: str,
    status: str,
    summary: str = "",
    files_touched: str = "",
    blockers: str = "",
    verification_result: str = "",
    last_failure_reason: str = "",
    last_recovery_action: str = "",
    changed_files_since_attempt_start: str = "",
    progress_summary: str = "",
    completed_scope_items: str = "",
    remaining_scope_items: str = "",
    verified_behaviors: str = "",
    partial_files_touched: str = "",
    recommended_resume_step: str = "",
) -> str:
    files = [item.strip() for item in files_touched.split(",") if item.strip()] or None
    blocker_items = [item.strip() for item in blockers.split(",") if item.strip()] or None
    changed_files = [item.strip() for item in changed_files_since_attempt_start.split(",") if item.strip()] or None
    completed_items = [item.strip() for item in completed_scope_items.split("|") if item.strip()] or None
    remaining_items = [item.strip() for item in remaining_scope_items.split("|") if item.strip()] or None
    verified_items = [item.strip() for item in verified_behaviors.split("|") if item.strip()] or None
    partial_files = [item.strip() for item in partial_files_touched.split(",") if item.strip()] or None
    state = _tasks.update_task_state(
        task_id,
        status,
        summary=summary or None,
        files_touched=files,
        blockers=blocker_items,
        verification_result=verification_result or None,
        last_failure_reason=last_failure_reason or None,
        last_recovery_action=last_recovery_action or None,
        changed_files_since_attempt_start=changed_files,
        progress_summary=progress_summary or None,
        completed_scope_items=completed_items,
        remaining_scope_items=remaining_items,
        verified_behaviors=verified_items,
        partial_files_touched=partial_files,
        recommended_resume_step=recommended_resume_step or None,
    )
    return json.dumps(state, indent=2, ensure_ascii=False)


def get_verification_checklist(task_id: str) -> str:
    return json.dumps(
        _tasks.build_verification_checklist(task_id),
        indent=2,
        ensure_ascii=False,
    )


def get_active_subtask_tool(task_id: str) -> str:
    return json.dumps(
        _tasks.get_active_subtask(task_id),
        indent=2,
        ensure_ascii=False,
    )


def update_subtask_state_tool(
    task_id: str,
    subtask_id: str,
    status: str,
    summary: str = "",
) -> str:
    return json.dumps(
        _tasks.update_subtask_state(
            task_id,
            subtask_id,
            status,
            summary=summary or None,
        ),
        indent=2,
        ensure_ascii=False,
    )


def safe_run_command(command: str) -> str:
    for pattern in FORBIDDEN_COMMAND_PATTERNS:
        if re.search(pattern, command, flags=re.IGNORECASE):
            return f"ERROR: Refusing potentially destructive command: {command}"
    return run_command(command)


def safe_read_file(path: str) -> str:
    try:
        refusal = validate_read_file(path)
    except ValueError as exc:
        return f"ERROR: {exc}"
    if refusal:
        return f"ERROR: {refusal}"
    return read_file(path)


def safe_search_files(query: str, path: str = ".") -> str:
    refusal = validate_search_files(query, path)
    if refusal:
        return f"ERROR: {refusal}"
    result = search_files(query, path)
    capped = cap_search_results(result, path)
    if capped.startswith("ERROR:"):
        return capped
    return result


def get_subtask_context_tool(task_id: str) -> str:
    try:
        return get_subtask_context_json(task_id)
    except ValueError as exc:
        return f"ERROR: {exc}"


def read_handoff(handoff_id: str = "", handoff_path: str = "", include_result: bool = False) -> str:
    record = _handoffs.read_handoff_record(handoff_id, handoff_path)
    response = {
        "handoffId": record["handoffId"],
        "previousHandoffId": record.get("previousHandoffId"),
        "stepIndex": record["stepIndex"],
        "chainId": record["chainId"],
        "taskId": record["taskId"],
        "tool": record["tool"],
        "summary": record.get("summary", ""),
        "nextAction": record.get("nextAction"),
        "copyPastePrompt": record.get("copyPastePrompt", ""),
    }
    if include_result:
        response["result"] = record.get("result")
    return json.dumps(response, indent=2, ensure_ascii=False)


def record_handoff_tool(
    task_id: str,
    tool: str,
    input_json: str = "",
    result_json: str = "",
    summary: str = "",
    previous_handoff_id: str = "",
    chain_id: str = "",
) -> str:
    handoff = _handoffs.record_handoff(
        task_id=task_id,
        tool=tool,
        input_data=_handoffs.parse_json_object(input_json, "input_json"),
        result=_handoffs.parse_json_value(result_json),
        summary=summary,
        previous_handoff_id=previous_handoff_id,
        chain_id=chain_id,
    )
    return json.dumps(handoff["compact"], indent=2, ensure_ascii=False)


def delete_handoff_chain(chain_id: str, mode: str = "consume") -> str:
    return json.dumps(
        _handoffs.delete_or_consume_chain(chain_id, mode),
        indent=2,
        ensure_ascii=False,
    )
