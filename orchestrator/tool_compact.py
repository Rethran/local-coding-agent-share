from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from tools import _handoffs

from orchestrator.config import MAX_COMPACT_EXCERPT_CHARS, MAX_PARALLEL_READONLY_TOOLS, READ_ONLY_TOOLS, TRUNCATED_EXCERPT_MARKER
from orchestrator.session import get_session
from orchestrator.tools_registry import TOOL_FUNCTIONS
from orchestrator.search_guard import load_active_subtask_definition
from orchestrator.worker_context import get_worker_attempt, get_worker_task_id
from orchestrator.write_guard import refuse_worker_write


@dataclass(frozen=True)
class ToolExecutionResult:
    name: str
    arguments: dict[str, Any]
    status: str
    raw_result: str


def get_tool_name_and_args(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = getattr(tool_call, "function", None)
    if function is None and isinstance(tool_call, dict):
        function = tool_call.get("function", {})

    name = getattr(function, "name", None) if function is not None else None
    if name is None and isinstance(function, dict):
        name = function.get("name")

    raw_args = getattr(function, "arguments", None) if function is not None else None
    if raw_args is None and isinstance(function, dict):
        raw_args = function.get("arguments", {})

    if isinstance(raw_args, str):
        args = json.loads(raw_args) if raw_args else {}
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        args = {}

    if not name:
        raise ValueError(f"Tool call missing name: {tool_call}")
    return name, args


def summarize_tool_result(tool_name: str, result: str) -> tuple[str, str]:
    lines = result.splitlines()
    line_count = len(lines)
    char_count = len(result)

    if tool_name == "extract_symbol":
        return _summarize_extract_symbol(result, line_count, char_count)

    if tool_name == "read_file":
        summary = f"Read file output: {line_count} lines, {char_count} chars."
    elif tool_name == "search_files":
        try:
            matches = json.loads(result)
            summary = f"Search returned {len(matches)} matches."
        except json.JSONDecodeError:
            summary = f"Search output: {line_count} lines, {char_count} chars."
    elif tool_name == "run_command":
        first_line = lines[0] if lines else "No command output"
        summary = (
            f"Command result: {first_line}. "
            f"Output {line_count} lines, {char_count} chars."
        )
    else:
        summary = f"{tool_name} returned {line_count} lines, {char_count} chars."

    excerpt = result[:MAX_COMPACT_EXCERPT_CHARS]
    if len(result) > MAX_COMPACT_EXCERPT_CHARS:
        excerpt += f"\n{TRUNCATED_EXCERPT_MARKER}"
    return summary, excerpt


def _summarize_extract_symbol(
    result: str,
    line_count: int,
    char_count: int,
) -> tuple[str, str]:
    try:
        matches = json.loads(result)
    except json.JSONDecodeError:
        excerpt = result[:MAX_COMPACT_EXCERPT_CHARS]
        if len(result) > MAX_COMPACT_EXCERPT_CHARS:
            excerpt += f"\n{TRUNCATED_EXCERPT_MARKER}"
        return (
            f"extract_symbol returned {line_count} lines, {char_count} chars.",
            excerpt,
        )

    if not isinstance(matches, list):
        excerpt = result[:MAX_COMPACT_EXCERPT_CHARS]
        if len(result) > MAX_COMPACT_EXCERPT_CHARS:
            excerpt += f"\n{TRUNCATED_EXCERPT_MARKER}"
        return (
            f"extract_symbol returned {line_count} lines, {char_count} chars.",
            excerpt,
        )

    if not matches:
        return "extract_symbol returned 0 matches.", "[]"

    first = matches[0] if isinstance(matches[0], dict) else {}
    name = first.get("name", "?")
    file_path = first.get("file", "?")
    start_line = first.get("start_line", "?")
    end_line = first.get("end_line", "?")
    source = str(first.get("source", ""))
    summary = (
        f"extract_symbol returned {len(matches)} match(es); "
        f"first {name} in {file_path} lines {start_line}-{end_line}, "
        f"{len(source)} source chars."
    )

    header = (
        f"{file_path}:{start_line}-{end_line} {name}\n"
        if file_path != "?"
        else ""
    )
    budget = max(MAX_COMPACT_EXCERPT_CHARS - len(header), 0)
    body = source[:budget]
    excerpt = header + body
    if len(source) > budget or len(matches) > 1:
        excerpt += f"\n{TRUNCATED_EXCERPT_MARKER}"
    return summary, excerpt


def execute_tool(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return "error", f"ERROR: Unknown tool: {name}"

    task_id = get_worker_task_id()
    session = get_session()
    if task_id and session is not None:
        preflight = session.check_before_execute(task_id, name, arguments)
        if preflight is not None:
            if preflight.startswith("CACHED:"):
                return "ok", preflight.removeprefix("CACHED:")
            return "error", preflight

    try:
        if name in {"write_file", "replace_in_file"}:
            attempt = get_worker_attempt()
            if attempt is not None:
                refusal = refuse_worker_write(
                    str(arguments.get("path") or ""),
                    role=attempt.role,
                    merged_subtask=attempt.merged_subtask,
                )
                if refusal:
                    return "error", refusal
        if name == "update_subtask_state":
            attempt = get_worker_attempt()
            if (
                attempt is not None
                and attempt.role == "test_writer"
                and str(arguments.get("status") or "").upper() == "DONE"
            ):
                return (
                    "error",
                    "ERROR: test-writer cannot mark the subtask DONE. "
                    "Stop after the declared verification command. "
                    "If it fails, the implementer will make it pass. "
                    "If it already passes, the coordinator will mark the subtask DONE.",
                )
        result = function(**arguments)
        return "ok", str(result)
    except Exception as exc:  # noqa: BLE001 - return tool errors to model
        return "error", f"ERROR executing {name}: {exc}"


def _preflight_blocks_parallel(task_id: str, parsed_calls: list[tuple[str, dict[str, Any]]]) -> bool:
    session = get_session()
    if session is None:
        return False
    for name, arguments in parsed_calls:
        preflight = session.check_before_execute(task_id, name, arguments)
        if preflight is not None and not preflight.startswith("CACHED:"):
            return True
    return False


def _has_duplicate_fingerprints(parsed_calls: list[tuple[str, dict[str, Any]]]) -> bool:
    session = get_session()
    if session is None:
        return False
    seen: set[str] = set()
    for name, arguments in parsed_calls:
        fingerprint = session.normalize_fingerprint(name, arguments)
        if fingerprint in seen:
            return True
        seen.add(fingerprint)
    return False


def execute_tool_batch(tool_calls: list[Any]) -> list[ToolExecutionResult]:
    parsed_calls: list[tuple[str, dict[str, Any], str | None]] = []
    for tool_call in tool_calls:
        try:
            name, arguments = get_tool_name_and_args(tool_call)
            parsed_calls.append((name, arguments, None))
        except (ValueError, json.JSONDecodeError) as exc:
            parsed_calls.append(("invalid_tool_call", {}, str(exc)))

    if len(parsed_calls) <= 1:
        results: list[ToolExecutionResult] = []
        for name, arguments, parse_error in parsed_calls:
            if parse_error is not None:
                results.append(
                    ToolExecutionResult(
                        name=name,
                        arguments=arguments,
                        status="error",
                        raw_result=f"ERROR parsing tool call: {parse_error}",
                    )
                )
                continue
            status, raw_result = execute_tool(name, arguments)
            results.append(
                ToolExecutionResult(
                    name=name,
                    arguments=arguments,
                    status=status,
                    raw_result=raw_result,
                )
            )
        return results

    executable_calls = [
        (name, arguments)
        for name, arguments, parse_error in parsed_calls
        if parse_error is None
    ]
    all_readonly = bool(executable_calls) and all(
        name in READ_ONLY_TOOLS for name, _ in executable_calls
    )
    task_id = get_worker_task_id() or ""
    session = get_session()
    preflight_satisfied = False
    if session is not None and task_id:
        preflight_satisfied = session._task(task_id).context_pack_preflight_satisfied
    use_parallel = (
        all_readonly
        and len(executable_calls) == len(parsed_calls)
        and not preflight_satisfied
        and not _preflight_blocks_parallel(task_id, executable_calls)
        and not _has_duplicate_fingerprints(executable_calls)
    )

    if use_parallel:
        max_workers = min(len(executable_calls), MAX_PARALLEL_READONLY_TOOLS)
        index_map = {
            call_index: parsed_index
            for call_index, parsed_index in enumerate(
                index
                for index, (_, _, parse_error) in enumerate(parsed_calls)
                if parse_error is None
            )
        }

        def _run(call_index: int) -> tuple[int, ToolExecutionResult]:
            parsed_index = index_map[call_index]
            name, arguments, _ = parsed_calls[parsed_index]
            status, raw_result = execute_tool(name, arguments)
            return parsed_index, ToolExecutionResult(
                name=name,
                arguments=arguments,
                status=status,
                raw_result=raw_result,
            )

        ordered: list[ToolExecutionResult | None] = [None] * len(parsed_calls)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for parsed_index, result in executor.map(_run, range(len(executable_calls))):
                ordered[parsed_index] = result
        return [item for item in ordered if item is not None]

    results: list[ToolExecutionResult] = []
    for name, arguments, parse_error in parsed_calls:
        if parse_error is not None:
            results.append(
                ToolExecutionResult(
                    name=name,
                    arguments=arguments,
                    status="error",
                    raw_result=f"ERROR parsing tool call: {parse_error}",
                )
            )
            continue
        status, raw_result = execute_tool(name, arguments)
        results.append(
            ToolExecutionResult(
                name=name,
                arguments=arguments,
                status=status,
                raw_result=raw_result,
            )
        )
    return results


def append_compact_tool_result(
    *,
    messages: list[dict[str, Any]],
    task_id: str,
    previous_handoff_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    raw_result: str,
    merged_subtask: dict[str, Any] | None = None,
) -> str:
    summary, excerpt = summarize_tool_result(tool_name, raw_result)
    next_action = None
    from orchestrator.continuity import infer_tool_next_action

    subtask = merged_subtask or load_active_subtask_definition(task_id)
    next_action = infer_tool_next_action(
        tool_name,
        arguments,
        status,
        raw_result,
        merged_subtask=subtask,
    )
    handoff = _handoffs.record_handoff(
        task_id=task_id,
        tool=tool_name,
        input_data=arguments,
        result={
            "status": status,
            "output": raw_result,
        },
        summary=summary,
        previous_handoff_id=previous_handoff_id,
        chain_id=task_id,
        next_action=next_action,
    )
    compact = handoff["compact"]
    compact_result = {
        "tool": tool_name,
        "status": status,
        "handoffId": compact["handoffId"],
        "handoffPath": compact["handoffPath"],
        "summary": summary,
        "excerpt": excerpt,
    }
    if next_action is not None:
        compact_result["nextAction"] = next_action
    messages.append({
        "role": "tool",
        "content": json.dumps(compact_result, indent=2, ensure_ascii=False),
    })
    print(
        f"{task_id}: compact result: {status}; {summary} "
        f"(handoff {compact['handoffId']})"
    )
    session = get_session()
    if session is not None:
        session.note_truncated_handoff(task_id, compact["handoffId"], excerpt)
        session.record_after_execute(
            task_id,
            tool_name,
            arguments,
            status,
            compact["handoffId"],
            raw_result,
            next_action=next_action,
        )
    return compact["handoffId"]
