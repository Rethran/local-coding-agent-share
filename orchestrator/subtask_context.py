"""Assemble curated subtask context for worker implementation."""

from __future__ import annotations

import json
from typing import Any

from tools._common import WORKSPACE_ROOT, safe_path

from orchestrator import config
from orchestrator.search_guard import load_active_subtask_definition, load_subtask_definition


def _parse_line_range(value: str) -> tuple[int, int] | None:
    text = value.strip()
    if "-" not in text:
        return None
    start_text, end_text = text.split("-", 1)
    try:
        return int(start_text.strip()), int(end_text.strip())
    except ValueError:
        return None


def _read_line_slice(relative_path: str, line_range: str) -> dict[str, Any]:
    parsed = _parse_line_range(line_range)
    file_path = safe_path(relative_path)
    if parsed is None or not file_path.is_file():
        return {
            "file": relative_path,
            "lines": line_range,
            "source": "",
            "error": f"Could not read lines {line_range} from {relative_path}",
        }

    start_line, end_line = parsed
    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return {
            "file": relative_path,
            "lines": line_range,
            "source": "",
            "error": f"Could not read lines {line_range} from {relative_path}",
        }
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(lines))
    numbered = [
        f"{index + 1}: {lines[index]}"
        for index in range(start_index, end_index)
    ]
    return {
        "file": relative_path,
        "lines": line_range,
        "source": "\n".join(numbered),
    }


def _fixture_snippet(task: dict[str, Any], subtask: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[str] = []
    impl = task.get("implementationLogic") or {}
    candidates.extend(str(item) for item in impl.get("testFixtures") or [])
    for item in subtask.get("verification") or []:
        text = str(item)
        if ".xml" in text:
            for token in text.split():
                if token.endswith(".xml"):
                    candidates.append(token)

    seen: set[str] = set()
    for relative in candidates:
        if relative in seen:
            continue
        seen.add(relative)
        file_path = safe_path(relative)
        if file_path.exists() and file_path.suffix.lower() == ".xml":
            return _read_line_slice(relative, "1-45")
    return None


def build_subtask_context(task_id: str, subtask_id: str | None = None) -> dict[str, Any]:
    from tools import _tasks

    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    if subtask_id:
        subtask = load_subtask_definition(task_id, subtask_id)
    else:
        subtask = load_active_subtask_definition(task_id)
    if subtask is None:
        label = subtask_id or "active subtask"
        raise ValueError(f"No {label} found for {task_id}")

    legacy_slices = []
    for symbol in subtask.get("legacySymbols") or []:
        if not isinstance(symbol, dict):
            continue
        line_range = str(symbol.get("relevantLines") or "")
        relative_path = str(symbol.get("path") or "")
        slice_payload = {
            "name": symbol.get("name"),
            "purpose": symbol.get("purpose", ""),
            **_read_line_slice(relative_path, line_range),
        }
        legacy_slices.append(slice_payload)

    target = subtask.get("targetArtifacts") or {}
    verification = subtask.get("verification") or []
    next_steps = list(subtask.get("instructions") or [])
    if not next_steps and isinstance(target, dict):
        if target.get("test"):
            next_steps.append(f"Write failing test in {target['test']}.")
        if target.get("module") and target.get("function"):
            next_steps.append(
                f"Implement {target['function']} in {target['module']}."
            )
        if verification:
            next_steps.append(f"Run verification: {verification[0]}")

    return {
        "taskId": task_id,
        "subtaskId": subtask.get("id"),
        "title": subtask.get("title"),
        "summary": subtask.get("summary"),
        "targetArtifacts": target,
        "legacySlices": legacy_slices,
        "fixtureSnippet": _fixture_snippet(task, subtask),
        "dataExamples": subtask.get("dataExamples") or [],
        "allowedSearchTerms": subtask.get("allowedSearchTerms") or [],
        "forbiddenTools": (
            [f"search_files on {prefix}*.py when legacySymbols is defined" for prefix in config.LEGACY_PREFIXES]
            if config.LEGACY_PREFIXES
            else []
        ),
        "nextSteps": next_steps,
    }


def get_subtask_context_json(task_id: str) -> str:
    return json.dumps(build_subtask_context(task_id), indent=2, ensure_ascii=False)
