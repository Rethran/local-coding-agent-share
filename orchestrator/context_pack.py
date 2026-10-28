"""Deterministic preflight context assembly for orchestrator workers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools._common import WORKSPACE_ROOT, is_within_ignored_directory, safe_path
from tools._symbols import scan_for_extract_matches

from orchestrator.config import (
    IMPLEMENTATION_PREFIXES,
    MAX_CONTEXT_PACK_SNIPPET_CHARS,
    MODULE_PREAMBLE_MAX_LINES,
    READ_FILE_MAX_PY_LINES,
    TARGET_SNAPSHOT_EXCERPT_LINES,
    DEFAULT_MAX_READ_ONLY_BEFORE_WRITE,
    REPO_CONTEXT_TOKEN_BUDGET,
    MAX_REPO_CONTEXT_MATCHES,
    TASK_PREFIX,
)
from orchestrator.read_guard import normalize_relative_path
from orchestrator.subtask_context import _parse_line_range, _read_line_slice, build_subtask_context
from tools import _tasks

MAX_DIRECTORY_SNIPPET_FILES = 12


def _workspace_relative(path: Path) -> str:
    return normalize_relative_path(str(path.relative_to(WORKSPACE_ROOT)))


def _safe_read_text(file_path: Path) -> str | None:
    if not file_path.is_file():
        return None
    try:
        return file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _directory_file_paths(
    relative_path: str,
    *,
    limit: int = MAX_DIRECTORY_SNIPPET_FILES,
) -> list[str]:
    directory = safe_path(relative_path)
    try:
        if not directory.is_dir():
            return []
        children = [
            child
            for child in sorted(directory.iterdir())
            if child.is_file() and not is_within_ignored_directory(child)
        ]
    except OSError:
        return []
    return [_workspace_relative(child) for child in children[:limit]]


def _expand_artifact_paths(relative_path: str) -> list[str]:
    """Return file paths for a spec artifact. Directories expand to contained files."""
    normalized = normalize_relative_path(relative_path)
    if not normalized:
        return []
    path = safe_path(normalized)
    if path.is_dir():
        return _directory_file_paths(normalized)
    return [normalized]


def _read_file_excerpt(relative_path: str, *, max_lines: int = TARGET_SNAPSHOT_EXCERPT_LINES) -> str:
    file_path = safe_path(relative_path)
    content = _safe_read_text(file_path)
    if content is None:
        return ""
    lines = content.splitlines()
    numbered = [f"{index + 1}: {line}" for index, line in enumerate(lines[:max_lines])]
    excerpt = "\n".join(numbered)
    if len(lines) > max_lines:
        excerpt += f"\n[truncated; {len(lines)} total lines]"
    return excerpt


def _extract_symbol_with_preamble(relative_path: str, name: str) -> str:
    file_path = safe_path(relative_path)
    if not file_path.exists():
        return ""
    try:
        matches = scan_for_extract_matches(name, relative_path)
    except (OSError, ValueError):
        return ""
    normalized = normalize_relative_path(relative_path)
    match = None
    for candidate in matches:
        if candidate.get("file", "").replace("\\", "/") == normalized:
            match = candidate
            break
    if match is None:
        return ""

    symbol_source = str(match.get("source", ""))
    start_line = int(match.get("start_line") or 1)
    if start_line <= 1:
        return symbol_source

    preamble_end = min(start_line - 1, MODULE_PREAMBLE_MAX_LINES)
    if preamble_end < 1:
        return symbol_source

    preamble = _read_line_slice(relative_path, f"1-{preamble_end}").get("source", "")
    if not preamble.strip():
        return symbol_source

    return f"{preamble}\n\n# --- {name} ---\n{symbol_source}"


def _extract_symbol_source(relative_path: str, name: str) -> str:
    return _extract_symbol_with_preamble(relative_path, name)


def _artifact_to_snippet(artifact: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(artifact, dict):
        return None

    relative_path = normalize_relative_path(str(artifact.get("path") or ""))
    if not relative_path:
        return None

    name = str(artifact.get("name") or "").strip()
    purpose = str(artifact.get("purpose") or "")
    line_range = str(artifact.get("relevantLines") or "").strip()

    payload: dict[str, Any] = {
        "path": relative_path,
        "purpose": purpose,
    }
    if name:
        payload["name"] = name

    if line_range and _parse_line_range(line_range) is not None:
        slice_payload = _read_line_slice(relative_path, line_range)
        payload["source"] = slice_payload.get("source", "")
        if slice_payload.get("error"):
            payload["error"] = slice_payload["error"]
        payload["lines"] = line_range
    elif name:
        source = _extract_symbol_with_preamble(relative_path, name)
        payload["source"] = source
        if not source:
            payload["error"] = f"Symbol {name!r} not found in {relative_path}"
    else:
        file_path = safe_path(relative_path)
        if not file_path.exists():
            payload["error"] = f"File not found: {relative_path}"
            payload["source"] = ""
        elif not file_path.is_file():
            payload["error"] = f"Not a file: {relative_path}"
            payload["source"] = ""
        else:
            content = _safe_read_text(file_path)
            if content is None:
                payload["error"] = f"Could not read {relative_path}"
                payload["source"] = ""
            else:
                line_count = len(content.splitlines())
                if line_count <= READ_FILE_MAX_PY_LINES:
                    payload["source"] = _read_file_excerpt(relative_path, max_lines=line_count)
                else:
                    payload["source"] = _read_file_excerpt(relative_path)

    return payload


def _api_service_path(api_module_path: str) -> str | None:
    normalized = normalize_relative_path(api_module_path)
    if not normalized.startswith("app/api/") or not normalized.endswith(".py"):
        return None
    stem = Path(normalized).stem
    candidate = normalize_relative_path(f"app/services/{stem}_store.py")
    if safe_path(candidate).exists():
        return candidate
    candidate = normalize_relative_path(f"app/services/{stem}.py")
    if safe_path(candidate).exists():
        return candidate
    return None


def _auto_related_artifacts(subtask: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    target = subtask.get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return artifacts

    module_path = str(target.get("module") or "")
    if module_path:
        normalized = normalize_relative_path(module_path)
        if safe_path(normalized).exists():
            artifacts.append(
                {
                    "path": normalized,
                    "purpose": "Current target module state.",
                }
            )
        service_path = _api_service_path(module_path)
        if service_path:
            artifacts.append(
                {
                    "path": service_path,
                    "name": Path(service_path).stem.replace("_", " ").title().replace(" ", ""),
                    "purpose": "Paired service module for API subtask.",
                }
            )

    task_id = str(task.get("id") or "")
    subtask_id = str(subtask.get("id") or "")
    state = _tasks.load_task_state(task_id) or {}
    subtask_status = {
        str(item.get("id")): str(item.get("status"))
        for item in state.get("subtasks") or []
        if isinstance(item, dict)
    }
    for sibling in task.get("subtasks") or []:
        if not isinstance(sibling, dict):
            continue
        sibling_id = str(sibling.get("id") or "")
        if sibling_id == subtask_id or subtask_status.get(sibling_id) != "DONE":
            continue
        sibling_target = sibling.get("targetArtifacts") or {}
        if not isinstance(sibling_target, dict):
            continue
        sibling_module = str(sibling_target.get("module") or "")
        sibling_function = str(sibling_target.get("function") or "")
        if sibling_module and sibling_function:
            artifacts.append(
                {
                    "path": sibling_module,
                    "name": sibling_function,
                    "purpose": f"Completed output from {sibling_id}.",
                }
            )

    return artifacts


def _build_reference_snippets(
    subtask: dict[str, Any],
    task: dict[str, Any],
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    snippets: list[dict[str, Any]] = []
    total_chars = 0

    def add_one(artifact: dict[str, Any]) -> None:
        nonlocal total_chars
        snippet = _artifact_to_snippet(artifact)
        if snippet is None:
            return
        key = (str(snippet.get("path", "")), str(snippet.get("name", "")))
        if key in seen:
            return
        source = str(snippet.get("source") or "")
        if total_chars + len(source) > MAX_CONTEXT_PACK_SNIPPET_CHARS:
            remaining = max(MAX_CONTEXT_PACK_SNIPPET_CHARS - total_chars, 0)
            if remaining <= 0:
                return
            snippet["source"] = source[:remaining] + "\n[truncated; context pack budget]"
            snippet["truncated"] = True
        seen.add(key)
        snippets.append(snippet)
        total_chars += len(str(snippet.get("source") or ""))

    def add_snippet(artifact: dict[str, Any]) -> None:
        paths = _expand_artifact_paths(str(artifact.get("path") or ""))
        if not paths:
            add_one(artifact)
            return
        for path in paths:
            add_one({**artifact, "path": path})

    target = subtask.get("targetArtifacts") or {}
    if isinstance(target, dict):
        module_path = str(target.get("module") or "")
        function_name = str(target.get("function") or "")
        if module_path:
            if function_name:
                add_snippet(
                    {
                        "path": module_path,
                        "name": function_name,
                        "purpose": "Target function in module.",
                    }
                )
            else:
                add_snippet({"path": module_path, "purpose": "Target module."})

    for artifact in subtask.get("relatedArtifacts") or []:
        if isinstance(artifact, dict):
            add_snippet(artifact)

    for artifact in _auto_related_artifacts(subtask, task):
        add_snippet(artifact)

    return snippets


def _repo_context_needles(subtask: dict[str, Any]) -> list[str]:
    needles: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        text = name.strip()
        if not text or text in seen:
            return
        seen.add(text)
        needles.append(text)

    target = subtask.get("targetArtifacts") or {}
    if isinstance(target, dict):
        add(str(target.get("function") or ""))
    for artifact in subtask.get("relatedArtifacts") or []:
        if isinstance(artifact, dict):
            add(str(artifact.get("name") or ""))
    return needles


def build_repo_context(
    subtask: dict[str, Any],
    *,
    repo_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Token-capped blast-radius slice for names already on the subtask spec."""
    needles = _repo_context_needles(subtask)
    empty = {"matches": [], "token_budget_used": 0}
    if not needles:
        return empty
    if TASK_PREFIX != "HEAT":
        return empty

    try:
        from orchestrator.bug_ticket import callers_of, guard_candidates
        from orchestrator.heating_map import get_heating_repo_map
    except ImportError:
        return empty

    try:
        heating_map = repo_map if repo_map is not None else get_heating_repo_map()
        verified = guard_candidates(needles, heating_map)
    except Exception:
        return empty

    matches: list[dict[str, Any]] = []
    tokens_used = 0
    for entry in verified[:MAX_REPO_CONTEXT_MATCHES]:
        signature = str(entry.get("signature") or "")
        cost = max(1, len(signature) // 4)
        if tokens_used + cost > REPO_CONTEXT_TOKEN_BUDGET:
            break
        callers = callers_of(entry, heating_map)
        matches.append(
            {
                "name": entry.get("name"),
                "file": entry.get("file"),
                "signature": signature,
                "score": entry.get("score", 0.0),
                "start_line": entry.get("start_line"),
                "end_line": entry.get("end_line"),
                "blast_radius": [caller["name"] for caller in callers],
            }
        )
        tokens_used += cost
    return {"matches": matches, "token_budget_used": tokens_used}


def _domain_knowledge(task_id: str, subtask_id: str) -> str:
    from orchestrator.knowledge import knowledge_pack_text

    return knowledge_pack_text(subtask_id or task_id)


def _build_exploration_policy(
    subtask: dict[str, Any],
    subtask_context: dict[str, Any],
    reference_snippets: list[dict[str, Any]],
    repo_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    legacy_slices = subtask_context.get("legacySlices") or []
    read_legacy = subtask.get("readLegacy")
    preflight_satisfied = bool(legacy_slices) or read_legacy is False or bool(reference_snippets)

    spec_policy = subtask.get("explorationPolicy") or {}
    max_read_only = DEFAULT_MAX_READ_ONLY_BEFORE_WRITE
    if isinstance(spec_policy, dict) and spec_policy.get("maxReadOnlyBeforeWrite") is not None:
        max_read_only = int(spec_policy["maxReadOnlyBeforeWrite"])

    skip_tools: list[str] = []
    if legacy_slices or reference_snippets:
        skip_tools.append("get_subtask_context")
    if legacy_slices:
        skip_tools.extend(["extract_symbol"])
    if repo_context and repo_context.get("matches"):
        skip_tools.append("get_repo_context")

    return {
        "maxReadOnlyBeforeWrite": max_read_only,
        "preflightSatisfied": preflight_satisfied,
        "skipTools": skip_tools,
        "readHandoffRule": "only when compact excerpt ends with [truncated; full output stored in handoff]",
    }


def _target_file_snapshot(subtask: dict[str, Any]) -> dict[str, Any]:
    target = subtask.get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return {"path": None, "exists": False}

    module_path = target.get("module")
    function_name = target.get("function")
    if not module_path:
        for item in subtask.get("scope") or []:
            text = str(item)
            normalized_text = text.replace("\\", "/")
            if any(prefix in normalized_text for prefix in IMPLEMENTATION_PREFIXES) and ".py" in normalized_text:
                for token in text.split():
                    normalized_token = token.replace("\\", "/")
                    if token.endswith(".py") and any(
                        normalized_token.startswith(prefix) for prefix in IMPLEMENTATION_PREFIXES
                    ):
                        module_path = token
                        break
            if module_path:
                break

    if not module_path:
        return {"path": None, "exists": False}

    relative = normalize_relative_path(str(module_path))
    file_path = safe_path(relative)
    snapshot: dict[str, Any] = {
        "path": relative,
        "exists": file_path.exists(),
    }
    if not file_path.exists():
        return snapshot

    if file_path.is_dir():
        children = _directory_file_paths(relative)
        snapshot["isDirectory"] = True
        snapshot["files"] = children
        if children:
            snapshot["excerpt"] = _read_file_excerpt(children[0])
        return snapshot

    content = _safe_read_text(file_path)
    if content is None:
        snapshot["error"] = f"Could not read {relative}"
        return snapshot
    lines = content.splitlines()
    snapshot["lineCount"] = len(lines)
    if function_name:
        snapshot["containsFunction"] = str(function_name) in content
        snapshot["containsNotImplementedError"] = "NotImplementedError" in content
        excerpt = _extract_symbol_with_preamble(relative, str(function_name))
        if excerpt:
            snapshot["excerpt"] = excerpt
        else:
            snapshot["excerpt"] = _read_file_excerpt(relative)
    else:
        snapshot["excerpt"] = _read_file_excerpt(relative)
    return snapshot


def _dependency_status(task_id: str, task: dict[str, Any]) -> list[dict[str, str]]:
    states = _tasks.load_all_states(list(task.get("dependsOn", [])))
    rows: list[dict[str, str]] = []
    for dep_id in task.get("dependsOn", []):
        status = _tasks.effective_status(dep_id, states)
        rows.append({"taskId": dep_id, "status": status})
    return rows


def _deterministic_session_lessons(task_id: str, subtask_id: str) -> list[dict[str, Any]]:
    from orchestrator.session import SessionStore

    session = SessionStore(run_id="preflight-context")
    session.seed_subtask_lessons(task_id)
    task_lessons = session.tasks.get(task_id)
    if task_lessons is None:
        return []
    return [
        lesson
        for lesson in task_lessons.lessons
        if lesson.get("source") in ("subtask_data", "subtask_target", "subtask_scope")
    ]


def build_context_pack(task_id: str, subtask_id: str) -> dict[str, Any]:
    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    subtask = _tasks.get_subtask_spec_definition(task_id, subtask_id)
    if subtask is None:
        raise ValueError(f"Subtask not found for {task_id}: {subtask_id}")

    subtask_context = build_subtask_context(task_id, subtask_id=subtask_id)
    reference_snippets = _build_reference_snippets(subtask, task)
    repo_context = build_repo_context(subtask)

    return {
        "taskId": task_id,
        "subtaskId": subtask_id,
        "subtaskContext": subtask_context,
        "verificationChecklist": _tasks.build_verification_checklist(task_id),
        "dependencyStatus": _dependency_status(task_id, task),
        "targetFileSnapshot": _target_file_snapshot(subtask),
        "referenceSnippets": reference_snippets,
        "repoContext": repo_context,
        "domainKnowledge": _domain_knowledge(task_id, subtask_id),
        "explorationPolicy": _build_exploration_policy(
            subtask,
            subtask_context,
            reference_snippets,
            repo_context,
        ),
        "sessionLessons": _deterministic_session_lessons(task_id, subtask_id),
    }
