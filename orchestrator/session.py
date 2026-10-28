"""Session-scoped memory for orchestrator continuous delivery."""

from __future__ import annotations

import json
import re
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from orchestrator.config import (
    ATTEMPT_REPORTS_DIR,
    DEFAULT_MAX_READ_ONLY_BEFORE_WRITE,
    DEFAULT_READ_ONLY_WARN_THRESHOLD,
    READ_ONLY_TOOLS,
    TRUNCATED_EXCERPT_MARKER,
)
from orchestrator.read_guard import normalize_relative_path
from tools import _handoffs, _tasks
from tools._common import safe_path

SESSIONS_DIR = ATTEMPT_REPORTS_DIR / "sessions"
MAX_LESSONS_PER_TASK = 15

DISCOVERY_PHRASES = (
    "price.amount",
    "implementation is missing",
    "flattened",
    "in_domain.mrid",
    "do not create temp",
    "tmp_debug",
)

TEMP_SCRIPT_PATTERN = re.compile(r"^tmp_.*\.py$", re.IGNORECASE)
TEST_COMMAND_PATTERN = re.compile(
    r"(?:^|[;&|]\s*)(?:python\s+-m\s+)?(?:pytest|unittest)(?:\s|$)",
    re.IGNORECASE,
)

_current_session: ContextVar["SessionStore | None"] = ContextVar(
    "orchestrator_session",
    default=None,
)


def get_session() -> SessionStore | None:
    return _current_session.get()


def set_session(session: SessionStore | None) -> None:
    _current_session.set(session)


def _load_active_subtask(task_id: str) -> dict[str, Any] | None:
    from orchestrator.search_guard import load_active_subtask_definition

    return load_active_subtask_definition(task_id)


@dataclass
class TaskSessionState:
    tool_records: dict[str, dict[str, Any]] = field(default_factory=dict)
    read_mtimes: dict[str, float] = field(default_factory=dict)
    written_paths: set[str] = field(default_factory=set)
    lessons: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    subtask_context_cache: dict[str, Any] = field(default_factory=dict)
    last_handoff_id: str = ""
    visible_outputs: list[str] = field(default_factory=list)
    lessons_seeded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)
    context_pack_preflight_satisfied: bool = False
    read_only_tools_used: int = 0
    has_successful_write: bool = False
    max_read_only_before_write: int = DEFAULT_MAX_READ_ONLY_BEFORE_WRITE
    read_only_warn_threshold: int = DEFAULT_READ_ONLY_WARN_THRESHOLD
    truncated_handoff_ids: set[str] = field(default_factory=set)
    expanded_handoff_ids: set[str] = field(default_factory=set)
    preloaded_paths: set[str] = field(default_factory=set)
    target_module_path: str = ""
    target_module_read: bool = False
    pending_next_action: dict[str, Any] | None = None


class SessionStore:
    """In-memory session state for one main.py run."""

    def __init__(self, run_id: str | None = None) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.tasks: dict[str, TaskSessionState] = {}
        self.started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @classmethod
    def create_and_set(cls) -> SessionStore:
        session = cls()
        set_session(session)
        return session

    def _task(self, task_id: str) -> TaskSessionState:
        if task_id not in self.tasks:
            self.tasks[task_id] = TaskSessionState()
        return self.tasks[task_id]

    def normalize_fingerprint(self, tool: str, args: dict[str, Any]) -> str:
        if tool == "read_file":
            return f"read_file:{normalize_relative_path(str(args.get('path', '')))}"
        if tool == "extract_symbol":
            path = normalize_relative_path(str(args.get("path", ".")))
            start = args.get("start_line")
            end = args.get("end_line")
            line_part = f":{start}-{end}" if start is not None or end is not None else ""
            return f"extract_symbol:{args.get('name', '')}:{path}{line_part}"
        if tool == "search_files":
            query = str(args.get("query", "")).strip().lower()
            path = normalize_relative_path(str(args.get("path", ".")))
            return f"search_files:{query}:{path}"
        if tool == "get_subtask_context":
            subtask_id = args.get("_subtask_id") or args.get("subtask_id") or ""
            return f"get_subtask_context:{args.get('task_id', '')}:{subtask_id}"
        if tool == "get_active_subtask":
            return f"get_active_subtask:{args.get('task_id', '')}"
        if tool in {"write_file", "replace_in_file"}:
            return f"{tool}:{normalize_relative_path(str(args.get('path', '')))}"
        if tool == "read_handoff":
            handoff_id = str(args.get("handoff_id") or args.get("handoffId") or "")
            handoff_path = normalize_relative_path(str(args.get("handoff_path") or ""))
            include = bool(args.get("include_result"))
            return f"read_handoff:{handoff_id}:{handoff_path}:{include}"
        if tool == "list_directory":
            return f"list_directory:{normalize_relative_path(str(args.get('path', '.')))}"
        return f"{tool}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"

    def _file_mtime(self, relative_path: str) -> float | None:
        try:
            return safe_path(relative_path).stat().st_mtime
        except OSError:
            return None

    def _subtask_forbids_temp_scripts(self, task_id: str) -> bool:
        subtask = _load_active_subtask(task_id)
        if subtask is None:
            return False
        instructions = " ".join(str(item) for item in subtask.get("instructions") or [])
        combined = instructions.lower()
        return "temp" in combined and ("script" in combined or "inspection" in combined)

    def _add_lesson(
        self,
        task_id: str,
        *,
        kind: str,
        text: str,
        source: str = "heuristic",
        handoff_id: str | None = None,
    ) -> None:
        task = self._task(task_id)
        normalized = text.strip().lower()
        for existing in task.lessons:
            if existing.get("text", "").strip().lower() == normalized:
                existing.update(
                    {
                        "kind": kind,
                        "text": text.strip(),
                        "source": source,
                        "handoffId": handoff_id or existing.get("handoffId"),
                    }
                )
                return

        lesson_id = f"{kind}-{len(task.lessons) + 1}"
        task.lessons.append(
            {
                "id": lesson_id,
                "kind": kind,
                "text": text.strip(),
                "source": source,
                "handoffId": handoff_id,
            }
        )
        if len(task.lessons) > MAX_LESSONS_PER_TASK:
            task.lessons = task.lessons[-MAX_LESSONS_PER_TASK:]

    def set_exploration_policy(
        self,
        task_id: str,
        *,
        preflight_satisfied: bool,
        max_read_only_before_write: int | None = None,
    ) -> None:
        task = self._task(task_id)
        task.context_pack_preflight_satisfied = preflight_satisfied
        if max_read_only_before_write is not None:
            task.max_read_only_before_write = max(int(max_read_only_before_write), 1)

    def set_preloaded_context(
        self,
        task_id: str,
        *,
        preloaded_paths: set[str] | list[str] | None = None,
        target_module_path: str | None = None,
    ) -> None:
        task = self._task(task_id)
        if preloaded_paths:
            task.preloaded_paths = {
                normalize_relative_path(path)
                for path in preloaded_paths
                if path
            }
        if target_module_path:
            task.target_module_path = normalize_relative_path(target_module_path)

    def get_read_only_tools_used(self, task_id: str) -> int:
        return self._task(task_id).read_only_tools_used

    def note_truncated_handoff(self, task_id: str, handoff_id: str, excerpt: str) -> None:
        if not handoff_id or TRUNCATED_EXCERPT_MARKER not in excerpt:
            return
        self._task(task_id).truncated_handoff_ids.add(handoff_id)

    def exploration_budget_warning(self, task_id: str) -> str | None:
        task = self._task(task_id)
        if task.has_successful_write:
            return None
        if task.read_only_tools_used >= task.read_only_warn_threshold:
            remaining = max(task.max_read_only_before_write - task.read_only_tools_used, 0)
            return (
                f"Exploration budget: {task.read_only_tools_used} read-only tools used; "
                f"{remaining} remaining before replace_in_file or write_file is required."
            )
        return None

    def _check_exploration_budget(self, task_id: str, tool: str) -> str | None:
        if tool not in READ_ONLY_TOOLS:
            return None
        task = self._task(task_id)
        if task.has_successful_write:
            return None
        if task.read_only_tools_used >= task.max_read_only_before_write:
            return (
                f"ERROR: Exploration budget exhausted ({task.max_read_only_before_write} "
                "read-only tools). Proceed to replace_in_file or write_file."
            )
        return None

    def seed_subtask_lessons(self, task_id: str) -> None:
        task = self._task(task_id)
        if task.lessons_seeded:
            return
        task.lessons_seeded = True

        subtask = _load_active_subtask(task_id)
        if subtask is None:
            return

        for example in subtask.get("dataExamples") or []:
            self._add_lesson(
                task_id,
                kind="discovery",
                text=str(example),
                source="subtask_dataExamples",
            )

        if subtask.get("readLegacy") and self._subtask_forbids_temp_scripts(task_id):
            self._add_lesson(
                task_id,
                kind="do_not_repeat",
                text="Do not create temp inspection scripts; use dataExamples and tests only.",
                source="subtask_instructions",
            )

        target = subtask.get("targetArtifacts") or {}
        if isinstance(target, dict):
            module = target.get("module")
            function = target.get("function")
            test_path = target.get("test")
            if module and function:
                self._add_lesson(
                    task_id,
                    kind="next_action",
                    text=f"Implement {function} in {module}.",
                    source="subtask_target",
                )
            if test_path:
                self._add_lesson(
                    task_id,
                    kind="artifact",
                    text=f"Tests belong in {test_path}.",
                    source="subtask_target",
                )

    def get_tool_record(
        self,
        task_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> dict[str, Any] | None:
        fingerprint_args = dict(args)
        if tool == "get_subtask_context":
            subtask = _load_active_subtask(task_id)
            fingerprint_args["_subtask_id"] = subtask.get("id") if subtask else ""
        fingerprint = self.normalize_fingerprint(tool, fingerprint_args)
        return self._task(task_id).tool_records.get(fingerprint)

    def check_before_execute(
        self,
        task_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> str | None:
        """Return None to proceed, ERROR to refuse, or CACHED: result."""
        task = self._task(task_id)
        with task.lock:
            return self._check_before_execute_locked(task_id, tool, args)

    def _check_before_execute_locked(
        self,
        task_id: str,
        tool: str,
        args: dict[str, Any],
    ) -> str | None:
        self.seed_subtask_lessons(task_id)
        task = self._task(task_id)

        budget_error = self._check_exploration_budget(task_id, tool)
        if budget_error:
            return budget_error

        if tool == "get_subtask_context" and task.context_pack_preflight_satisfied:
            return (
                "ERROR: contextPack already includes legacySlices and referenceSnippets. "
                "Do not call get_subtask_context; proceed to replace_in_file or write_file."
            )

        if (
            not task.context_pack_preflight_satisfied
            and task.pending_next_action
            and task.pending_next_action.get("tool") in {"write_file", "replace_in_file"}
            and tool in READ_ONLY_TOOLS
            and not (
                tool == "read_handoff"
                and args.get("include_result")
                and str(args.get("handoff_id") or args.get("handoffId") or "")
                in task.truncated_handoff_ids
                and str(args.get("handoff_id") or args.get("handoffId") or "")
                not in task.expanded_handoff_ids
            )
        ):
            hint = str(task.pending_next_action.get("hint") or "").strip()
            hint_text = f" ({hint})" if hint else ""
            pending_tool = task.pending_next_action.get("tool")
            return (
                f"ERROR: Pending next action is {pending_tool}{hint_text}. "
                "Do not explore further until you write."
            )

        if tool == "read_file" and task.context_pack_preflight_satisfied:
            path = normalize_relative_path(str(args.get("path", "")))
            if path in task.preloaded_paths:
                return (
                    f"ERROR: contextPack.referenceSnippets already includes {path!r}. "
                    "Proceed to replace_in_file or write_file."
                )
            if (
                task.target_module_path
                and path == task.target_module_path
                and task.target_module_read
            ):
                return (
                    f"ERROR: Duplicate read_file({path!r}) on target module this session. "
                    "Proceed to replace_in_file or write_file."
                )

        if tool == "write_file":
            path = normalize_relative_path(str(args.get("path", "")))
            if TEMP_SCRIPT_PATTERN.match(Path(path).name):
                if self._subtask_forbids_temp_scripts(task_id):
                    return (
                        "ERROR: Refusing write_file on temp inspection script "
                        f"({path}). Fixture shape is in sessionLessons/dataExamples; "
                        "proceed to implement target module or run tests."
                    )

        fingerprint_args = dict(args)
        if tool == "get_subtask_context":
            subtask = _load_active_subtask(task_id)
            fingerprint_args["_subtask_id"] = subtask.get("id") if subtask else ""
            cache_key = fingerprint_args["_subtask_id"]
            if cache_key and cache_key in task.subtask_context_cache:
                cached = task.subtask_context_cache[cache_key]
                return f"CACHED:{json.dumps(cached, indent=2, ensure_ascii=False)}"

        if tool == "run_command":
            command = str(args.get("command", ""))
            read_paths: dict[str, str] = {}
            for path in task.preloaded_paths:
                read_paths[path] = ""
            if task.target_module_path and task.target_module_read:
                read_paths[task.target_module_path] = ""
            for key, record in task.tool_records.items():
                if record.get("tool") != "read_file" or record.get("status") != "ok":
                    continue
                path = key.split(":", 1)[1] if ":" in key else ""
                if path:
                    read_paths[path] = str(record.get("handoffId", ""))
            from orchestrator.continuity import is_cat_file_command

            matched = is_cat_file_command(command, set(read_paths.keys()))
            if matched:
                read_handoff_id = read_paths.get(matched, "")
                if not read_handoff_id:
                    return (
                        f"ERROR: Refusing run_command that reads guarded path {matched!r}. "
                        "Its content is already in contextPack; proceed to "
                        "replace_in_file or run verification."
                    )
                if read_handoff_id in task.truncated_handoff_ids:
                    return (
                        f"ERROR: Refusing run_command that re-reads {matched!r} already "
                        f"loaded via read_file (handoff {read_handoff_id}). "
                        f"Use read_handoff(handoff_id={read_handoff_id!r}, include_result=true) "
                        "once to load the full result, or proceed to replace_in_file."
                    )
                return (
                    f"ERROR: Refusing run_command that re-reads {matched!r} already "
                    f"loaded via read_file (handoff {read_handoff_id}). Proceed to "
                    "replace_in_file or run verification."
                )

        if tool == "read_handoff":
            handoff_id = str(args.get("handoff_id") or args.get("handoffId") or "")
            if (
                args.get("include_result")
                and handoff_id
                and handoff_id in task.truncated_handoff_ids
                and handoff_id not in task.expanded_handoff_ids
            ):
                return None

        fingerprint = self.normalize_fingerprint(tool, fingerprint_args)
        prior = task.tool_records.get(fingerprint)
        if prior is None:
            return None
        if prior.get("status") == "error":
            if tool != "read_file":
                return None
            path = normalize_relative_path(str(args.get("path", "")))
            if path in task.written_paths:
                task.written_paths.discard(path)
                return None
            return (
                f"ERROR: Repeated read_file({path!r}) was already refused "
                f"(handoff {prior.get('handoffId', '')}). "
                "Do not repeat it; use the prior error's allowed next action."
            )
        if prior.get("status") != "ok":
            return None

        handoff_id = prior.get("handoffId", "")

        if tool == "read_file":
            path = normalize_relative_path(str(args.get("path", "")))
            if args.get("force"):
                return None
            current_mtime = self._file_mtime(path)
            last_mtime = task.read_mtimes.get(path)
            if path in task.written_paths:
                task.written_paths.discard(path)
                return None
            if current_mtime is not None and last_mtime is not None and current_mtime != last_mtime:
                return None
            if (
                handoff_id in task.truncated_handoff_ids
                and handoff_id not in task.expanded_handoff_ids
            ):
                return (
                    f"ERROR: Duplicate read_file({path!r}) in this session "
                    f"(handoff {handoff_id}). Use read_handoff(handoff_id={handoff_id!r}, "
                    "include_result=true) once to load the full result, or proceed to "
                    "replace_in_file."
                )
            return (
                f"ERROR: Duplicate read_file({path!r}) in this session "
                f"(handoff {handoff_id}). Proceed to replace_in_file or run verification."
            )

        if tool == "read_handoff":
            requested_id = str(args.get("handoff_id") or args.get("handoffId") or "")
            if (
                args.get("include_result")
                and requested_id
                and requested_id in task.truncated_handoff_ids
                and requested_id not in task.expanded_handoff_ids
            ):
                return None
            return (
                f"ERROR: Duplicate {tool} in this session (handoff {handoff_id}). "
                "Proceed to replace_in_file or run verification."
            )

        if tool in ("extract_symbol", "search_files", "get_active_subtask", "list_directory"):
            return (
                f"ERROR: Duplicate {tool} in this session (handoff {handoff_id}). "
                "Proceed to replace_in_file or run verification."
            )

        return None

    def record_after_execute(
        self,
        task_id: str,
        tool: str,
        args: dict[str, Any],
        status: str,
        handoff_id: str,
        raw_result: str,
        *,
        next_action: dict[str, Any] | None = None,
    ) -> None:
        task = self._task(task_id)
        with task.lock:
            self._record_after_execute_locked(
                task_id,
                tool,
                args,
                status,
                handoff_id,
                raw_result,
                next_action=next_action,
            )

    def _record_after_execute_locked(
        self,
        task_id: str,
        tool: str,
        args: dict[str, Any],
        status: str,
        handoff_id: str,
        raw_result: str,
        *,
        next_action: dict[str, Any] | None = None,
    ) -> None:
        task = self._task(task_id)
        if handoff_id:
            task.last_handoff_id = handoff_id

        fingerprint_args = dict(args)
        if tool == "get_subtask_context":
            subtask = _load_active_subtask(task_id)
            fingerprint_args["_subtask_id"] = subtask.get("id") if subtask else ""
        fingerprint = self.normalize_fingerprint(tool, fingerprint_args)

        if status == "ok":
            task.tool_records[fingerprint] = {
                "tool": tool,
                "args": args,
                "handoffId": handoff_id,
                "status": status,
            }
            if tool in READ_ONLY_TOOLS:
                task.read_only_tools_used += 1
                warning = self.exploration_budget_warning(task_id)
                if warning:
                    self._add_lesson(
                        task_id,
                        kind="do_not_repeat",
                        text=warning,
                        source="exploration_budget",
                    )
        elif tool == "read_file":
            task.tool_records[fingerprint] = {
                "tool": tool,
                "args": args,
                "handoffId": handoff_id,
                "status": status,
                "error": raw_result[:500],
            }
            self._add_lesson(
                task_id,
                kind="do_not_repeat",
                text=(
                    f"Do not repeat read_file for "
                    f"{normalize_relative_path(str(args.get('path', '')))}; "
                    f"use the allowed next action from handoff {handoff_id}."
                ),
                source="read_refusal",
                handoff_id=handoff_id,
            )

        if tool == "read_file" and status == "ok":
            path = normalize_relative_path(str(args.get("path", "")))
            mtime = self._file_mtime(path)
            if mtime is not None:
                task.read_mtimes[path] = mtime
            if task.target_module_path and path == task.target_module_path:
                task.target_module_read = True

        if tool in ("read_file", "extract_symbol", "get_subtask_context") and status == "ok":
            if not task.context_pack_preflight_satisfied and next_action:
                task.pending_next_action = next_action

        if tool in {"write_file", "replace_in_file"} and status == "ok":
            path = normalize_relative_path(str(args.get("path", "")))
            task.has_successful_write = True
            task.read_only_tools_used = 0
            task.pending_next_action = None
            task.written_paths.add(path)
            task.artifacts.append(
                {
                    "path": path,
                    "status": "written",
                    "handoffId": handoff_id,
                }
            )
            self._add_lesson(
                task_id,
                kind="artifact",
                text=f"{path} written or updated (handoff {handoff_id}).",
                source="tool",
                handoff_id=handoff_id,
            )

        if tool == "run_command" and status == "ok":
            command = str(args.get("command", ""))
            if TEST_COMMAND_PATTERN.search(command):
                exit_line = raw_result.splitlines()[0] if raw_result.splitlines() else ""
                if "Exit code: 1" in exit_line or "Exit code: 2" in exit_line:
                    self._add_lesson(
                        task_id,
                        kind="gap",
                        text=f"Tests still failing: {exit_line}",
                        source="tool",
                        handoff_id=handoff_id,
                    )
                elif "Exit code: 0" in exit_line:
                    self._add_lesson(
                        task_id,
                        kind="verified",
                        text=f"Tests passed: {command}",
                        source="tool",
                        handoff_id=handoff_id,
                    )

        if tool == "get_subtask_context" and status == "ok":
            subtask = _load_active_subtask(task_id)
            subtask_id = subtask.get("id") if subtask else ""
            if subtask_id:
                try:
                    task.subtask_context_cache[subtask_id] = json.loads(raw_result)
                except json.JSONDecodeError:
                    pass

        if tool == "read_handoff" and status == "ok":
            requested_id = str(args.get("handoff_id") or args.get("handoffId") or handoff_id)
            if args.get("include_result") and requested_id in task.truncated_handoff_ids:
                task.expanded_handoff_ids.add(requested_id)

        if tool == "extract_symbol" and status == "ok":
            name = args.get("name", "?")
            self._add_lesson(
                task_id,
                kind="discovery",
                text=f"Legacy symbol {name!r} already extracted (handoff {handoff_id}).",
                source="tool",
                handoff_id=handoff_id,
            )

    def record_visible_output(self, task_id: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        task = self._task(task_id)
        task.visible_outputs.append(cleaned)
        if len(task.visible_outputs) > 20:
            task.visible_outputs = task.visible_outputs[-20:]

        lowered = cleaned.lower()
        for phrase in DISCOVERY_PHRASES:
            if phrase in lowered:
                self._add_lesson(
                    task_id,
                    kind="discovery",
                    text=cleaned[:500],
                    source="visible_output",
                )
                break

    def _list_chain_handoffs(self, task_id: str) -> list[dict[str, Any]]:
        directory = _handoffs.chain_dir(task_id)
        if not directory.exists():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(directory.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return records

    def extract_lessons_from_handoffs(self, task_id: str, state: dict[str, Any]) -> None:
        self.seed_subtask_lessons(task_id)

        extract_counts: dict[str, int] = {}
        for record in self._list_chain_handoffs(task_id):
            tool = record.get("tool", "")
            handoff_id = record.get("handoffId", "")
            input_data = record.get("input") or {}
            result = record.get("result") or {}
            result_status = result.get("status")
            if result_status != "ok":
                continue

            if tool in {"write_file", "replace_in_file"}:
                path = normalize_relative_path(str(input_data.get("path", "")))
                if path.endswith(".py") and "test" in path:
                    self._add_lesson(
                        task_id,
                        kind="artifact",
                        text=f"{path} exists with tests (handoff {handoff_id}).",
                        source="handoff",
                        handoff_id=handoff_id,
                    )

            if tool == "extract_symbol":
                name = str(input_data.get("name", ""))
                key = self.normalize_fingerprint("extract_symbol", input_data)
                extract_counts[key] = extract_counts.get(key, 0) + 1
                if extract_counts[key] >= 2:
                    self._add_lesson(
                        task_id,
                        kind="do_not_repeat",
                        text=(
                            f"Do not re-extract {name!r}; already available "
                            f"(handoff {handoff_id})."
                        ),
                        source="handoff",
                        handoff_id=handoff_id,
                    )

            if tool == "run_command":
                command = str(input_data.get("command", ""))
                output = str(result.get("output", ""))
                if ("unittest" in command or "pytest" in command) and "Exit code: 1" in output:
                    self._add_lesson(
                        task_id,
                        kind="gap",
                        text="Implementation missing or incomplete; tests fail.",
                        source="handoff",
                        handoff_id=handoff_id,
                    )

        progress = state.get("progressSummary") or ""
        if progress:
            self._add_lesson(
                task_id,
                kind="discovery",
                text=progress,
                source="state",
            )

        for path in state.get("partialFilesTouched") or []:
            normalized = normalize_relative_path(str(path))
            from orchestrator import config

            if any(normalized.startswith(prefix) for prefix in config.IMPLEMENTATION_PREFIXES):
                self._add_lesson(
                    task_id,
                    kind="artifact",
                    text=f"Partial work exists in {normalized}.",
                    source="state",
                )

        active_id = state.get("activeSubtaskId")
        subtask = _load_active_subtask(task_id)
        if subtask and subtask.get("id") == active_id:
            target = subtask.get("targetArtifacts") or {}
            if isinstance(target, dict):
                module = target.get("module")
                function = target.get("function")
                if module and function:
                    module_path = normalize_relative_path(str(module))
                    try:
                        content = safe_path(module_path).read_text(encoding="utf-8")
                        if function not in content or "NotImplementedError" in content:
                            self._add_lesson(
                                task_id,
                                kind="gap",
                                text=f"{function} not implemented in {module_path}.",
                                source="heuristic",
                            )
                    except OSError:
                        self._add_lesson(
                            task_id,
                            kind="gap",
                            text=f"{module_path} missing or unreadable.",
                            source="heuristic",
                        )

        read_paths = [
            key.split(":", 1)[1]
            for key, record in self._task(task_id).tool_records.items()
            if record.get("tool") == "read_file" and record.get("status") == "ok"
        ]
        if len(read_paths) >= 2:
            preview = ", ".join(sorted(set(read_paths))[:4])
            self._add_lesson(
                task_id,
                kind="do_not_repeat",
                text=f"Do not re-read {preview}; contents unchanged this session.",
                source="heuristic",
            )

    def extract_lessons_on_failure(self, task_id: str, state: dict[str, Any]) -> None:
        self.extract_lessons_from_handoffs(task_id, state)
        failure = state.get("lastFailureReason") or ""
        if failure:
            self._add_lesson(
                task_id,
                kind="failure",
                text=failure[:400],
                source="state",
            )
        resume = state.get("recommendedResumeStep") or ""
        if resume:
            self._add_lesson(
                task_id,
                kind="next_action",
                text=resume,
                source="state",
            )

    def snapshot_before_handover(self, task_id: str, state: dict[str, Any]) -> None:
        self.extract_lessons_from_handoffs(task_id, state)

    def get_cached_subtask_context(self, task_id: str) -> dict[str, Any] | None:
        task = self._task(task_id)
        subtask = _load_active_subtask(task_id)
        if subtask is None:
            return None
        subtask_id = subtask.get("id")
        if subtask_id and subtask_id in task.subtask_context_cache:
            return task.subtask_context_cache[subtask_id]
        try:
            from orchestrator.subtask_context import build_subtask_context

            context = build_subtask_context(task_id)
            if subtask_id:
                task.subtask_context_cache[subtask_id] = context
            return context
        except ValueError:
            return None

    def get_session_payload_for_retry(
        self,
        task_id: str,
        *,
        include_cached_context: bool = True,
    ) -> dict[str, Any]:
        task = self._task(task_id)
        self.seed_subtask_lessons(task_id)
        resume_brief = _tasks.build_resume_brief(
            task_id,
            last_handoff_id=task.last_handoff_id or None,
        )
        payload: dict[str, Any] = {
            "sessionLessons": list(task.lessons),
            "resumeBrief": resume_brief,
            "knownArtifacts": list(task.artifacts),
            "continueFromHandoff": task.last_handoff_id or None,
        }
        if include_cached_context:
            payload["cachedSubtaskContext"] = self.get_cached_subtask_context(task_id)
        return payload

    def should_inject_on_first_attempt(self, task_id: str) -> bool:
        packet = _tasks.get_task_packet(task_id)
        state = packet.get("state") or {}
        if state.get("status") != "IN_PROGRESS":
            return False
        task = self._task(task_id)
        if task.lessons or task.tool_records or task.subtask_context_cache:
            return True
        directory = _handoffs.chain_dir(task_id)
        return directory.exists() and any(directory.glob("*.json"))

    def persist(self) -> Path:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        path = SESSIONS_DIR / f"{self.run_id}.json"
        payload = {
            "runId": self.run_id,
            "startedAt": self.started_at,
            "tasks": {
                task_id: {
                    "lessons": task.lessons,
                    "artifacts": task.artifacts,
                    "lastHandoffId": task.last_handoff_id,
                    "subtaskContextCache": task.subtask_context_cache,
                    "toolFingerprints": list(task.tool_records.keys()),
                }
                for task_id, task in self.tasks.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
