"""Golden-rule spec readiness gates before orchestrator coding attempts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from orchestrator import config
from tools._common import safe_path

from orchestrator.subtask_context import _parse_line_range
from orchestrator.verification_spec import (
    executable_commands,
    has_slice_selector,
    shared_test_file_siblings,
    sibling_subtasks,
    verification_lists_match,
)
from tools import _tasks


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    blockers: list[str]
    planner_hints: list[str]
    rule_ids: list[str]


def _targets_implementation_prefix(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    return any(normalized.startswith(prefix) for prefix in config.IMPLEMENTATION_PREFIXES)


def _scope_targets_implementation(subtask: dict[str, Any]) -> bool:
    for item in subtask.get("scope") or []:
        text = str(item).replace("\\", "/")
        if any(prefix in text for prefix in config.IMPLEMENTATION_PREFIXES):
            return True
    target = subtask.get("targetArtifacts") or {}
    if isinstance(target, dict):
        module = str(target.get("module") or "").replace("\\", "/")
        test_path = str(target.get("test") or "").replace("\\", "/")
        if _targets_implementation_prefix(module) or _targets_implementation_prefix(test_path):
            return True
    return False


def _has_verification(subtask: dict[str, Any], task: dict[str, Any]) -> bool:
    if subtask.get("verification"):
        return True
    impl = task.get("implementationLogic") or {}
    return bool(impl.get("verificationCommands"))


def _task_has_verification_path(task: dict[str, Any]) -> bool:
    if task.get("verification"):
        return True
    impl = task.get("implementationLogic") or {}
    if impl.get("verificationCommands"):
        return True
    for item in task.get("subtasks") or []:
        if isinstance(item, dict) and item.get("verification"):
            return True
    return False


def _is_derived_subtask(task: dict[str, Any], subtask: dict[str, Any]) -> bool:
    if _tasks.has_explicit_subtasks(task):
        return False
    has_verification = bool(subtask.get("verification"))
    target = subtask.get("targetArtifacts")
    has_target = isinstance(target, dict) and bool(target.get("module") or target.get("test"))
    return not has_verification or not has_target


def evaluate_spec_readiness(
    task_id: str,
    subtask_id: str,
    context_pack: dict[str, Any],
    merged_subtask: dict[str, Any],
) -> ReadinessResult:
    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    states = _tasks.load_all_states(list(task.get("dependsOn", [])))

    blockers: list[str] = []
    planner_hints: list[str] = []
    rule_ids: list[str] = []

    def fail(rule_id: str, message: str, *, planner: bool = False) -> None:
        rule_ids.append(rule_id)
        blockers.append(f"{rule_id}: {message}")
        if planner:
            planner_hints.append(message)

    acceptance = str(task.get("acceptanceCriteria") or "").strip()
    if not acceptance:
        fail("GR-TASK-001", "Task acceptanceCriteria is empty.")

    for dep in task.get("dependsOn", []):
        status = _tasks.effective_status(dep, states)
        if status != "DONE":
            fail("GR-TASK-002", f"Dependency {dep} is {status}, not DONE.")

    if not _task_has_verification_path(task):
        fail("GR-TASK-003", "Task has no verification path.", planner=True)

    scope = merged_subtask.get("scope") or []
    if not scope:
        fail("GR-SUB-001", f"Subtask {subtask_id} scope is empty.", planner=True)

    if not _has_verification(merged_subtask, task):
        fail("GR-SUB-002", f"Subtask {subtask_id} has no verification commands.", planner=True)

    if not (merged_subtask.get("title") or merged_subtask.get("summary")):
        fail("GR-SUB-003", f"Subtask {subtask_id} needs title or summary.", planner=True)

    current_commands = executable_commands(merged_subtask.get("verification"))
    if merged_subtask.get("verification") and not current_commands:
        fail(
            "GR-SUB-005",
            f"Subtask {subtask_id} verification must include an executable command.",
            planner=True,
        )

    for sibling in sibling_subtasks(task, subtask_id):
        sibling_commands = executable_commands(sibling.get("verification"))
        if current_commands and sibling_commands and verification_lists_match(
            current_commands,
            sibling_commands,
        ):
            fail(
                "GR-SUB-004",
                f"Subtask {subtask_id} verification matches sibling {sibling.get('id')}.",
                planner=True,
            )
            break

    if shared_test_file_siblings(task, merged_subtask, subtask_id) and current_commands:
        if not all(has_slice_selector(command) for command in current_commands):
            fail(
                "GR-SUB-006",
                f"Subtask {subtask_id} shares a test file and needs a -k or :: selector.",
                planner=True,
            )

    if _is_derived_subtask(task, merged_subtask):
        fail(
            "GR-DERIVED-001",
            f"Subtask {subtask_id} is scope-line-only; enrich with verification, "
            "targetArtifacts, and legacy metadata.",
            planner=True,
        )

    read_legacy = merged_subtask.get("readLegacy")
    if read_legacy is True:
        symbols = merged_subtask.get("legacySymbols") or []
        if not symbols:
            fail(
                "GR-LEGACY-001",
                f"Subtask {subtask_id} requires legacySymbols when readLegacy is true.",
                planner=True,
            )
        else:
            for index, symbol in enumerate(symbols, start=1):
                if not isinstance(symbol, dict):
                    fail("GR-LEGACY-002", f"legacySymbols[{index}] is not an object.", planner=True)
                    continue
                path = str(symbol.get("path") or "").strip()
                name = str(symbol.get("name") or "").strip()
                line_range = str(symbol.get("relevantLines") or "").strip()
                if not path or not name or _parse_line_range(line_range) is None:
                    fail(
                        "GR-LEGACY-002",
                        f"legacySymbols[{index}] needs path, name, and parseable relevantLines.",
                        planner=True,
                    )
                elif not safe_path(path).exists():
                    fail("GR-LEGACY-003", f"Legacy file not found: {path}", planner=True)

            slices = context_pack.get("subtaskContext", {}).get("legacySlices") or []
            if symbols and slices:
                usable = [
                    item for item in slices
                    if item.get("source") or not item.get("error")
                ]
                if not usable:
                    fail(
                        "GR-LEGACY-004",
                        "Context pack legacy slices are empty or unreadable.",
                        planner=True,
                    )

    if read_legacy is False:
        if not _scope_targets_implementation(merged_subtask):
            fail(
                "GR-SKELETON-001",
                f"Subtask {subtask_id} must name a target under configured implementation paths.",
                planner=True,
            )
        if merged_subtask.get("legacySymbols"):
            fail(
                "GR-SKELETON-002",
                f"Subtask {subtask_id} forbids legacySymbols when readLegacy is false.",
            )

    return ReadinessResult(
        ready=not blockers,
        blockers=blockers,
        planner_hints=planner_hints,
        rule_ids=rule_ids,
    )


def is_subtask_done(task_id: str, subtask_id: str) -> bool:
    state = _tasks.load_task_state(task_id) or {}
    for item in state.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return item.get("status") == "DONE"
    return False
