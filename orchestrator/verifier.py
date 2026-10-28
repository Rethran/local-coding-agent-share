"""Independent deterministic verification for completed subtask implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tools._common import safe_path

from orchestrator.read_guard import normalize_relative_path
from orchestrator.tool_adapters import safe_run_command
from orchestrator.verification_spec import executable_commands


@dataclass(frozen=True)
class VerificationEvidence:
    command: str
    passed: bool
    output: str


@dataclass(frozen=True)
class VerifierResult:
    passed: bool
    evidence: tuple[VerificationEvidence, ...]
    blockers: tuple[str, ...] = ()


def verification_commands(merged_subtask: dict[str, Any]) -> list[str]:
    """Return executable verification commands declared by a subtask."""
    return executable_commands(merged_subtask.get("verification"))


def command_exit_zero(output: str) -> bool:
    return output.lstrip().startswith("Exit code: 0")


def command_exit_code(output: str) -> int | None:
    first = output.lstrip().splitlines()[0] if output.strip() else ""
    prefix = "exit code:"
    lowered = first.lower()
    if not lowered.startswith(prefix):
        return None
    suffix = first.split(":", 1)[-1].strip()
    try:
        return int(suffix.split()[0])
    except (TypeError, ValueError, IndexError):
        return None


def command_collected_no_tests(output: str) -> bool:
    lowered = output.lower()
    return (
        "collected 0 item" in lowered
        or "no tests ran" in lowered
        or "no tests collected" in lowered
        or "no tests were found" in lowered
    )


def is_collection_or_syntax_error(output: str) -> bool:
    """True when pytest never ran assertions (collection/syntax/usage error)."""
    lowered = output.lower()
    if "syntaxerror" in lowered:
        return True
    if "error collecting" in lowered or "errors during collection" in lowered:
        return True
    if command_collected_no_tests(output):
        return True
    return command_exit_code(output) == 2


def is_verification_green(output: str) -> bool:
    return command_exit_zero(output) and not command_collected_no_tests(output)


def is_declared_verification_success(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    output: str,
    merged_subtask: dict[str, Any],
) -> bool:
    """Return whether a tool result proves an exact declared check passed."""
    if tool_name != "run_command" or status != "ok":
        return False
    command = str(arguments.get("command") or "").strip()
    return command in verification_commands(merged_subtask) and is_verification_green(output)


def is_declared_verification_failure(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    status: str,
    output: str,
    merged_subtask: dict[str, Any],
) -> bool:
    """Return whether a declared check ran and is red (test-writer success)."""
    if tool_name != "run_command" or status != "ok":
        return False
    command = str(arguments.get("command") or "").strip()
    if command not in verification_commands(merged_subtask):
        return False
    if is_collection_or_syntax_error(output):
        return False
    return not is_verification_green(output)


def _looks_like_file(path: str) -> bool:
    return bool(Path(path).suffix)


def artifact_blockers(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    target = merged_subtask.get("targetArtifacts") or {}
    created: set[str] = set()
    if isinstance(target, dict):
        for key in ("module", "test"):
            raw = str(target.get(key) or "").strip()
            if not raw:
                continue
            relative = normalize_relative_path(raw)
            created.add(relative)
            if not _looks_like_file(relative):
                continue
            if not safe_path(relative).is_file():
                blockers.append(
                    f"{task_id}.{subtask_id} missing {key} file: {relative}"
                )
    for artifact in merged_subtask.get("relatedArtifacts") or []:
        if not isinstance(artifact, dict):
            continue
        raw = str(artifact.get("path") or "").strip()
        if not raw:
            continue
        relative = normalize_relative_path(raw)
        if relative in created or not _looks_like_file(relative):
            continue
        if not safe_path(relative).is_file():
            blockers.append(
                f"{task_id}.{subtask_id} missing related artifact: {relative}"
            )
    return blockers


def target_call_blockers(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict[str, Any],
) -> list[str]:
    """Fail when the declared test never names the target function.

    A green pytest of an earlier module is not proof of this subtask.
    """
    target = merged_subtask.get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return []
    function = str(target.get("function") or "").strip()
    test_raw = str(target.get("test") or "").strip()
    if not function or not test_raw:
        return []
    relative = normalize_relative_path(test_raw)
    path = safe_path(relative)
    if not path.is_file():
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return [f"{task_id}.{subtask_id} cannot read declared test {relative}."]
    if function not in source:
        return [
            f"{task_id}.{subtask_id} declared test does not call {function}."
        ]
    return []


def run_verifier(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict[str, Any],
    *,
    command_runner: Callable[[str], str] = safe_run_command,
) -> VerifierResult:
    """Run declared checks after implementation and before the subtask commit."""
    missing = artifact_blockers(task_id, subtask_id, merged_subtask)
    missing.extend(target_call_blockers(task_id, subtask_id, merged_subtask))
    if missing:
        return VerifierResult(
            passed=False,
            evidence=(),
            blockers=tuple(missing),
        )

    commands = verification_commands(merged_subtask)
    if not commands:
        return VerifierResult(
            passed=False,
            evidence=(),
            blockers=(f"{task_id}.{subtask_id} declares no verification command.",),
        )

    evidence: list[VerificationEvidence] = []
    for command in commands:
        output = command_runner(command)
        passed = is_verification_green(output)
        evidence.append(
            VerificationEvidence(command=command, passed=passed, output=output)
        )
        if not passed:
            return VerifierResult(
                passed=False,
                evidence=tuple(evidence),
                blockers=(f"Verification failed: {command}",),
            )
    return VerifierResult(passed=True, evidence=tuple(evidence))
