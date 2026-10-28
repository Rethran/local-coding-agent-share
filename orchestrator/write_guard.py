"""Role-based write allowlists for test-writer vs implementer workers."""

from __future__ import annotations

from typing import Any

from orchestrator.read_guard import normalize_relative_path


def _target_paths(merged_subtask: dict[str, Any] | None) -> tuple[str, str]:
    target = (merged_subtask or {}).get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return "", ""
    test_path = normalize_relative_path(str(target.get("test") or ""))
    module_path = normalize_relative_path(str(target.get("module") or ""))
    return test_path, module_path


def refuse_worker_write(
    path: str,
    *,
    role: str,
    merged_subtask: dict[str, Any] | None,
) -> str | None:
    """Return an error message when the current worker role cannot write this path."""
    relative = normalize_relative_path(path)
    if not relative:
        return "ERROR: Write path is empty."
    test_path, module_path = _target_paths(merged_subtask)

    if role == "test_writer":
        if module_path and relative == module_path and not relative.startswith("tests/"):
            return (
                f"ERROR: test-writer cannot modify production file {path}. "
                "Write failing tests only."
            )
        if relative.startswith("tests/"):
            return None
        if test_path and (relative == test_path or relative.startswith(f"{test_path}/")):
            return None
        return (
            f"ERROR: test-writer can only modify test or fixture files, not {path}."
        )

    if role == "implementer" and test_path and relative == test_path:
        return (
            f"ERROR: implementer cannot modify test file {path}. "
            "Implement production code only."
        )
    return None
