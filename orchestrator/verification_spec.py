"""Project-agnostic classification of subtask verification commands."""

from __future__ import annotations

import re
from typing import Any

COMMAND_RUNNERS = frozenset({
    "python",
    "python3",
    "py",
    "pytest",
    "npm",
    "npx",
    "pnpm",
    "yarn",
    "cargo",
    "go",
    "unittest",
    "node",
    "uv",
    "poetry",
    "dotnet",
    "mvn",
    "gradle",
    "make",
})

_SLICE_SELECTOR = re.compile(r"(?:^|\s)(-k|--keyword|::)\b")
_NON_ALNUM = re.compile(r"[^A-Za-z0-9]+")


def slice_token(subtask_id: str) -> str:
    """Return a pytest/unittest -k token that is not an arithmetic expression."""
    return _NON_ALNUM.sub("_", subtask_id).strip("_")


def is_executable_command(command: str) -> bool:
    text = str(command or "").strip()
    if not text:
        return False
    first = text.split(maxsplit=1)[0]
    return first.lower() in COMMAND_RUNNERS


def executable_commands(verification: list[Any] | None) -> list[str]:
    return [
        str(item).strip()
        for item in verification or []
        if isinstance(item, str) and is_executable_command(item)
    ]


def has_slice_selector(command: str) -> bool:
    return bool(_SLICE_SELECTOR.search(command))


def add_slice_selector(command: str, token: str) -> str:
    """Insert -k TOKEN before trailing dash flags when the command has no selector."""
    stripped = command.strip()
    if not stripped or has_slice_selector(stripped):
        return stripped
    if "pytest" not in stripped and "unittest" not in stripped:
        return stripped
    parts = stripped.split()
    index = len(parts)
    while index > 0 and parts[index - 1].startswith("-"):
        index -= 1
    parts.insert(index, "-k")
    parts.insert(index + 1, token)
    return " ".join(parts)


def verification_lists_match(left: list[str], right: list[str]) -> bool:
    return [item.strip() for item in left] == [item.strip() for item in right]


def sibling_subtasks(task: dict[str, Any], subtask_id: str) -> list[dict[str, Any]]:
    siblings: list[dict[str, Any]] = []
    for item in task.get("subtasks") or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "")
        if item_id and item_id != subtask_id:
            siblings.append(item)
    return siblings


def shared_test_file_siblings(
    task: dict[str, Any],
    merged_subtask: dict[str, Any],
    subtask_id: str = "",
) -> bool:
    target = merged_subtask.get("targetArtifacts") or {}
    test_path = str(target.get("test") or "").replace("\\", "/").strip()
    if not test_path:
        return False
    current_id = subtask_id or str(merged_subtask.get("id") or "")
    for sibling in sibling_subtasks(task, current_id):
        sibling_target = sibling.get("targetArtifacts") or {}
        if not isinstance(sibling_target, dict):
            continue
        sibling_test = str(sibling_target.get("test") or "").replace("\\", "/").strip()
        if sibling_test == test_path:
            return True
    return False
