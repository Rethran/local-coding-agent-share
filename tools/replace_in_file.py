"""Deterministic, atomic text replacement for focused code edits."""

from __future__ import annotations

import os
import tempfile

from tools._common import WORKSPACE_ROOT, mcp, safe_path

from orchestrator.python_write_gate import refuse_invalid_python_write


@mcp.tool()
def replace_in_file(path: str, old_text: str, new_text: str) -> str:
    """Replace exactly one UTF-8 text occurrence inside an existing workspace file."""
    file = safe_path(path)
    if not file.is_file():
        raise ValueError(f"File does not exist: {path}")
    if not old_text:
        raise ValueError("old_text must not be empty")

    content = file.read_text(encoding="utf-8")
    occurrences = content.count(old_text)
    if occurrences != 1:
        raise ValueError(
            f"Expected old_text to occur exactly once in {path}, found {occurrences}"
        )

    updated = content.replace(old_text, new_text, 1)
    refusal = refuse_invalid_python_write(path, updated)
    if refusal:
        raise ValueError(refusal)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=file.parent,
            prefix=f".{file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(updated)
            temp_path = temp_file.name
        os.replace(temp_path, file)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    return f"Replaced text in: {file.relative_to(WORKSPACE_ROOT)}"
