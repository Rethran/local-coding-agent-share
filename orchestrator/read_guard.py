"""Guardrails for orchestrator read_file calls."""

from __future__ import annotations

from pathlib import Path

from tools._common import ensure_file_exists, safe_path
from tools._handoffs import CONSUMED_DIR, HANDOFFS_DIR

from orchestrator.config import (
    READ_FILE_ALLOWED_PREFIXES,
    READ_FILE_ALLOWED_SUFFIXES,
    READ_FILE_MAX_PY_BYTES,
    READ_FILE_MAX_PY_LINES,
    READ_FILE_SMALL_FILE_BYTES,
)


def normalize_relative_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_handoff_path(resolved: Path) -> bool:
    """True when the path is under active or consumed handoff storage."""
    try:
        resolved.relative_to(HANDOFFS_DIR.resolve())
        return True
    except ValueError:
        pass
    try:
        resolved.relative_to(CONSUMED_DIR.resolve())
        return True
    except ValueError:
        return False


def is_allowlisted_path(relative: str, file_size: int) -> bool:
    normalized = normalize_relative_path(relative)
    if file_size <= READ_FILE_SMALL_FILE_BYTES:
        return True
    if any(normalized.startswith(prefix) for prefix in READ_FILE_ALLOWED_PREFIXES):
        return True
    suffix = Path(normalized).suffix.lower()
    return suffix in READ_FILE_ALLOWED_SUFFIXES


def count_text_lines(file_path: Path) -> int:
    with file_path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def validate_read_file(path: str) -> str | None:
    """Return an error message when read_file must be refused, else None."""
    file_path = safe_path(path)
    ensure_file_exists(file_path, path)

    if is_handoff_path(file_path):
        return (
            f"Refusing read_file on handoff storage ({path}). "
            "Use read_handoff with handoff_id or handoff_path instead."
        )

    relative = normalize_relative_path(str(file_path.relative_to(safe_path("."))))
    file_size = file_path.stat().st_size

    if is_allowlisted_path(relative, file_size):
        return None

    if file_path.suffix.lower() == ".py":
        if file_size > READ_FILE_MAX_PY_BYTES:
            return (
                f"Refusing read_file on large Python module ({path}, "
                f"{file_size} bytes). Use extract_symbol(name=..., path=...) "
                "or find_symbol to inspect one definition instead."
            )
        line_count = count_text_lines(file_path)
        if line_count > READ_FILE_MAX_PY_LINES:
            return (
                f"Refusing read_file on large Python module ({path}, "
                f"{line_count} lines). Use extract_symbol(name=..., path=...) "
                "or find_symbol to inspect one definition instead."
            )

    return None
