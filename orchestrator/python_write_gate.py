"""Reject writes of Python files that do not parse."""

from __future__ import annotations

import ast

from orchestrator.read_guard import normalize_relative_path

PYTHON_SYNTAX_INVALID_MARKER = "Python syntax invalid"
SYNTAX_WRITE_RETRY_HINT = (
    "retry this write; use the SyntaxError diagnostic; do not switch tools"
)


def is_python_syntax_write_error(message: str) -> bool:
    """True when a tool error is a Python write-gate syntax refusal."""
    return PYTHON_SYNTAX_INVALID_MARKER in message


def _caret_line(source: str, offset: int | None, end_offset: int | None) -> str | None:
    if offset is None or offset <= 0:
        return None
    start = min(max(offset, 1), len(source) + 1)
    width = 1
    if end_offset is not None and end_offset > start:
        width = min(end_offset, len(source) + 1) - start
        width = max(width, 1)
    return "    " + (" " * (start - 1)) + ("^" * width)


def format_python_syntax_error(path: str, exc: SyntaxError) -> str:
    """Format a SyntaxError the way CPython prints it, without dropping fields."""
    relative = normalize_relative_path(path) or path
    lineno = exc.lineno or 1
    lines = [
        f"ERROR: {PYTHON_SYNTAX_INVALID_MARKER} in {relative}",
        f'  File "{relative}", line {lineno}',
    ]
    source = (exc.text or "").rstrip("\n")
    if source:
        lines.append(f"    {source}")
        caret = _caret_line(source, exc.offset, getattr(exc, "end_offset", None))
        if caret:
            lines.append(caret)
    exc_name = type(exc).__name__
    detail = exc.msg or "invalid syntax"
    lines.append(f"{exc_name}: {detail}")
    return "\n".join(lines)


def refuse_invalid_python_write(path: str, content: str) -> str | None:
    """Return an error when path is .py and content is not valid Python."""
    relative = normalize_relative_path(path)
    if not relative.lower().endswith(".py"):
        return None
    try:
        ast.parse(content, filename=relative)
    except SyntaxError as exc:
        return format_python_syntax_error(relative, exc)
    return None
