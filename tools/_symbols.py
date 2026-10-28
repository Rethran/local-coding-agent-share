"""Shared AST helpers for locating and extracting Python symbol definitions."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tools._common import WORKSPACE_ROOT, is_within_ignored_directory, safe_path

SYMBOL_DEFINITION_NODE_TYPES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)

# Soft cap so extracting a huge class cannot refill the worker context.
DEFAULT_SOURCE_CHAR_CAP = 10_000


def iter_python_source_files(path: str = ".") -> list[Path]:
    """Return readable ``*.py`` files under *path* (file or directory)."""
    root = safe_path(path)
    if not root.exists():
        raise ValueError(f"Path does not exist: {path}")

    if root.is_file():
        if root.suffix != ".py" or is_within_ignored_directory(root):
            return []
        return [root]

    if not root.is_dir():
        raise ValueError(f"Not a file or directory: {path}")

    return [
        candidate
        for candidate in root.rglob("*.py")
        if candidate.is_file() and not is_within_ignored_directory(candidate)
    ]


def parse_python_module(source_file: Path) -> tuple[ast.AST, str] | None:
    try:
        source = source_file.read_text(encoding="utf-8")
        return ast.parse(source), source
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


def is_symbol_definition(node: ast.AST) -> bool:
    return isinstance(node, SYMBOL_DEFINITION_NODE_TYPES)


def find_named_definitions(source_file: Path, symbol_name: str) -> list[ast.AST]:
    parsed = parse_python_module(source_file)
    if parsed is None:
        return []

    syntax_tree, _ = parsed
    return [
        node
        for node in ast.walk(syntax_tree)
        if is_symbol_definition(node) and getattr(node, "name", None) == symbol_name
    ]


def symbol_start_line(node: ast.AST) -> int:
    """1-indexed start line including leading decorators when present."""
    decorators = getattr(node, "decorator_list", None) or []
    if decorators:
        return min(decorator.lineno for decorator in decorators)
    return int(node.lineno)


def symbol_end_line(node: ast.AST) -> int:
    end = getattr(node, "end_lineno", None)
    if end is not None:
        return int(end)
    return int(node.lineno)


def slice_source_lines(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    start_index = max(start_line - 1, 0)
    end_index = min(end_line, len(lines))
    return "\n".join(lines[start_index:end_index])


def maybe_truncate_source(
    source: str,
    *,
    start_line: int,
    end_line: int,
    char_cap: int = DEFAULT_SOURCE_CHAR_CAP,
) -> tuple[str, bool]:
    if len(source) <= char_cap:
        return source, False

    truncated = source[:char_cap].rstrip()
    marker = (
        f"\n\n[truncated; full symbol spans lines {start_line}-{end_line}, "
        f"{len(source)} chars total]"
    )
    return truncated + marker, True


def to_location_match(source_file: Path, node: ast.AST) -> dict[str, Any]:
    return {
        "file": str(source_file.relative_to(WORKSPACE_ROOT)),
        "line": node.lineno,
        "end_line": symbol_end_line(node),
        "type": type(node).__name__,
        "name": node.name,
    }


def to_extract_match(
    source_file: Path,
    node: ast.AST,
    source: str,
    *,
    char_cap: int = DEFAULT_SOURCE_CHAR_CAP,
) -> dict[str, Any]:
    start_line = symbol_start_line(node)
    end_line = symbol_end_line(node)
    body = slice_source_lines(source, start_line, end_line)
    body, truncated = maybe_truncate_source(
        body,
        start_line=start_line,
        end_line=end_line,
        char_cap=char_cap,
    )
    return {
        "file": str(source_file.relative_to(WORKSPACE_ROOT)),
        "name": node.name,
        "type": type(node).__name__,
        "start_line": start_line,
        "end_line": end_line,
        "truncated": truncated,
        "source": body,
    }


def scan_for_location_matches(
    symbol_name: str,
    path: str = ".",
) -> list[dict[str, Any]]:
    return [
        to_location_match(source_file, node)
        for source_file in iter_python_source_files(path)
        for node in find_named_definitions(source_file, symbol_name)
    ]


def scan_for_extract_matches(
    symbol_name: str,
    path: str = ".",
    *,
    char_cap: int = DEFAULT_SOURCE_CHAR_CAP,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[dict[str, Any]]:
    if start_line is not None and end_line is not None:
        source_file = safe_path(path)
        if not source_file.is_file():
            raise ValueError(f"Line slice requires a file path, not a directory: {path}")
        parsed = parse_python_module(source_file)
        if parsed is None:
            return []
        _, source = parsed
        body = slice_source_lines(source, start_line, end_line)
        return [{
            "file": str(source_file.relative_to(WORKSPACE_ROOT)),
            "name": symbol_name,
            "type": "LineSlice",
            "start_line": start_line,
            "end_line": end_line,
            "truncated": False,
            "source": body,
        }]

    matches: list[dict[str, Any]] = []
    for source_file in iter_python_source_files(path):
        parsed = parse_python_module(source_file)
        if parsed is None:
            continue
        syntax_tree, source = parsed
        for node in ast.walk(syntax_tree):
            if is_symbol_definition(node) and node.name == symbol_name:
                match = to_extract_match(
                    source_file,
                    node,
                    source,
                    char_cap=char_cap,
                )
                if start_line is not None and end_line is not None:
                    match["source"] = slice_source_lines(source, start_line, end_line)
                    match["start_line"] = start_line
                    match["end_line"] = end_line
                    match["truncated"] = False
                matches.append(match)
    return matches
