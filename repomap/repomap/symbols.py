"""
Phase 1 — symbol extraction.

Walks Python source with Tree-sitter and produces a flat, deterministic
table of function/method symbols: name, signature (no body), and
location. Every later phase joins onto this table by `symbol_id`, so
its shape and determinism guarantees matter more than raw speed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_python

_PARSER = Parser(Language(tree_sitter_python.language()))

# Directories we never want to walk into when scanning a repo.
_SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".pytest_cache"}


@dataclass(frozen=True)
class Symbol:
    """A single function or method definition."""

    name: str
    """Qualified name, e.g. 'Cache.get' for a method, 'compute_key' for a function."""

    file: Path
    signature: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    docstring: str = ""

    @property
    def symbol_id(self) -> str:
        """Stable identifier used to join this symbol across phases."""
        return f"{self.file}:{self.name}:{self.start_line}"


def _decode_signature(source: bytes, func_node) -> str:
    """Everything up to (not including) the function body's ':' colon-newline block."""
    header_end = func_node.child_by_field_name("body").start_byte
    header_bytes = source[func_node.start_byte:header_end]
    text = header_bytes.decode("utf-8", errors="replace").rstrip()
    if text.endswith(":"):
        text = text[:-1].rstrip()
    return text


def _docstring_of(func_node, source: bytes) -> str:
    body = func_node.child_by_field_name("body")
    if body is None or body.child_count == 0:
        return ""
    first_stmt = body.children[0]
    if first_stmt.type == "expression_statement" and first_stmt.child_count:
        expr = first_stmt.children[0]
        if expr.type == "string":
            raw = source[expr.start_byte:expr.end_byte].decode("utf-8", errors="replace")
            return raw.strip("\"' \n")
    return ""


def _walk_functions(node, source: bytes, class_prefix: str, out: list[Symbol]) -> None:
    for child in node.children:
        if child.type == "class_definition":
            name_node = child.child_by_field_name("name")
            class_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_functions(body, source, class_name, out)
        elif child.type == "function_definition":
            name_node = child.child_by_field_name("name")
            fn_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8")
            qualified = f"{class_prefix}.{fn_name}" if class_prefix else fn_name
            out.append(
                Symbol(
                    name=qualified,
                    file=None,  # filled in by caller
                    signature=_decode_signature(source, child),
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                    docstring=_docstring_of(child, source),
                )
            )
            # nested/local functions: keep walking, unqualified (rare in practice)
            body = child.child_by_field_name("body")
            if body is not None:
                _walk_functions(body, source, class_prefix, out)
        else:
            _walk_functions(child, source, class_prefix, out)


def extract_symbols_from_file(path: Path) -> list[Symbol]:
    """Parse one Python file into a list of Symbols. Never raises on bad input."""
    path = Path(path)
    try:
        source = path.read_bytes()
        tree = _PARSER.parse(source)
    except Exception:
        return []

    if tree.root_node.has_error and tree.root_node.child_count == 0:
        return []

    raw: list[Symbol] = []
    _walk_functions(tree.root_node, source, class_prefix="", out=raw)

    # attach file path (dataclass is frozen, so rebuild)
    return [
        Symbol(
            name=s.name,
            file=path,
            signature=s.signature,
            start_line=s.start_line,
            end_line=s.end_line,
            docstring=s.docstring,
        )
        for s in raw
    ]


def extract_symbols_from_repo(root: Path) -> list[Symbol]:
    """Walk a repo root and extract symbols from every .py file, deterministically ordered."""
    root = Path(root)
    py_files = sorted(
        p for p in root.rglob("*.py")
        if not any(part in _SKIP_DIRS for part in p.parts)
    )

    symbols: list[Symbol] = []
    for f in py_files:
        symbols.extend(extract_symbols_from_file(f))
    return symbols
