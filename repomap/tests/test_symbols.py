"""
Phase 1: extract a symbol table from Python source using Tree-sitter.

A "symbol" here means a top-level or method-level function definition:
name, signature, file path, and line range. This is the raw material
every later phase (graph, churn, coverage, blast radius) builds on,
so we pin its shape down tightly before anything else is written.
"""
from pathlib import Path

import pytest

from repomap.symbols import Symbol, extract_symbols_from_file, extract_symbols_from_repo

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_extracts_a_single_top_level_function(tmp_path):
    source = tmp_path / "mod.py"
    source.write_text(
        "def compute_key(namespace: str, item_id: int) -> str:\n"
        '    """Build a cache key."""\n'
        "    return f\"{namespace}:{item_id}\"\n"
    )

    symbols = extract_symbols_from_file(source)

    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.name == "compute_key"
    assert sym.file == source
    assert sym.start_line == 1
    assert sym.end_line == 3
    assert "namespace: str" in sym.signature
    assert "item_id: int" in sym.signature


def test_extracts_multiple_functions_in_order(tmp_path):
    source = tmp_path / "mod.py"
    source.write_text(
        "def first():\n    pass\n\n\ndef second():\n    pass\n"
    )

    symbols = extract_symbols_from_file(source)

    assert [s.name for s in symbols] == ["first", "second"]


def test_signature_excludes_function_body(tmp_path):
    source = tmp_path / "mod.py"
    source.write_text(
        "def noisy():\n"
        "    x = 1\n"
        "    y = 2\n"
        "    return x + y\n"
    )

    symbols = extract_symbols_from_file(source)

    assert "x = 1" not in symbols[0].signature
    assert "return" not in symbols[0].signature


def test_extracts_class_methods_with_qualified_names(tmp_path):
    source = tmp_path / "mod.py"
    source.write_text(
        "class Cache:\n"
        "    def get(self, key):\n"
        "        pass\n"
        "\n"
        "    def set(self, key, value):\n"
        "        pass\n"
    )

    symbols = extract_symbols_from_file(source)

    names = [s.name for s in symbols]
    assert "Cache.get" in names
    assert "Cache.set" in names


def test_unparseable_file_returns_empty_list_not_an_exception(tmp_path):
    source = tmp_path / "broken.py"
    source.write_bytes(b"\x00\x01\x02 def(((")

    symbols = extract_symbols_from_file(source)

    assert symbols == []


def test_extract_symbols_from_repo_walks_all_python_files():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)

    names = {s.name for s in symbols}
    assert "compute_key" in names
    assert "invalidate" in names
    assert "invalidate_namespace" in names
    assert "update_item" in names
    assert "bulk_update" in names


def test_extract_symbols_from_repo_skips_non_python_and_vcs_dirs():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)

    for sym in symbols:
        assert ".git" not in sym.file.parts


def test_symbol_id_is_stable_and_unique(tmp_path):
    source = tmp_path / "mod.py"
    source.write_text("def a():\n    pass\n\n\ndef b():\n    pass\n")

    symbols = extract_symbols_from_file(source)

    ids = [s.symbol_id for s in symbols]
    assert len(ids) == len(set(ids)), "symbol ids must be unique within a file"
    # same input -> same ids (determinism requirement)
    again = extract_symbols_from_file(source)
    assert ids == [s.symbol_id for s in again]
