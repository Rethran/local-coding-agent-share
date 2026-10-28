"""Unit tests for symbol extraction and file-scoped search."""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from tools.extract_symbol import extract_symbol  # noqa: E402
from tools.find_symbol import find_symbol  # noqa: E402
from tools.search_files import search_files  # noqa: E402
from tools._symbols import DEFAULT_SOURCE_CHAR_CAP, scan_for_extract_matches  # noqa: E402

FIXTURE_DIR = "local-coding-agent/tests/fixtures"
FIXTURE_FILE = f"{FIXTURE_DIR}/sample_symbols.py"


def test_extract_symbol_returns_function_body() -> None:
    matches = json.loads(extract_symbol("plain_helper", path=FIXTURE_FILE))
    assert len(matches) == 1
    match = matches[0]
    assert match["name"] == "plain_helper"
    assert match["type"] == "FunctionDef"
    assert match["file"].replace("\\", "/").endswith("sample_symbols.py")
    assert "def plain_helper(value: int) -> int:" in match["source"]
    assert "return value + 1" in match["source"]
    assert match["truncated"] is False
    assert match["start_line"] <= match["end_line"]


def test_extract_symbol_includes_decorators() -> None:
    matches = json.loads(extract_symbol("decorated_helper", path=FIXTURE_FILE))
    assert len(matches) == 1
    source = matches[0]["source"]
    assert "@_noop_decorator" in source
    assert "def decorated_helper" in source
    assert matches[0]["start_line"] < matches[0]["end_line"]


def test_extract_symbol_missing_returns_empty() -> None:
    matches = json.loads(
        extract_symbol("definitely_missing_symbol_xyz", path=FIXTURE_DIR)
    )
    assert matches == []


def test_extract_symbol_path_limits_scan() -> None:
    matches = json.loads(extract_symbol("plain_helper", path=FIXTURE_DIR))
    assert len(matches) >= 1
    assert all(
        m["file"].replace("\\", "/").endswith("sample_symbols.py") for m in matches
    )


def test_find_symbol_includes_end_line() -> None:
    matches = json.loads(find_symbol("plain_helper"))
    fixture_matches = [
        m
        for m in matches
        if m["file"].replace("\\", "/").endswith("sample_symbols.py")
    ]
    assert fixture_matches
    match = fixture_matches[0]
    assert "end_line" in match
    assert match["end_line"] >= match["line"]


def test_search_files_accepts_file_path() -> None:
    matches = json.loads(search_files("plain_helper", path=FIXTURE_FILE))
    assert len(matches) >= 1
    assert matches[0]["file"].replace("\\", "/").endswith("sample_symbols.py")
    assert "plain_helper" in matches[0]["text"]


def test_extract_symbol_line_slice_returns_requested_range() -> None:
    matches = json.loads(
        extract_symbol("plain_helper", path=FIXTURE_FILE, start_line=3, end_line=5)
    )
    assert len(matches) == 1
    assert matches[0]["type"] == "LineSlice"
    assert matches[0]["start_line"] == 3
    assert matches[0]["end_line"] == 5


def test_extract_symbol_truncates_oversized_body() -> None:
    oversized = AGENT_ROOT / "tests" / "fixtures" / "_oversized_tmp.py"
    try:
        body_lines = ["def oversized_fn():"] + ["    x = 1"] * 2000
        oversized.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
        relative = "local-coding-agent/tests/fixtures/_oversized_tmp.py"
        matches = scan_for_extract_matches(
            "oversized_fn",
            relative,
            char_cap=500,
        )
        assert len(matches) == 1
        assert matches[0]["truncated"] is True
        assert len(matches[0]["source"]) < DEFAULT_SOURCE_CHAR_CAP
        assert "truncated" in matches[0]["source"]
    finally:
        if oversized.exists():
            oversized.unlink()
