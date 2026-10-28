"""Unit tests for orchestrator read_file guardrails."""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.read_guard import (  # noqa: E402
    is_allowlisted_path,
    is_handoff_path,
    validate_read_file,
)
from orchestrator.tool_adapters import safe_read_file  # noqa: E402


def test_handoff_path_is_detected() -> None:
    handoffs_root = AGENT_ROOT / "handoffs" / "HEAT-003"
    assert is_handoff_path(handoffs_root / "HEAT-003-0001.json") is True


def test_allowlisted_prefix_permits_large_python() -> None:
    assert is_allowlisted_path("app/domain/prices.py", file_size=100_000) is True


def test_allowlisted_suffix_permits_xml() -> None:
    assert is_allowlisted_path("old_entsoes/entsoe_data.xml", file_size=100_000) is True


def test_small_file_is_allowlisted_by_size() -> None:
    assert is_allowlisted_path("old_entsoes/small_helper.py", file_size=1024) is True


def test_validate_read_file_blocks_handoff_json() -> None:
    handoff_path = "local-coding-agent/handoffs/HEAT-003/HEAT-003-0001.json"
    handoff_file = AGENT_ROOT / "handoffs" / "HEAT-003" / "HEAT-003-0001.json"
    if not handoff_file.exists():
        handoff_file.parent.mkdir(parents=True, exist_ok=True)
        handoff_file.write_text("{}", encoding="utf-8")
    try:
        error = validate_read_file(handoff_path)
        assert error is not None
        assert "read_handoff" in error
    finally:
        if handoff_file.exists() and handoff_file.read_text(encoding="utf-8") == "{}":
            handoff_file.unlink()
            if handoff_file.parent.exists() and not any(handoff_file.parent.iterdir()):
                handoff_file.parent.rmdir()


def test_validate_read_file_blocks_large_legacy_python() -> None:
    error = validate_read_file("old_entsoes/entsoe_prod.py")
    assert error is not None
    assert "extract_symbol" in error


def test_validate_read_file_allows_xml_fixture() -> None:
    assert validate_read_file("old_entsoes/entsoe_data.xml") is None


def test_validate_read_file_allows_requirements_json() -> None:
    assert validate_read_file("local-coding-agent/requirements/living-graph/GRAPH-001.json") is None


def test_validate_read_file_allows_fixture_python() -> None:
    assert validate_read_file("local-coding-agent/tests/fixtures/sample_symbols.py") is None


def test_safe_read_file_returns_error_for_handoff_path() -> None:
    handoff_path = "local-coding-agent/handoffs/HEAT-003/HEAT-003-0099.json"
    handoff_file = AGENT_ROOT / "handoffs" / "HEAT-003" / "HEAT-003-0099.json"
    handoff_file.parent.mkdir(parents=True, exist_ok=True)
    handoff_file.write_text("{}", encoding="utf-8")
    try:
        result = safe_read_file(handoff_path)
        assert result.startswith("ERROR:")
        assert "read_handoff" in result
    finally:
        handoff_file.unlink()
        if handoff_file.parent.exists() and not any(handoff_file.parent.iterdir()):
            handoff_file.parent.rmdir()


def test_safe_read_file_returns_error_for_large_legacy_python() -> None:
    result = safe_read_file("old_entsoes/entsoe_prod.py")
    assert result.startswith("ERROR:")
    assert "extract_symbol" in result
