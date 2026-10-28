"""Unit tests for orchestrator search guardrails."""

from __future__ import annotations

import json
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.search_guard import (  # noqa: E402
    build_allowed_search_terms,
    cap_search_results,
    query_matches_allowed_term,
    validate_search_files,
)
from orchestrator.worker_context import begin_worker_attempt, clear_worker_context


def _heat_003_2_subtask() -> dict:
    task = json.loads((AGENT_ROOT / "requirements" / "HEAT-003.json").read_text(encoding="utf-8"))
    return next(item for item in task["subtasks"] if item["id"] == "HEAT-003.2")


def test_build_allowed_search_terms_includes_legacy_symbols() -> None:
    task = json.loads((AGENT_ROOT / "requirements" / "HEAT-003.json").read_text(encoding="utf-8"))
    subtask = _heat_003_2_subtask()
    terms = build_allowed_search_terms(subtask, task)
    assert "haehinnat" in terms
    assert "parse_entsoe_prices" in terms
    assert "old_entsoes/entsoe_data.xml" in terms


def test_query_matches_allowed_term_is_case_insensitive() -> None:
    assert query_matches_allowed_term("PT15M", ["pt15m"]) is True
    assert query_matches_allowed_term("price.amount", ["price.amount"]) is True
    assert query_matches_allowed_term("position", ["haehinnat"]) is False


def test_validate_search_blocks_def_parse_on_legacy_py() -> None:
    begin_worker_attempt("HEAT-003")
    try:
        error = validate_search_files("def parse", "old_entsoes/entsoe_prod.py")
        assert error is not None
        assert "def parse" in error or "forbidden" in error.lower() or "Refusing" in error
        assert "haehinnat" in error
    finally:
        clear_worker_context()


def test_validate_search_blocks_legacy_py_when_legacy_symbols_defined() -> None:
    begin_worker_attempt("HEAT-003")
    try:
        error = validate_search_files("haehinnat", "old_entsoes/entsoe_prod.py")
        assert error is not None
        assert "legacySymbols" in error or "blocked" in error
    finally:
        clear_worker_context()


def test_validate_search_allows_app_implementation_queries() -> None:
    begin_worker_attempt("HEAT-003")
    try:
        assert validate_search_files("parse_entsoe_prices", "app/domain/prices.py") is None
    finally:
        clear_worker_context()


def test_validate_search_blocks_duplicate_query_in_attempt() -> None:
    begin_worker_attempt("HEAT-003")
    try:
        first = validate_search_files("PriceSlot", "app/domain/prices.py")
        assert first is None
        second = validate_search_files("PriceSlot", "app/domain/prices.py")
        assert second is not None
        assert "Duplicate" in second
    finally:
        clear_worker_context()


def test_cap_search_results_returns_error_when_too_many_matches() -> None:
    matches = [{"file": "x.py", "line": index, "text": "position"} for index in range(50)]
    result = cap_search_results(json.dumps(matches), "old_entsoes/entsoe_prod.py")
    assert result.startswith("ERROR:")
    assert "40" in result or "limit" in result


def test_validate_search_blocks_fixture_xml_discovery() -> None:
    begin_worker_attempt("HEAT-003")
    try:
        error = validate_search_files("TimeSeries", "old_entsoes/entsoe_data.xml")
        assert error is not None
        assert "Refusing" in error
        assert "dataExamples" in error
    finally:
        clear_worker_context()


def test_get_subtask_context_includes_legacy_slices() -> None:
    from orchestrator.subtask_context import build_subtask_context
    from tools import _tasks

    task_id = "HEAT-003"
    state_path = AGENT_ROOT / "requirements" / f"{task_id}.state.json"
    backup = state_path.read_text(encoding="utf-8") if state_path.exists() else None

    try:
        task = json.loads((AGENT_ROOT / "requirements" / f"{task_id}.json").read_text(encoding="utf-8"))
        subtasks = [
            {
                "id": item["id"],
                "title": item["title"],
                "status": "DONE" if item["id"] == "HEAT-003.1" else (
                    "IN_PROGRESS" if item["id"] == "HEAT-003.2" else "PENDING"
                ),
                "summary": item.get("summary", ""),
            }
            for item in task["subtasks"]
        ]
        _tasks.update_task_state(
            task_id,
            "IN_PROGRESS",
            summary="Pin active subtask to HEAT-003.2 for context test.",
            subtasks=subtasks,
            active_subtask_id="HEAT-003.2",
        )

        context = build_subtask_context(task_id)
        assert context["subtaskId"] == "HEAT-003.2"
        assert context["targetArtifacts"]["function"] == "parse_entsoe_prices"
        assert len(context["legacySlices"]) >= 1
        assert "price.amount" in context["legacySlices"][0]["source"] or "Point" in context["legacySlices"][0]["source"]
        assert len(context["dataExamples"]) >= 1
        assert "107.15" in context["dataExamples"][0] or any("107.15" in ex for ex in context["dataExamples"])
    finally:
        if backup is None:
            if state_path.exists():
                state_path.unlink()
        else:
            state_path.write_text(backup, encoding="utf-8")


def test_get_subtask_context_json_is_valid() -> None:
    import json

    from orchestrator.subtask_context import get_subtask_context_json

    payload = json.loads(get_subtask_context_json("HEAT-003"))
    assert payload["taskId"] == "HEAT-003"
    assert payload["dataExamples"]
