"""Tests for balanced thinking-effort transitions."""

from types import SimpleNamespace

from orchestrator.thinking_policy import effort_after_tool_results


def _result(name: str, *, status: str = "ok", raw_result: str = "") -> SimpleNamespace:
    return SimpleNamespace(name=name, status=status, raw_result=raw_result)


def test_read_results_route_to_medium_effort() -> None:
    assert effort_after_tool_results([_result("read_file")]) == "medium"


def test_successful_edit_routes_to_medium_effort() -> None:
    assert effort_after_tool_results([_result("replace_in_file")]) == "medium"


def test_successful_verification_routes_to_medium_effort() -> None:
    result = _result("run_command", raw_result="Exit code: 0\n4 passed")

    assert effort_after_tool_results([result]) == "medium"


def test_failed_tool_or_verification_routes_to_high_effort() -> None:
    tool_error = _result("read_file", status="error", raw_result="refused")
    failed_test = _result("run_command", raw_result="Exit code: 1\n1 failed")

    assert effort_after_tool_results([tool_error]) == "high"
    assert effort_after_tool_results([failed_test]) == "high"


def test_expected_context_pack_refusal_stays_at_medium_effort() -> None:
    refusal = _result(
        "read_file",
        status="error",
        raw_result=(
            "ERROR: contextPack.referenceSnippets already includes "
            "'app/api/schedule.py'."
        ),
    )

    assert effort_after_tool_results([refusal]) == "medium"
