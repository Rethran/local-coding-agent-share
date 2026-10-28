"""Case allowlists for the worker tool schema. No Ollama calls."""

from __future__ import annotations

from orchestrator.prompts import system_prompt
from orchestrator.tool_cases import (
    EDIT_TOOL_NAMES,
    EXPLORE_TOOL_NAMES,
    TEST_WRITE_TOOL_NAMES,
    WorkerSessionFlags,
    select_worker_tools,
    session_flags_from,
)
from orchestrator.tools_registry import TOOL_FUNCTIONS, TOOLS

_EXCLUDED = {
    "record_handoff",
    "delete_handoff_chain",
    "update_task_state",
    "get_verification_checklist",
    "get_active_subtask",
    "create_folder",
}

_READY = {"explorationPolicy": {"preflightSatisfied": True}}
_NEEDS_EXPLORE = {"explorationPolicy": {"preflightSatisfied": False}}


def _names(tools: list[dict]) -> list[str]:
    return [tool["function"]["name"] for tool in tools]


def test_edit_when_preflight_satisfied() -> None:
    case_name, tools = select_worker_tools("implementer", _READY)
    names = _names(tools)
    assert case_name == "edit"
    assert names == list(EDIT_TOOL_NAMES)
    for absent in ("get_subtask_context", "extract_symbol", "search_files", "read_handoff"):
        assert absent not in names


def test_explore_when_preflight_not_satisfied() -> None:
    case_name, tools = select_worker_tools("implementer", _NEEDS_EXPLORE)
    names = _names(tools)
    assert case_name == "explore"
    assert names == list(EXPLORE_TOOL_NAMES)
    for present in ("search_files", "find_symbol", "extract_symbol", "get_subtask_context"):
        assert present in names
    assert _EXCLUDED.isdisjoint(names)


def test_budget_or_successful_write_switches_explore_to_edit() -> None:
    exhausted, _ = select_worker_tools(
        "implementer",
        _NEEDS_EXPLORE,
        WorkerSessionFlags(exploration_budget_exhausted=True),
    )
    written, _ = select_worker_tools(
        "implementer",
        _NEEDS_EXPLORE,
        WorkerSessionFlags(has_successful_write=True),
    )
    assert exhausted == "edit"
    assert written == "edit"


def test_truncated_handoff_adds_read_handoff_until_expanded() -> None:
    pending, pending_tools = select_worker_tools(
        "implementer",
        _READY,
        WorkerSessionFlags(pending_truncated_handoff=True),
    )
    expanded, expanded_tools = select_worker_tools(
        "implementer",
        _READY,
        WorkerSessionFlags(pending_truncated_handoff=False),
    )
    assert pending == "edit"
    assert "read_handoff" in _names(pending_tools)
    assert expanded == "edit"
    assert "read_handoff" not in _names(expanded_tools)


def test_test_writer_allowlist_excludes_subtask_context() -> None:
    case_name, tools = select_worker_tools("test_writer", _NEEDS_EXPLORE)
    names = _names(tools)
    assert case_name == "test_write"
    assert names == list(TEST_WRITE_TOOL_NAMES)
    assert "get_subtask_context" not in names
    assert "list_directory" in names


def test_selected_schemas_are_a_registry_subset() -> None:
    packs = (_READY, _NEEDS_EXPLORE, None)
    flags = (
        WorkerSessionFlags(),
        WorkerSessionFlags(pending_truncated_handoff=True),
        WorkerSessionFlags(exploration_budget_exhausted=True),
        WorkerSessionFlags(has_successful_write=True),
    )
    for role in ("implementer", "test_writer"):
        for pack in packs:
            for flag in flags:
                _case_name, tools = select_worker_tools(role, pack, flag)
                assert tools
                for tool in tools:
                    name = tool["function"]["name"]
                    assert name in TOOL_FUNCTIONS
                    assert tool in TOOLS
                    assert name not in _EXCLUDED


def test_session_flags_from_task_state() -> None:
    class _Task:
        read_only_tools_used = 2
        max_read_only_before_write = 2
        has_successful_write = False
        truncated_handoff_ids = {"h1", "h2"}
        expanded_handoff_ids = {"h2"}

    class _Session:
        def _task(self, task_id: str) -> _Task:
            assert task_id == "CTX-013"
            return _Task()

    flags = session_flags_from(_Session(), "CTX-013")
    assert flags.exploration_budget_exhausted is True
    assert flags.has_successful_write is False
    assert flags.pending_truncated_handoff is True
    assert session_flags_from(None, "CTX-013") == WorkerSessionFlags()


def test_cased_system_prompt_mentions_only_allowlisted_tools() -> None:
    edit_prompt = system_prompt("CTX-013.1", role="implementer", tool_case="edit")
    for absent in (
        "get_active_subtask",
        "search_files",
        "extract_symbol",
        "get_subtask_context",
        "update_task_state",
        "read_handoff",
    ):
        assert absent not in edit_prompt
    assert "update_subtask_state" in edit_prompt
    assert "replace_in_file" in edit_prompt

    handoff_prompt = system_prompt(
        "CTX-013.1",
        role="implementer",
        tool_case="edit",
        include_read_handoff=True,
    )
    assert "read_handoff" in handoff_prompt

    explore_prompt = system_prompt("CTX-013.1", role="implementer", tool_case="explore")
    for present in ("search_files", "extract_symbol", "get_subtask_context", "list_directory"):
        assert present in explore_prompt
    assert "get_active_subtask" not in explore_prompt
    assert "update_task_state" not in explore_prompt

    test_prompt = system_prompt("CTX-013.1", role="test_writer", tool_case="test_write")
    assert "list_directory" in test_prompt
    assert "get_subtask_context" not in test_prompt

    unchanged = system_prompt("GRAPH-007.1")
    assert "get_active_subtask" in unchanged
    assert "update_task_state" in unchanged
