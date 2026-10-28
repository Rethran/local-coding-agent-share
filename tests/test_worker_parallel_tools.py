from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from orchestrator.session import SessionStore, set_session
from orchestrator.tool_compact import ToolExecutionResult, append_compact_tool_result, execute_tool_batch
from orchestrator.worker_context import begin_worker_attempt, clear_worker_context


def _tool_call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "function": {
            "name": name,
            "arguments": arguments,
        }
    }


def test_execute_tool_batch_preserves_order_for_parallel_readonly() -> None:
    session = SessionStore(run_id="parallel-order")
    set_session(session)
    begin_worker_attempt("HEAT-009")
    delays = {"a.py": 0.05, "b.py": 0.01, "c.py": 0.03}

    def slow_read_file(path: str) -> str:
        time.sleep(delays[path])
        return f"content:{path}"

    with patch.dict(
        "orchestrator.tool_compact.TOOL_FUNCTIONS",
        {"read_file": slow_read_file},
    ):
        started = time.perf_counter()
        results = execute_tool_batch(
            [
                _tool_call("read_file", {"path": "a.py"}),
                _tool_call("read_file", {"path": "b.py"}),
                _tool_call("read_file", {"path": "c.py"}),
            ]
        )
        elapsed = time.perf_counter() - started

    assert [result.arguments["path"] for result in results] == ["a.py", "b.py", "c.py"]
    assert elapsed < 0.12
    clear_worker_context()
    set_session(None)


def test_execute_tool_batch_runs_mixed_calls_sequentially() -> None:
    session = SessionStore(run_id="parallel-mixed")
    set_session(session)
    begin_worker_attempt("HEAT-009")
    calls: list[str] = []

    def read_file(path: str) -> str:
        calls.append(f"read:{path}")
        return path

    def write_file(path: str, content: str) -> str:
        calls.append(f"write:{path}")
        return "ok"

    with patch.dict(
        "orchestrator.tool_compact.TOOL_FUNCTIONS",
        {"read_file": read_file, "write_file": write_file},
    ):
        results = execute_tool_batch(
            [
                _tool_call("read_file", {"path": "app/a.py"}),
                _tool_call("write_file", {"path": "app/a.py", "content": "x"}),
            ]
        )

    assert calls == ["read:app/a.py", "write:app/a.py"]
    assert len(results) == 2
    assert results[0].name == "read_file"
    assert results[1].name == "write_file"
    clear_worker_context()
    set_session(None)


def test_execute_tool_batch_falls_back_to_sequential_on_preflight_refusal() -> None:
    session = SessionStore(run_id="parallel-refusal")
    set_session(session)
    begin_worker_attempt("HEAT-009")
    session.record_after_execute(
        "HEAT-009",
        "read_file",
        {"path": "app/a.py"},
        "ok",
        "HEAT-009-0001",
        "old",
    )

    results = execute_tool_batch(
        [
            _tool_call("read_file", {"path": "app/a.py"}),
            _tool_call("read_file", {"path": "app/b.py"}),
        ]
    )

    assert len(results) == 2
    assert results[0].status == "error"
    assert "Duplicate read_file" in results[0].raw_result
    assert results[1].status == "ok"
    clear_worker_context()
    set_session(None)


def test_parallel_batch_disabled_when_preflight_satisfied() -> None:
    session = SessionStore(run_id="parallel-preflight-off")
    set_session(session)
    begin_worker_attempt("HEAT-009")
    session.set_exploration_policy("HEAT-009", preflight_satisfied=True)
    delays = {"a.py": 0.05, "b.py": 0.05}

    def slow_read_file(path: str) -> str:
        time.sleep(delays[path])
        return f"content:{path}"

    with patch.dict(
        "orchestrator.tool_compact.TOOL_FUNCTIONS",
        {"read_file": slow_read_file},
    ):
        started = time.perf_counter()
        results = execute_tool_batch(
            [
                _tool_call("read_file", {"path": "a.py"}),
                _tool_call("read_file", {"path": "b.py"}),
            ]
        )
        elapsed = time.perf_counter() - started

    assert len(results) == 2
    assert results[0].status == "ok"
    assert results[1].status == "ok"
    assert elapsed >= 0.09
    clear_worker_context()
    set_session(None)


def test_append_compact_tool_result_includes_next_action() -> None:
    session = SessionStore(run_id="compact-next-action")
    set_session(session)
    messages: list[dict[str, Any]] = []
    fake_handoff = {
        "compact": {
            "handoffId": "HEAT-009-0099",
            "handoffPath": "handoffs/HEAT-009/HEAT-009-0099.json",
        }
    }

    with patch("orchestrator.tool_compact._handoffs.record_handoff", return_value=fake_handoff):
        append_compact_tool_result(
            messages=messages,
            task_id="HEAT-009",
            previous_handoff_id="",
            tool_name="read_file",
            arguments={"path": "app/example.py"},
            status="ok",
            raw_result="file content",
            merged_subtask={"targetArtifacts": {"module": "app/example.py"}},
        )

    assert len(messages) == 1
    payload = json.loads(messages[0]["content"])
    assert payload.get("nextAction") == {
        "tool": "replace_in_file",
        "hint": "implement using context from app/example.py",
    }
    set_session(None)
