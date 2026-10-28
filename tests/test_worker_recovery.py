"""Focused tests for worker recovery from unproductive model responses."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.config import SYNTAX_WRITE_RETRY_MAX
from orchestrator.tool_compact import ToolExecutionResult
from orchestrator.worker import _run_worker_loop, response_to_message

_SYNTAX_DIAGNOSTIC = (
    "ERROR executing write_file: ERROR: Python syntax invalid in "
    "tests/fixtures/count_items.py\n"
    '  File "tests/fixtures/count_items.py", line 1\n'
    "    def count_items(items):\n"
    "    ^\n"
    "IndentationError: unexpected indent"
)


def _response(message: dict, *, eval_count: int = 10) -> SimpleNamespace:
    return SimpleNamespace(
        message=message,
        prompt_eval_count=100,
        eval_count=eval_count,
    )


def _run(**overrides):
    arguments = {
        "task_id": "HEAT-009",
        "subtask_id": "HEAT-009.2",
        "model": "heat-agent:latest",
        "max_tool_steps": 4,
        "show_model_output": False,
        "attempt_number": 1,
        "recovery_context": None,
        "num_ctx": 24576,
        "num_predict": 2048,
        "context_threshold": 0.85,
        "max_context_handovers": 3,
        "attempt_ctx": SimpleNamespace(text_only_retries=0),
        "session": None,
        "context_pack": None,
        "merged_subtask": {"targetArtifacts": {"module": "app/api/schedule.py"}},
    }
    arguments.update(overrides)
    return _run_worker_loop(**arguments)


def test_response_to_message_strips_thinking_without_mutating_response() -> None:
    original_message = {
        "role": "assistant",
        "content": "",
        "thinking": "large hidden reasoning",
        "tool_calls": [{"function": {"name": "run_command", "arguments": {}}}],
    }

    normalized = response_to_message(SimpleNamespace(message=original_message))

    assert "thinking" not in normalized
    assert normalized["tool_calls"] == original_message["tool_calls"]
    assert original_message["thinking"] == "large hidden reasoning"


def test_worker_recovers_xml_tool_call_from_assistant_content() -> None:
    response = _response({
        "role": "assistant",
        "content": (
            "<tool_call><function=run_command>"
            "<parameter=command>python -m pytest -q</parameter>"
            "</function></tool_call>"
        ),
    })
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": "python -m pytest -q"},
        status="ok",
        raw_result="Exit code: 0",
    )

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", return_value=response),
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]) as execute,
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", return_value=True),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run()

    assert worker_result.ok is True
    recovered_call = execute.call_args.args[0][0]
    assert recovered_call["function"]["name"] == "run_command"
    assert recovered_call["function"]["arguments"]["command"] == "python -m pytest -q"


def test_worker_omits_long_text_only_turn_before_single_retry() -> None:
    responses = [
        _response({"role": "assistant", "content": "x" * 10_000}, eval_count=2500),
        _response({"role": "assistant", "content": "still no tool"}, eval_count=20),
    ]

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", side_effect=responses) as chat,
        patch("orchestrator.worker.persist_and_sync", return_value={}),
        patch("orchestrator.worker.is_subtask_done", return_value=False),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
        patch("orchestrator.worker._tasks.update_task_state", return_value={"status": "FAILED"}),
    ):
        worker_result = _run()

    assert worker_result.ok is False
    assert chat.call_count == 2
    retry_messages = chat.call_args_list[1].kwargs["messages"]
    serialized = str(retry_messages)
    assert "x" * 100 not in serialized
    assert "omitted text-only response: 10000 chars" in serialized
    assert "Call replace_in_file" in serialized
    assert [call.kwargs["think"] for call in chat.call_args_list] == ["medium", "medium"]


def test_worker_resets_text_only_retry_after_successful_tool_call() -> None:
    tool_response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "run_command",
                "arguments": {"command": "python -m pytest -q"},
            },
        }],
    })
    responses = [
        tool_response,
        _response({"role": "assistant", "content": ""}, eval_count=8192),
        tool_response,
    ]
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": "python -m pytest -q"},
        status="ok",
        raw_result="Exit code: 0",
    )
    attempt_ctx = SimpleNamespace(text_only_retries=1)

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", side_effect=responses) as chat,
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", side_effect=[False, True]),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(
            max_tool_steps=3,
            attempt_ctx=attempt_ctx,
        )

    assert worker_result.ok is True
    assert chat.call_count == 3
    assert attempt_ctx.text_only_retries == 0
    assert [call.kwargs["think"] for call in chat.call_args_list] == [
        "medium",
        "medium",
        "medium",
    ]


def test_worker_routes_effort_from_read_through_verification() -> None:
    def tool_response(name: str, arguments: dict) -> SimpleNamespace:
        return _response({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }],
        })

    responses = [
        tool_response("read_file", {"path": "app/api/schedule.py"}),
        tool_response(
            "replace_in_file",
            {
                "path": "app/api/schedule.py",
                "old_text": "before",
                "new_text": "after",
            },
        ),
        tool_response("run_command", {"command": "python -m pytest -q"}),
        tool_response(
            "update_subtask_state",
            {
                "task_id": "HEAT-009",
                "subtask_id": "HEAT-009.2",
                "status": "DONE",
            },
        ),
    ]
    results = [
        [ToolExecutionResult("read_file", {}, "ok", "source")],
        [ToolExecutionResult("replace_in_file", {}, "ok", "Replaced")],
        [ToolExecutionResult("run_command", {}, "ok", "Exit code: 0\n4 passed")],
        [ToolExecutionResult("update_subtask_state", {}, "ok", "{}")],
    ]

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", side_effect=responses) as chat,
        patch("orchestrator.worker.tool_compact.execute_tool_batch", side_effect=results),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", side_effect=[False, False, False, True]),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(max_tool_steps=4)

    assert worker_result.ok is True
    assert [call.kwargs["think"] for call in chat.call_args_list] == [
        "medium",
        "medium",
        "medium",
        "medium",
    ]


def test_worker_yields_after_exact_declared_verification_passes() -> None:
    command = "pytest tests/ -k toggle_api -v"
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "run_command",
                "arguments": {"command": command},
            },
        }],
    })
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": command},
        status="ok",
        raw_result="Exit code: 0\n2 passed",
    )

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", return_value=response) as chat,
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", return_value=False),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(
            merged_subtask={
                "targetArtifacts": {"module": "app/api/schedule.py"},
                "verification": [command],
            },
        )

    assert worker_result.ok is True
    assert worker_result.ready_for_verification is True
    assert chat.call_count == 1


def test_test_writer_yields_after_declared_verification_fails() -> None:
    command = "pytest tests/ -k toggle_api -v"
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "run_command",
                "arguments": {"command": command},
            },
        }],
    })
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": command},
        status="ok",
        raw_result="Exit code: 1\n1 failed",
    )

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", return_value=response) as chat,
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", return_value=False),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(
            role="test_writer",
            merged_subtask={
                "targetArtifacts": {
                    "module": "app/api/schedule.py",
                    "test": "tests/test_schedule_toggle.py",
                },
                "verification": [command],
            },
        )

    assert worker_result.ok is True
    assert worker_result.ready_for_verification is False
    assert chat.call_count == 1


def test_test_writer_yields_to_verifier_when_declared_verification_already_passes() -> None:
    command = "pytest tests/ -k toggle_api -v"
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "run_command",
                "arguments": {"command": command},
            },
        }],
    })
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": command},
        status="ok",
        raw_result="Exit code: 0\n2 passed",
    )

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", return_value=response),
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.persist_and_sync"),
        patch("orchestrator.worker.is_subtask_done", return_value=False),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(
            role="test_writer",
            merged_subtask={"verification": [command]},
        )

    assert worker_result.ok is True
    assert worker_result.ready_for_verification is True


def test_worker_compacts_context_once_after_malformed_tool_call() -> None:
    malformed = RuntimeError(
        'llama-server returned invalid tool call arguments for "write_file": '
        "unexpected end of JSON input"
    )
    response = _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "run_command",
                "arguments": {"command": "python -m pytest -q"},
            },
        }],
    })
    result = ToolExecutionResult(
        name="run_command",
        arguments={"command": "python -m pytest -q"},
        status="ok",
        raw_result="Exit code: 0",
    )

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", side_effect=[malformed, response]) as chat,
        patch(
            "orchestrator.worker.perform_context_handover",
            return_value=(
                [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "compact continuation"},
                ],
                "handoff",
            ),
        ) as handover,
        patch("orchestrator.worker.tool_compact.execute_tool_batch", return_value=[result]),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", return_value=True),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run()

    assert worker_result.ok is True
    assert chat.call_count == 2
    handover.assert_called_once()
    recovery_messages = chat.call_args_list[1].kwargs["messages"]
    assert any(
        message["content"].startswith("The previous tool call was truncated")
        for message in recovery_messages
    )
    assert chat.call_args_list[0].kwargs["num_predict"] == 2048
    assert [call.kwargs["think"] for call in chat.call_args_list] == ["medium", "medium"]


def _write_file_response() -> SimpleNamespace:
    return _response({
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "function": {
                "name": "write_file",
                "arguments": {
                    "path": "tests/fixtures/count_items.py",
                    "content": "    def count_items(items):\n        pass\n",
                },
            },
        }],
    })


def test_worker_forces_retry_after_python_syntax_write_error() -> None:
    syntax_result = ToolExecutionResult(
        name="write_file",
        arguments={"path": "tests/fixtures/count_items.py"},
        status="error",
        raw_result=_SYNTAX_DIAGNOSTIC,
    )
    ok_result = ToolExecutionResult(
        name="write_file",
        arguments={"path": "tests/fixtures/count_items.py"},
        status="ok",
        raw_result="Wrote file: tests/fixtures/count_items.py",
    )
    attempt_ctx = SimpleNamespace(text_only_retries=0, syntax_write_retries=0)

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch(
            "orchestrator.worker.ollama_chat",
            side_effect=[_write_file_response(), _write_file_response()],
        ) as chat,
        patch(
            "orchestrator.worker.tool_compact.execute_tool_batch",
            side_effect=[[syntax_result], [ok_result]],
        ),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", side_effect=[False, True]),
        patch("orchestrator.worker._tasks.load_task_state", return_value={"status": "IN_PROGRESS"}),
    ):
        worker_result = _run(max_tool_steps=3, attempt_ctx=attempt_ctx)

    assert worker_result.ok is True
    assert chat.call_count == 2
    retry_messages = chat.call_args_list[1].kwargs["messages"]
    retry_text = "\n".join(str(message.get("content") or "") for message in retry_messages)
    assert "You must call write_file on tests/fixtures/count_items.py" in retry_text
    assert "IndentationError: unexpected indent" in retry_text
    assert "do not switch tools" in retry_text.lower() or "Do not reply with a plan" in retry_text
    assert attempt_ctx.syntax_write_retries == 0


def test_worker_fails_after_python_syntax_write_retry_cap() -> None:
    syntax_result = ToolExecutionResult(
        name="write_file",
        arguments={"path": "tests/fixtures/count_items.py"},
        status="error",
        raw_result=_SYNTAX_DIAGNOSTIC,
    )
    attempt_ctx = SimpleNamespace(text_only_retries=0, syntax_write_retries=0)
    failed_state = {"status": "FAILED", "blockers": [_SYNTAX_DIAGNOSTIC]}
    responses = [_write_file_response() for _ in range(SYNTAX_WRITE_RETRY_MAX + 1)]

    with (
        patch("orchestrator.worker.task_user_prompt_for_subtask", return_value="payload"),
        patch("orchestrator.worker.ollama_chat", side_effect=responses) as chat,
        patch(
            "orchestrator.worker.tool_compact.execute_tool_batch",
            return_value=[syntax_result],
        ),
        patch("orchestrator.worker.tool_compact.append_compact_tool_result", return_value="handoff"),
        patch("orchestrator.worker.is_subtask_done", return_value=False),
        patch("orchestrator.worker.persist_and_sync", return_value={}) as persist,
        patch(
            "orchestrator.worker._tasks.load_task_state",
            return_value={"status": "IN_PROGRESS"},
        ),
        patch("orchestrator.worker._tasks.update_task_state", return_value=failed_state) as update,
    ):
        worker_result = _run(
            max_tool_steps=SYNTAX_WRITE_RETRY_MAX + 2,
            attempt_ctx=attempt_ctx,
        )

    assert worker_result.ok is False
    assert chat.call_count == SYNTAX_WRITE_RETRY_MAX + 1
    persist.assert_called_once()
    assert persist.call_args.kwargs["stop_reason"] == "python_syntax_write"
    assert persist.call_args.kwargs["blockers"] == [_SYNTAX_DIAGNOSTIC]
    update.assert_called_once()
    assert update.call_args.kwargs["blockers"] == [_SYNTAX_DIAGNOSTIC]
    assert update.call_args.kwargs["last_failure_reason"] == _SYNTAX_DIAGNOSTIC
    assert attempt_ctx.syntax_write_retries == SYNTAX_WRITE_RETRY_MAX + 1
