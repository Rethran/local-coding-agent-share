"""Unit tests for orchestrator session-scoped continuous delivery."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.attempts import build_retry_context  # noqa: E402
from orchestrator.prompts import task_user_prompt  # noqa: E402
from orchestrator.session import SessionStore, set_session  # noqa: E402
from orchestrator.worker_context import begin_worker_attempt, clear_worker_context  # noqa: E402


def test_diagnostic_command_mentioning_pytest_file_is_not_test_evidence() -> None:
    session = SessionStore(run_id="test-command-classification")

    session.record_after_execute(
        "HEAT-009",
        "run_command",
        {"command": "dir /b conftest.py pytest.ini pyproject.toml"},
        "ok",
        "HEAT-009-0001",
        "Exit code: 0\npytest.ini",
    )

    assert not any(
        lesson.get("kind") == "verified"
        for lesson in session._task("HEAT-009").lessons
    )


def test_record_tool_call_dedupes_identical_fingerprints() -> None:
    session = SessionStore(run_id="test-dedup")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-003",
            "read_file",
            {"path": "app/domain/prices.py"},
            "ok",
            "HEAT-003-0001",
            "content",
        )
        refusal = session.check_before_execute(
            "HEAT-003",
            "read_file",
            {"path": "app/domain/prices.py"},
        )
        assert refusal is not None
        assert "Duplicate read_file" in refusal
        assert "HEAT-003-0001" in refusal
    finally:
        set_session(None)


def test_read_file_allowed_after_write_to_same_path() -> None:
    session = SessionStore(run_id="test-write-read")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-003",
            "read_file",
            {"path": "app/domain/prices.py"},
            "ok",
            "HEAT-003-0001",
            "old",
        )
        session.record_after_execute(
            "HEAT-003",
            "write_file",
            {"path": "app/domain/prices.py", "content": "new"},
            "ok",
            "HEAT-003-0002",
            "ok",
        )
        refusal = session.check_before_execute(
            "HEAT-003",
            "read_file",
            {"path": "app/domain/prices.py"},
        )
        assert refusal is None
    finally:
        set_session(None)


def test_refuses_tmp_debug_when_subtask_forbids_temp_scripts(monkeypatch: pytest.MonkeyPatch) -> None:
    session = SessionStore(run_id="test-tmp")
    set_session(session)
    begin_worker_attempt("HEAT-003")
    monkeypatch.setattr(
        "orchestrator.session._load_active_subtask",
        lambda task_id: {
            "id": "HEAT-003.2",
            "instructions": ["Do not create temp inspection scripts."],
        },
    )
    try:
        refusal = session.check_before_execute(
            "HEAT-003",
            "write_file",
            {"path": "tmp_debug.py", "content": "print('x')"},
        )
        assert refusal is not None
        assert "Refusing write_file on temp inspection script" in refusal
    finally:
        clear_worker_context()
        set_session(None)


def test_extract_lessons_from_handoffs_records_test_artifact() -> None:
    session = SessionStore(run_id="test-lessons")
    handoff_dir = AGENT_ROOT / "handoffs" / "HEAT-003"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "handoffId": "HEAT-003-9999",
        "stepIndex": 9999,
        "chainId": "HEAT-003",
        "taskId": "HEAT-003",
        "tool": "write_file",
        "input": {"path": "tests/test_prices.py", "content": "..."},
        "result": {"status": "ok", "output": "ok"},
        "summary": "write_file tests/test_prices.py",
    }
    path = handoff_dir / "HEAT-003-9999.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    try:
        session.extract_lessons_from_handoffs(
            "HEAT-003",
            {"progressSummary": "Subtask in progress", "partialFilesTouched": []},
        )
        texts = [lesson["text"] for lesson in session._task("HEAT-003").lessons]
        assert any("tests/test_prices.py" in text for text in texts)
    finally:
        path.unlink(missing_ok=True)


def test_build_retry_context_includes_session_lessons() -> None:
    session = SessionStore(run_id="test-retry")
    set_session(session)
    try:
        session._add_lesson(
            "HEAT-003",
            kind="do_not_repeat",
            text="Do not re-extract haehinnat.",
            source="test",
        )
        context = build_retry_context(
            task_id="HEAT-003",
            failure_progress={
                "taskId": "HEAT-003",
                "status": "FAILED",
                "progressSummary": "partial",
                "previousHandoffId": "HEAT-003-0099",
            },
            recovery_action="Rollback skipped.",
            session=session,
        )
        assert "sessionLessons" in context
        assert any(
            "haehinnat" in lesson.get("text", "")
            for lesson in context["sessionLessons"]
        )
        assert "cachedSubtaskContext" in context
        assert "resumeBrief" in context
        assert "continueFromHandoff" in context
        assert "continuityPacket is authoritative" in context["instruction"]
    finally:
        set_session(None)


def test_build_retry_context_omits_cached_context_when_preflight_has_it() -> None:
    session = SessionStore(run_id="test-compact-retry")
    set_session(session)
    try:
        session._task("HEAT-003").subtask_context_cache["HEAT-003.2"] = {
            "subtaskId": "HEAT-003.2",
            "largeLegacySlice": "x" * 10_000,
        }

        context = build_retry_context(
            task_id="HEAT-003",
            failure_progress={
                "taskId": "HEAT-003",
                "status": "FAILED",
                "activeSubtaskId": "HEAT-003.2",
            },
            recovery_action="Rollback completed.",
            session=session,
            include_cached_context=False,
        )

        assert "cachedSubtaskContext" not in context
        assert "largeLegacySlice" not in json.dumps(context)
    finally:
        set_session(None)


def test_task_user_prompt_includes_session_context_on_retry_payload() -> None:
    session = SessionStore(run_id="test-prompt")
    set_session(session)
    try:
        recovery = {
            "taskId": "HEAT-003",
            "sessionLessons": [{"kind": "gap", "text": "Implement parser."}],
            "cachedSubtaskContext": {"subtaskId": "HEAT-003.2"},
            "instruction": "retry",
        }
        prompt = task_user_prompt("HEAT-003", recovery, session)
        assert "sessionLessons" in prompt
        assert "Implement parser." in prompt
        assert "cachedSubtaskContext" in prompt
    finally:
        set_session(None)


def test_get_session_payload_for_retry_includes_cached_context_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SessionStore(run_id="test-retry-payload")
    monkeypatch.setattr(
        "orchestrator.session._load_active_subtask",
        lambda task_id: {"id": "HEAT-003.2"},
    )
    session._task("HEAT-003").subtask_context_cache["HEAT-003.2"] = {
        "subtaskId": "HEAT-003.2",
        "title": "cached",
    }
    payload = session.get_session_payload_for_retry("HEAT-003")
    assert payload["cachedSubtaskContext"]["subtaskId"] == "HEAT-003.2"


def test_get_session_payload_for_retry_omits_cached_context_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = SessionStore(run_id="test-retry-payload-skip")
    monkeypatch.setattr(
        "orchestrator.session._load_active_subtask",
        lambda task_id: {"id": "HEAT-003.2"},
    )
    session._task("HEAT-003").subtask_context_cache["HEAT-003.2"] = {
        "subtaskId": "HEAT-003.2",
        "title": "cached",
    }
    payload = session.get_session_payload_for_retry(
        "HEAT-003",
        include_cached_context=False,
    )
    assert "cachedSubtaskContext" not in payload
    assert "sessionLessons" in payload
    assert "resumeBrief" in payload


def test_get_subtask_context_returns_cached_result(monkeypatch: pytest.MonkeyPatch) -> None:
    session = SessionStore(run_id="test-cache")
    set_session(session)
    monkeypatch.setattr(
        "orchestrator.session._load_active_subtask",
        lambda task_id: {"id": "HEAT-003.2"},
    )
    try:
        session._task("HEAT-003").subtask_context_cache["HEAT-003.2"] = {
            "subtaskId": "HEAT-003.2",
            "title": "cached",
        }
        cached = session.check_before_execute(
            "HEAT-003",
            "get_subtask_context",
            {"task_id": "HEAT-003"},
        )
        assert cached is not None
        assert cached.startswith("CACHED:")
        assert "HEAT-003.2" in cached
    finally:
        set_session(None)


def test_refuses_run_command_that_cats_already_read_file() -> None:
    session = SessionStore(run_id="test-cat-file")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/services/schedule_store.py"},
            "ok",
            "HEAT-009-0005",
            "file content",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "run_command",
            {
                "command": (
                    "python -c \"print(open('app/services/schedule_store.py').read())\""
                ),
            },
        )
        assert refusal is not None
        assert "Refusing run_command" in refusal
        assert "schedule_store.py" in refusal
    finally:
        set_session(None)


def test_visible_output_creates_discovery_lesson() -> None:
    session = SessionStore(run_id="test-visible")
    session.record_visible_output(
        "HEAT-003",
        "The fixture uses flattened price.amount tags while tests use nested price/amount.",
    )
    kinds = [lesson["kind"] for lesson in session._task("HEAT-003").lessons]
    assert "discovery" in kinds


def test_blocks_duplicate_read_handoff() -> None:
    session = SessionStore(run_id="test-read-handoff")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0015", "include_result": True},
            "ok",
            "HEAT-009-0015",
            "payload",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0015", "include_result": True},
        )
        assert refusal is not None
        assert "Duplicate read_handoff" in refusal
    finally:
        set_session(None)


def test_allows_read_handoff_once_for_truncated_handoff() -> None:
    session = SessionStore(run_id="test-truncated-expand")
    set_session(session)
    try:
        session.note_truncated_handoff("HEAT-009", "HEAT-009-0005", "partial\n[truncated; full output stored in handoff]")
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "tests/test_schedule_toggle.py"},
            "ok",
            "HEAT-009-0005",
            "x" * 2000,
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0005", "include_result": True},
        )
        assert refusal is None

        session.record_after_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0005", "include_result": True},
            "ok",
            "HEAT-009-0005",
            "full payload",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0005", "include_result": True},
        )
        assert refusal is not None
        assert "Duplicate read_handoff" in refusal
    finally:
        set_session(None)


def test_run_command_still_blocked_when_truncated() -> None:
    session = SessionStore(run_id="test-cat-truncated")
    set_session(session)
    try:
        session.note_truncated_handoff("HEAT-009", "HEAT-009-0005", "partial\n[truncated; full output stored in handoff]")
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "tests/test_schedule_toggle.py"},
            "ok",
            "HEAT-009-0005",
            "x" * 2000,
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "run_command",
            {
                "command": (
                    "python -c \"print(open('tests/test_schedule_toggle.py').read())\""
                ),
            },
        )
        assert refusal is not None
        assert "Refusing run_command" in refusal
        assert "read_handoff" in refusal
        assert "HEAT-009-0005" in refusal
    finally:
        set_session(None)


def test_duplicate_read_file_suggests_read_handoff_when_truncated() -> None:
    session = SessionStore(run_id="test-read-file-truncated")
    set_session(session)
    try:
        session.note_truncated_handoff("HEAT-009", "HEAT-009-0005", "partial\n[truncated; full output stored in handoff]")
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "tests/test_schedule_toggle.py"},
            "ok",
            "HEAT-009-0005",
            "x" * 2000,
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "tests/test_schedule_toggle.py"},
        )
        assert refusal is not None
        assert "Duplicate read_file" in refusal
        assert "read_handoff" in refusal
        assert "HEAT-009-0005" in refusal
    finally:
        set_session(None)


def test_blocks_get_subtask_context_when_preflight_satisfied() -> None:
    session = SessionStore(run_id="test-preflight-satisfied")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=True)
        refusal = session.check_before_execute(
            "HEAT-009",
            "get_subtask_context",
            {"task_id": "HEAT-009"},
        )
        assert refusal is not None
        assert "contextPack already includes" in refusal
    finally:
        set_session(None)


def test_blocks_read_file_on_preloaded_path_when_preflight_satisfied() -> None:
    session = SessionStore(run_id="test-preloaded-path")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=True)
        session.set_preloaded_context(
            "HEAT-009",
            preloaded_paths={"app/services/schedule_store.py", "app/domain/schedule.py"},
            target_module_path="app/api/schedule.py",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/services/schedule_store.py"},
        )
        assert refusal is not None
        assert "referenceSnippets already includes" in refusal
    finally:
        set_session(None)


@pytest.mark.parametrize(
    "command",
    [
        "type app\\services\\schedule_store.py",
        "powershell -NoProfile -Command \"Get-Content app/services/schedule_store.py\"",
        "cat app/services/schedule_store.py",
    ],
)
def test_shell_file_reads_cannot_bypass_preloaded_context(
    command: str,
) -> None:
    session = SessionStore(run_id="test-preloaded-shell-read")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=True)
        session.set_preloaded_context(
            "HEAT-009",
            preloaded_paths={"app/services/schedule_store.py"},
        )

        refusal = session.check_before_execute(
            "HEAT-009",
            "run_command",
            {"command": command},
        )

        assert refusal is not None
        assert "guarded path" in refusal
        assert "replace_in_file" in refusal
    finally:
        set_session(None)


def test_repeated_failed_read_is_deduplicated() -> None:
    session = SessionStore(run_id="test-failed-read-dedup")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/missing.py"},
            "error",
            "HEAT-009-0001",
            "ERROR executing read_file: File does not exist",
        )

        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/missing.py"},
        )

        assert refusal is not None
        assert "already refused" in refusal
        assert "Do not repeat" in refusal
    finally:
        set_session(None)


def test_allows_read_file_on_target_module_not_in_snippets() -> None:
    session = SessionStore(run_id="test-target-module")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=True)
        session.set_preloaded_context(
            "HEAT-009",
            preloaded_paths={"app/domain/schedule.py"},
            target_module_path="app/api/schedule.py",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/api/schedule.py"},
        )
        assert refusal is None

        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/api/schedule.py"},
            "ok",
            "HEAT-009-0001",
            "router content",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/api/schedule.py"},
        )
        assert refusal is not None
        assert "target module" in refusal or "Duplicate read_file" in refusal
    finally:
        set_session(None)


def test_pending_write_gate_blocks_exploration_after_read_file() -> None:
    session = SessionStore(run_id="test-pending-write")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=False)
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example.py"},
            "ok",
            "HEAT-009-0001",
            "content",
            next_action={"tool": "write_file", "hint": "implement using context from app/example.py"},
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/other.py"},
        )
        assert refusal is not None
        assert "Pending next action is write_file" in refusal

        write_refusal = session.check_before_execute(
            "HEAT-009",
            "write_file",
            {"path": "app/example.py", "content": "new"},
        )
        assert write_refusal is None
    finally:
        set_session(None)


def test_pending_write_gate_cleared_after_successful_write() -> None:
    session = SessionStore(run_id="test-pending-clear")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=False)
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example.py"},
            "ok",
            "HEAT-009-0001",
            "content",
            next_action={"tool": "write_file", "hint": "implement"},
        )
        session.record_after_execute(
            "HEAT-009",
            "write_file",
            {"path": "app/example.py", "content": "new"},
            "ok",
            "HEAT-009-0002",
            "ok",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/other.py"},
        )
        assert refusal is None
    finally:
        set_session(None)


def test_exploration_budget_soft_then_hard() -> None:
    session = SessionStore(run_id="test-budget")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=False, max_read_only_before_write=4)
        task = session._task("HEAT-009")
        task.read_only_warn_threshold = 2
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example_0.py"},
            "ok",
            "HEAT-009-0001",
            "content",
        )
        assert session.exploration_budget_warning("HEAT-009") is None

        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example_1.py"},
            "ok",
            "HEAT-009-0002",
            "content",
        )
        warning = session.exploration_budget_warning("HEAT-009")
        assert warning is not None
        assert "Exploration budget" in warning

        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example_2.py"},
            "ok",
            "HEAT-009-0003",
            "content",
        )
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example_3.py"},
            "ok",
            "HEAT-009-0004",
            "content",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "search_files",
            {"query": "toggle", "path": "app"},
        )
        assert refusal is not None
        assert "Exploration budget exhausted" in refusal
    finally:
        set_session(None)


def test_write_file_resets_exploration_budget() -> None:
    session = SessionStore(run_id="test-write-reset")
    set_session(session)
    try:
        session.set_exploration_policy("HEAT-009", preflight_satisfied=False, max_read_only_before_write=2)
        session.record_after_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/example.py"},
            "ok",
            "HEAT-009-0001",
            "content",
        )
        session.record_after_execute(
            "HEAT-009",
            "write_file",
            {"path": "app/example.py", "content": "new"},
            "ok",
            "HEAT-009-0002",
            "ok",
        )
        refusal = session.check_before_execute(
            "HEAT-009",
            "read_file",
            {"path": "app/other.py"},
        )
        assert refusal is None
        assert session.get_read_only_tools_used("HEAT-009") == 0
    finally:
        set_session(None)
