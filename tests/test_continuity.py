"""Tests for Subtask Continuity Protocol (SCP)."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from orchestrator.attempts import build_retry_context
from orchestrator.continuity import (
    CONTINUITY_DIR,
    build_continuity_packet,
    infer_tool_next_action,
    is_cat_file_command,
    load_continuity_packet,
    persist_continuity_packet,
    persist_duplicate_failure,
    validate_continuity_packet,
)
from orchestrator.prompts import task_user_prompt_for_subtask
from orchestrator.session import SessionStore, set_session
from tools import _handoffs, _tasks

AGENT_ROOT = Path(__file__).resolve().parent.parent


def test_unrelated_successful_command_routes_to_exact_declared_verification() -> None:
    action = infer_tool_next_action(
        "run_command",
        {"command": "pytest tests/test_schedule_api.py -v"},
        "ok",
        "Exit code: 0\npassed",
        merged_subtask={"verification": ["pytest tests/ -k toggle_api -v"]},
    )

    assert action == {
        "tool": "run_command",
        "hint": "run exact declared verification: pytest tests/ -k toggle_api -v",
    }


@pytest.fixture
def continuity_temp(monkeypatch: pytest.MonkeyPatch):
    with TemporaryDirectory() as temp_dir:
        continuity_root = Path(temp_dir) / "continuity"
        handoffs_root = Path(temp_dir) / "handoffs"
        monkeypatch.setattr("orchestrator.continuity.CONTINUITY_DIR", continuity_root)
        monkeypatch.setattr("orchestrator.continuity.continuity_dir", lambda task_id: continuity_root / task_id)
        monkeypatch.setattr(
            "orchestrator.continuity.latest_packet_path",
            lambda task_id, subtask_id: continuity_root / task_id / f"{subtask_id}.latest.json",
        )
        monkeypatch.setattr(
            "orchestrator.continuity.attempt_packet_path",
            lambda task_id, subtask_id, attempt: continuity_root / task_id / f"{subtask_id}.attempt-{attempt}.json",
        )
        monkeypatch.setattr(_handoffs, "HANDOFFS_DIR", handoffs_root)
        monkeypatch.setattr(_handoffs, "chain_dir", lambda chain_id: handoffs_root / chain_id)
        yield continuity_root


def _minimal_subtask() -> dict:
    return {
        "id": "HEAT-005.2",
        "title": "Implement select_cheapest_indices pure function",
        "scope": ["Add select_cheapest_indices to app/domain/schedule.py"],
        "verification": ["python -m unittest tests.test_schedule.SelectCheapestIndicesTest -v"],
        "targetArtifacts": {
            "module": "app/domain/schedule.py",
            "function": "select_cheapest_indices",
            "test": "tests/test_schedule.py",
        },
    }


def test_validate_continuity_packet_requires_next_step_on_failed() -> None:
    packet = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        status="failed",
        merged_subtask=_minimal_subtask(),
        stop_reason="no_tool_call",
        blockers=["stopped"],
    )
    assert validate_continuity_packet(packet) == []


def test_build_continuity_packet_from_session_and_handoffs(continuity_temp: Path) -> None:
    task_id = "HEAT-005"
    subtask_id = "HEAT-005.2"
    chain_dir = _handoffs.chain_dir(task_id)
    chain_dir.mkdir(parents=True, exist_ok=True)
    (chain_dir / "HEAT-005-0001.json").write_text(
        json.dumps(
            {
                "handoffId": "HEAT-005-0001",
                "tool": "read_file",
                "summary": "Read file output: 10 lines, 100 chars.",
                "input": {"path": "app/domain/schedule.py"},
                "result": {"status": "ok", "output": "content"},
            }
        ),
        encoding="utf-8",
    )

    session = SessionStore(run_id="scp-test")
    session.record_after_execute(
        task_id,
        "read_file",
        {"path": "app/domain/schedule.py"},
        "ok",
        "HEAT-005-0001",
        "content",
    )

    packet = build_continuity_packet(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt=1,
        status="failed",
        merged_subtask=_minimal_subtask(),
        session=session,
        stop_reason="no_tool_call",
        blockers=["Assistant stopped without tool call at step 72."],
        last_handoff_id="HEAT-005-0001",
    )

    paths = {item["path"] for item in packet["artifactsReviewed"]}
    assert "app/domain/schedule.py" in paths
    assert any(
        item["target"] == "app/domain/schedule.py"
        for item in packet["doNotRepeat"]
    )
    assert packet["nextStep"]
    assert packet["evidence"]


def test_persist_and_load_continuity_packet(continuity_temp: Path) -> None:
    packet = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=2,
        status="failed",
        merged_subtask=_minimal_subtask(),
        stop_reason="tool_step_budget",
        blockers=["budget exhausted"],
    )
    persist_continuity_packet("HEAT-005", "HEAT-005.2", packet)
    loaded = load_continuity_packet("HEAT-005", "HEAT-005.2")
    assert loaded is not None
    assert loaded["schemaVersion"] == "scp/v1"
    assert loaded["attempt"] == 2


def test_duplicate_failure_replaces_already_satisfied_packet(
    continuity_temp: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        status="in_progress",
        merged_subtask=_minimal_subtask(),
        stop_reason="already_satisfied",
        recommended_resume_step=(
            "Existing implementation already satisfies the declared tests; "
            "the coordinator will mark the subtask DONE after the verifier."
        ),
    )
    stale["verifiedBehaviors"] = ["Tests passed: pytest tests/test_schedule.py"]
    stale["doNotRepeat"] = [
        {"action": "read_file", "target": "app/domain/schedule.py", "reason": "already reviewed"}
    ]
    stale["evidence"] = [{"kind": "test", "command": "pytest", "exitCode": 0, "summary": "green"}]
    stale["lastHandoffId"] = "HEAT-005-0004"
    persist_continuity_packet("HEAT-005", "HEAT-005.2", stale)

    written: dict = {}

    def update_task_state(task_id: str, status: str, **kwargs):
        written["task_id"] = task_id
        written["status"] = status
        written.update(kwargs)
        return {"status": status, **kwargs}

    monkeypatch.setattr(
        "orchestrator.continuity._tasks.load_task_state",
        lambda task_id: {
            "status": "FAILED",
            "verifiedBehaviors": ["Tests passed: pytest tests/test_schedule.py"],
        },
    )
    monkeypatch.setattr("orchestrator.continuity._tasks.update_task_state", update_task_state)

    blocker = "HEAT-005.HEAT-005.2 dataExamples are not executable by the coordinator oracle."
    resume = (
        "Make the implementation satisfy this subtask's dataExamples, or give "
        "the examples an executable input/output shape."
    )
    persist_duplicate_failure(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        merged_subtask=_minimal_subtask(),
        session=None,
        blockers=[blocker],
        recommended_resume_step=resume,
    )

    loaded = load_continuity_packet("HEAT-005", "HEAT-005.2")
    assert loaded is not None
    assert loaded["status"] == "failed"
    assert loaded["stopReason"] == "duplicate_contract"
    assert loaded["nextStep"] == resume
    assert loaded["blockers"] == [blocker]
    assert loaded["verifiedBehaviors"] == ["Tests passed: pytest tests/test_schedule.py"]
    assert loaded["lastHandoffId"] == "HEAT-005-0004"
    assert loaded["doNotRepeat"][0]["target"] == "app/domain/schedule.py"
    lesson = next(
        item for item in loaded["doNotRepeat"] if item["reason"] == "duplicate_contract"
    )
    assert "Do not submit another green test as done" in lesson["target"]
    assert blocker in lesson["target"]
    assert written["status"] == "FAILED"
    assert written["recommended_resume_step"] == resume
    assert written["last_failure_reason"] == blocker


def test_build_retry_context_includes_continuity_packet(continuity_temp: Path) -> None:
    packet = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        status="failed",
        merged_subtask=_minimal_subtask(),
        stop_reason="no_tool_call",
        blockers=["stopped"],
    )
    persist_continuity_packet("HEAT-005", "HEAT-005.2", packet)

    session = SessionStore(run_id="retry-scp")
    set_session(session)
    try:
        context = build_retry_context(
            task_id="HEAT-005",
            failure_progress={
                "taskId": "HEAT-005",
                "activeSubtaskId": "HEAT-005.2",
                "progressSummary": "failed",
                "recommendedResumeStep": "continue",
            },
            recovery_action="Rollback skipped.",
            session=session,
        )
        assert "continuityPacket" in context
        assert context["continuityPacket"]["nextStep"]
        assert "continuityPacket is authoritative" in context["instruction"]
    finally:
        set_session(None)


def test_task_user_prompt_includes_continuity_packet(continuity_temp: Path) -> None:
    packet = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        status="failed",
        merged_subtask={
            "id": "HEAT-005.2",
            "title": "Cheapest selection",
            "scope": ["Add select_cheapest_indices"],
            "verification": ["python -m unittest tests.test_schedule -v"],
        },
        stop_reason="no_tool_call",
        blockers=["stopped"],
    )
    persist_continuity_packet("HEAT-005", "HEAT-005.2", packet)

    prompt = task_user_prompt_for_subtask(
        "HEAT-005",
        "HEAT-005.2",
        attempt_number=2,
    )
    assert "continuityPacket" in prompt
    assert "doNotRepeat" in prompt


def test_build_continuity_packet_prefers_instructions_for_next_step(continuity_temp: Path) -> None:
    packet = build_continuity_packet(
        task_id="HEAT-009",
        subtask_id="HEAT-009.2",
        attempt=1,
        status="failed",
        merged_subtask={
            "id": "HEAT-009.2",
            "title": "Add schedule toggle API endpoint",
            "instructions": [
                "Implement toggle_today_slot in schedule_store first.",
                "Run pytest tests/ -k toggle_api -v.",
            ],
            "scope": ["Add slot toggle endpoint"],
            "verification": ["pytest tests/ -k toggle_api -v"],
        },
        stop_reason="tool_step_budget",
        blockers=["budget exhausted"],
    )
    assert packet["nextStep"] == "Implement toggle_today_slot in schedule_store first."


def test_build_do_not_repeat_includes_search_and_read_handoff() -> None:
    session = SessionStore(run_id="test-dnr")
    set_session(session)
    try:
        session.record_after_execute(
            "HEAT-009",
            "search_files",
            {"query": "def ", "path": "app/services/schedule_store.py"},
            "ok",
            "HEAT-009-0010",
            "[]",
        )
        session.record_after_execute(
            "HEAT-009",
            "read_handoff",
            {"handoff_id": "HEAT-009-0015", "include_result": True},
            "ok",
            "HEAT-009-0015",
            "payload",
        )
        packet = build_continuity_packet(
            task_id="HEAT-009",
            subtask_id="HEAT-009.2",
            attempt=1,
            status="failed",
            session=session,
            stop_reason="tool_step_budget",
            blockers=["budget exhausted"],
        )
        actions = {item["action"] for item in packet["doNotRepeat"]}
        assert "search_files" in actions
        assert "read_handoff" in actions
        assert packet["readOnlyToolsUsed"] == 2
    finally:
        set_session(None)


def test_is_cat_file_command_detects_duplicate_read() -> None:
    known = {"app/services/schedule_store.py"}
    command = 'python -c "print(open(\'app/services/schedule_store.py\').read())"'
    assert is_cat_file_command(command, known) == "app/services/schedule_store.py"


def test_build_resume_brief_uses_subtask_scope(continuity_temp: Path) -> None:
    packet = build_continuity_packet(
        task_id="HEAT-005",
        subtask_id="HEAT-005.2",
        attempt=1,
        status="failed",
        merged_subtask={
            "id": "HEAT-005.2",
            "title": "Cheapest selection",
            "scope": ["Add select_cheapest_indices"],
            "verification": ["python -m unittest tests.test_schedule -v"],
            "targetArtifacts": {"module": "app/domain/schedule.py", "function": "select_cheapest_indices"},
        },
        stop_reason="no_tool_call",
        blockers=["stopped"],
    )
    persist_continuity_packet("HEAT-005", "HEAT-005.2", packet)

    state_path = AGENT_ROOT / "requirements" / "HEAT-005.state.json"
    original = state_path.read_text(encoding="utf-8") if state_path.exists() else None
    try:
        if state_path.exists():
            state = json.loads(original or "{}")
            state["activeSubtaskId"] = "HEAT-005.2"
            state["status"] = "FAILED"
            state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

        brief = _tasks.build_resume_brief("HEAT-005")
        assert brief.get("activeSubtaskId") == "HEAT-005.2"
        assert "select_cheapest_indices" in str(brief.get("scope"))
        assert brief.get("continuityPacket") is not None
    finally:
        if original is not None:
            state_path.write_text(original, encoding="utf-8")
