from __future__ import annotations

import json
from pathlib import Path

from tools import _tasks

AGENT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = AGENT_ROOT / "requirements"


def test_active_subtask_is_in_progress_when_task_starts() -> None:
    task_id = "HEAT-003"
    state_path = REQUIREMENTS_DIR / f"{task_id}.state.json"
    backup = state_path.read_text(encoding="utf-8") if state_path.exists() else None

    try:
        if state_path.exists():
            state_path.unlink()
        state = _tasks.update_task_state(
            task_id,
            "IN_PROGRESS",
            summary="Test activation.",
        )
        active_id = state["activeSubtaskId"]
        assert active_id == "HEAT-003.1"

        active = next(item for item in state["subtasks"] if item["id"] == active_id)
        assert active["status"] == "IN_PROGRESS"

        pending = [item for item in state["subtasks"] if item["id"] != active_id]
        assert all(item["status"] == "PENDING" for item in pending)
    finally:
        if backup is None:
            if state_path.exists():
                state_path.unlink()
        else:
            state_path.write_text(backup, encoding="utf-8")


def test_derive_subtasks_preserves_summary_from_task_definition() -> None:
    task = json.loads((REQUIREMENTS_DIR / "HEAT-003.json").read_text(encoding="utf-8"))
    subtasks = _tasks.derive_subtasks("HEAT-003", task)

    first = subtasks[0]
    assert first["id"] == "HEAT-003.1"
    assert "PriceSlot" in first["summary"]
    assert "Do not read" in first["summary"]


def test_skeleton_subtask_scoped_prompt_omits_legacy_sources() -> None:
    from orchestrator.prompts import _scope_task_payload

    task = json.loads((REQUIREMENTS_DIR / "HEAT-003.json").read_text(encoding="utf-8"))
    subtask_state = _tasks.normalize_subtask_state("HEAT-003", task, {"status": "IN_PROGRESS"})
    scoped = _scope_task_payload(task, subtask_state)

    impl = scoped.get("implementationLogic", {})
    assert "legacySources" not in impl
    assert "testFixtures" not in impl
    assert scoped["activeSubtask"]["readLegacy"] is False
    assert "instructions" in scoped
    assert any("old_entsoes" in line for line in scoped["instructions"])


def test_legacy_subtask_scoped_prompt_includes_search_metadata() -> None:
    from orchestrator.prompts import _scope_task_payload

    task = json.loads((REQUIREMENTS_DIR / "HEAT-003.json").read_text(encoding="utf-8"))
    state = {
        "status": "IN_PROGRESS",
        "activeSubtaskId": "HEAT-003.2",
        "subtasks": [
            {"id": "HEAT-003.1", "title": "skeleton", "status": "DONE", "summary": ""},
            {"id": "HEAT-003.2", "title": "parser", "status": "IN_PROGRESS", "summary": ""},
        ],
    }
    subtask_state = _tasks.normalize_subtask_state("HEAT-003", task, state)
    scoped = _scope_task_payload(task, subtask_state)

    assert scoped["activeSubtask"]["readLegacy"] is True
    assert "legacySymbols" in scoped
    assert "allowedSearchTerms" in scoped
    assert "targetArtifacts" in scoped
    assert any("get_subtask_context" in line for line in scoped["instructions"])
    assert any(item["name"] == "haehinnat" for item in scoped["legacySymbols"])


def test_subtask_prompt_includes_context_pack_and_merged_subtask() -> None:
    from orchestrator.context_pack import build_context_pack
    from orchestrator.plan_merge import merge_subtask_definition
    from orchestrator.prompts import task_user_prompt_for_subtask

    task_id = "HEAT-003"
    subtask_id = "HEAT-003.2"
    spec_subtask = _tasks.get_subtask_spec_definition(task_id, subtask_id)
    assert spec_subtask is not None
    context_pack = build_context_pack(task_id, subtask_id)
    merged = merge_subtask_definition(spec_subtask, None)
    prompt = task_user_prompt_for_subtask(
        task_id,
        subtask_id,
        context_pack=context_pack,
        merged_subtask=merged,
    )

    assert subtask_id in prompt
    assert "contextPack" in prompt
    assert "parse_entsoe_prices" in prompt


def test_task_user_prompt_omits_cached_subtask_context_when_preflight_satisfied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from orchestrator.context_pack import build_context_pack
    from orchestrator.plan_merge import merge_subtask_definition
    from orchestrator.prompts import task_user_prompt_for_subtask
    from orchestrator.session import SessionStore, set_session

    session = SessionStore(run_id="test-dedupe-session")
    set_session(session)
    monkeypatch.setattr(
        SessionStore,
        "should_inject_on_first_attempt",
        lambda self, task_id: True,
    )
    try:
        task_id = "HEAT-009"
        subtask_id = "HEAT-009.2"
        context_pack = build_context_pack(task_id, subtask_id)
        assert context_pack["explorationPolicy"]["preflightSatisfied"] is True
        spec_subtask = _tasks.get_subtask_spec_definition(task_id, subtask_id)
        assert spec_subtask is not None
        merged = merge_subtask_definition(spec_subtask, None)
        session._task(task_id).subtask_context_cache[subtask_id] = {
            "subtaskId": subtask_id,
            "title": "cached",
        }

        prompt = task_user_prompt_for_subtask(
            task_id,
            subtask_id,
            context_pack=context_pack,
            merged_subtask=merged,
            session=session,
        )

        assert "sessionContext" in prompt
        assert "cachedSubtaskContext" not in prompt
    finally:
        set_session(None)


def test_cannot_mark_task_done_with_incomplete_subtasks() -> None:
    task_id = "HEAT-003"
    state_path = REQUIREMENTS_DIR / f"{task_id}.state.json"
    backup = state_path.read_text(encoding="utf-8") if state_path.exists() else None

    try:
        _tasks.update_task_state(task_id, "IN_PROGRESS", summary="Setup incomplete subtasks.")
        try:
            _tasks.update_task_state(task_id, "DONE", summary="Should fail.")
            raise AssertionError("Expected ValueError when marking DONE with incomplete subtasks")
        except ValueError as exc:
            message = str(exc)
            assert "Cannot mark HEAT-003 DONE" in message
            assert "subtasks remain incomplete" in message
            assert any(
                subtask_id in message
                for subtask_id in (
                    "HEAT-003.1",
                    "HEAT-003.2",
                    "HEAT-003.3",
                    "HEAT-003.4",
                    "HEAT-003.5",
                    "HEAT-003.6",
                    "HEAT-003.7",
                )
            )
    finally:
        if backup is None:
            if state_path.exists():
                state_path.unlink()
        else:
            state_path.write_text(backup, encoding="utf-8")
