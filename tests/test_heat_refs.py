"""Tests for task/subtask CLI refs (--from HEAT-003.4 --to HEAT-004)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import _tasks

AGENT_ROOT = Path(__file__).resolve().parent.parent


def _write_task(requirements_dir: Path, task_id: str, steps: int) -> None:
    task = {
        "id": task_id,
        "summary": task_id,
        "dependsOn": [],
        "acceptanceCriteria": "Works",
        "scope": [f"step {index}" for index in range(1, steps + 1)],
        "verification": ["python -m pytest tests -q"],
    }
    (requirements_dir / f"{task_id}.json").write_text(
        json.dumps(task),
        encoding="utf-8",
    )


@pytest.fixture
def heat_requirements(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    for task_id, steps in (
        ("HEAT-002", 1),
        ("HEAT-003", 7),
        ("HEAT-004", 1),
        ("HEAT-005", 1),
    ):
        _write_task(tmp_path, task_id, steps)
    monkeypatch.setattr(_tasks, "REQUIREMENTS_DIR", tmp_path)
    monkeypatch.setattr(_tasks, "TASK_PREFIX", "HEAT")
    return tmp_path


def test_parse_heat_ref_task_and_subtask() -> None:
    task_only = _tasks.parse_heat_ref("HEAT-003")
    assert task_only["taskId"] == "HEAT-003"
    assert task_only["subtaskId"] is None

    with_sub = _tasks.parse_heat_ref("HEAT-003.4")
    assert with_sub["taskId"] == "HEAT-003"
    assert with_sub["subtaskId"] == "HEAT-003.4"


def test_parse_heat_ref_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="Invalid HEAT id"):
        _tasks.parse_heat_ref("TASK-003")


def test_parse_task_number_accepts_subtask_ref() -> None:
    assert _tasks.parse_task_number("HEAT-003") == 3
    assert _tasks.parse_task_number("HEAT-003.4") == 3
    assert _tasks.parse_task_number("HEAT-010.2") == 10


def test_filter_task_range_with_subtask_bounds(heat_requirements: Path) -> None:
    tasks = _tasks.load_task_definitions()
    filtered = _tasks.filter_task_range(tasks, "HEAT-003.4", "HEAT-004")
    assert "HEAT-003" in filtered
    assert "HEAT-004" in filtered
    assert "HEAT-002" not in filtered
    assert "HEAT-005" not in filtered


def test_pin_from_subtask_sets_active(heat_requirements: Path) -> None:
    state = _tasks.pin_from_subtask("HEAT-003", "HEAT-003.4")
    assert state["status"] == "IN_PROGRESS"
    assert state["activeSubtaskId"] == "HEAT-003.4"
    by_id = {item["id"]: item for item in state["subtasks"]}
    assert by_id["HEAT-003.1"]["status"] == "DONE"
    assert by_id["HEAT-003.2"]["status"] == "DONE"
    assert by_id["HEAT-003.3"]["status"] == "DONE"
    assert by_id["HEAT-003.4"]["status"] == "IN_PROGRESS"


def test_skip_subtasks_after_marks_later_done(heat_requirements: Path) -> None:
    _tasks.pin_from_subtask("HEAT-003", "HEAT-003.4")
    state = _tasks.skip_subtasks_after("HEAT-003", "HEAT-003.6")
    by_id = {item["id"]: item for item in state["subtasks"]}
    assert by_id["HEAT-003.6"]["status"] in ("PENDING", "IN_PROGRESS", "DONE")
    assert by_id["HEAT-003.7"]["status"] == "DONE"
    assert "Skipped by orchestrator" in by_id["HEAT-003.7"]["summary"]
    assert state["activeSubtaskId"] == "HEAT-003.4"


def test_apply_cli_heat_range_from_subtask_to_task(heat_requirements: Path) -> None:
    plan = _tasks.apply_cli_heat_range("HEAT-003.5", "HEAT-004")
    assert plan["fromSubtaskId"] == "HEAT-003.5"
    assert plan["toSubtaskId"] is None
    assert plan["pinned"]["activeSubtaskId"] == "HEAT-003.5"
    assert plan["skipped"] is None


def test_cli_mentions_subtask_range_examples() -> None:
    text = (AGENT_ROOT / "orchestrator" / "cli.py").read_text(encoding="utf-8")
    assert "--from {TASK_PREFIX}-003.4 --to {TASK_PREFIX}-004" in text
    assert "--from {TASK_PREFIX}-003.4 --to {TASK_PREFIX}-003.6" in text
