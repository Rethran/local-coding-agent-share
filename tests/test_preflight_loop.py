from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orchestrator import config
from orchestrator.preflight import run_preflight
from tools import _tasks


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "model": "test-model",
        "planner_model": "",
        "max_planner_rounds": 2,
        "enable_planner": False,
        "skip_planner": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_task(requirements_dir: Path, task: dict) -> None:
    task_id = task["id"]
    (requirements_dir / f"{task_id}.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_preflight_ready_for_rich_subtask_without_planner() -> None:
    state_path = _tasks.state_path("HEAT-003")
    backup = state_path.read_text(encoding="utf-8") if state_path.exists() else None
    try:
        result = run_preflight("HEAT-003", "HEAT-003.2", args=_args())
        assert result.ready is True
        assert result.context_pack is not None
        assert result.merged_subtask is not None
    finally:
        if backup is None:
            if state_path.exists():
                state_path.unlink()
        else:
            state_path.write_text(backup, encoding="utf-8")


def test_skip_planner_blocks_when_golden_rules_fail() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        plans_dir = Path(temp_dir) / "plans"
        original_requirements = _tasks.REQUIREMENTS_DIR
        original_plans = config.PLANS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        config.PLANS_DIR = plans_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-903",
                    "summary": "Blocked preflight",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "scope": ["Create app/domain/example.py"],
                    "verification": ["pytest tests/ -v"],
                },
            )
            result = run_preflight(
                "HEAT-903",
                "HEAT-903.1",
                args=_args(max_planner_rounds=1),
            )
            assert result.ready is False
            assert result.blockers
        finally:
            _tasks.REQUIREMENTS_DIR = original_requirements
            config.PLANS_DIR = original_plans


def test_preflight_calls_planner_then_becomes_ready() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        plans_dir = Path(temp_dir) / "plans"
        original_requirements = _tasks.REQUIREMENTS_DIR
        original_plans = config.PLANS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        config.PLANS_DIR = plans_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-904",
                    "summary": "Planner rescue",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "scope": ["Create app/domain/example.py"],
                    "verification": ["pytest tests/ -v"],
                },
            )

            def fake_planner(*_args, **_kwargs):
                sidecar = {
                    "taskId": "HEAT-904",
                    "subtaskId": "HEAT-904.1",
                    "plannerRound": 1,
                    "enrichments": {
                        "verification": ["python -c \"import app.domain.example\""],
                        "targetArtifacts": {
                            "module": "app/domain/example.py",
                            "function": "example",
                        },
                    },
                    "addressedRuleIds": ["GR-DERIVED-001"],
                }
                path = plans_dir / "HEAT-904.HEAT-904.1.plan.json"
                plans_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")
                return sidecar

            with patch("orchestrator.preflight.run_llm_planner", side_effect=fake_planner):
                result = run_preflight(
                    "HEAT-904",
                    "HEAT-904.1",
                    args=_args(max_planner_rounds=2, enable_planner=True),
                )
            assert result.ready is True
            assert result.merged_subtask is not None
            assert result.merged_subtask.get("targetArtifacts")
        finally:
            _tasks.REQUIREMENTS_DIR = original_requirements
            config.PLANS_DIR = original_plans


def test_preflight_completer_repairs_shared_verification_without_planner() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        plans_dir = Path(temp_dir) / "plans"
        original_requirements = _tasks.REQUIREMENTS_DIR
        original_plans = config.PLANS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        config.PLANS_DIR = plans_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-905",
                    "summary": "Completer rescue",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "subtasks": [
                        {
                            "id": "HEAT-905.1",
                            "title": "First",
                            "summary": "First",
                            "readLegacy": False,
                            "scope": ["Add app/domain/example.py"],
                            "targetArtifacts": {
                                "module": "app/domain/example.py",
                                "test": "tests/test_example.py",
                            },
                            "verification": ["pytest tests/test_example.py -q"],
                        },
                        {
                            "id": "HEAT-905.2",
                            "title": "Second",
                            "summary": "Second",
                            "readLegacy": False,
                            "scope": ["Add app/domain/example.py"],
                            "targetArtifacts": {
                                "module": "app/domain/example.py",
                                "test": "tests/test_example.py",
                            },
                            "verification": ["pytest tests/test_example.py -q"],
                        },
                    ],
                },
            )
            result = run_preflight("HEAT-905", "HEAT-905.1", args=_args())
            assert result.ready is True
            saved = json.loads((requirements_dir / "HEAT-905.json").read_text(encoding="utf-8"))
            assert saved["subtasks"][0]["verification"] == [
                "pytest tests/test_example.py -k HEAT_905_1 -q"
            ]
            assert saved["acceptanceCriteria"] == "Works"
        finally:
            _tasks.REQUIREMENTS_DIR = original_requirements
            config.PLANS_DIR = original_plans
