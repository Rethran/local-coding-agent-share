from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import config
from orchestrator.spec_completer import (
    complete_subtask_verification,
    persist_verification_repair,
)
from orchestrator.verification_spec import slice_token
from tools import _tasks


def _write_task(requirements_dir: Path, task: dict) -> None:
    (requirements_dir / f"{task['id']}.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_slice_token_strips_pytest_expression_characters() -> None:
    assert slice_token("GRAPH-003.1") == "GRAPH_003_1"
    assert slice_token("HEAT-009.2") == "HEAT_009_2"


def test_completer_adds_unique_k_and_leaves_heat_style_selectors() -> None:
    task = {
        "id": "DEMO-001",
        "subtasks": [
            {
                "id": "DEMO-001.1",
                "targetArtifacts": {"test": "tests/test_demo.py"},
                "verification": ["python -m pytest tests/test_demo.py -q"],
            },
            {
                "id": "DEMO-001.2",
                "targetArtifacts": {"test": "tests/test_demo.py"},
                "verification": ["python -m pytest tests/test_demo.py -q"],
            },
        ],
    }
    repaired = complete_subtask_verification(task, "DEMO-001.1", task["subtasks"][0])
    assert repaired == ["python -m pytest tests/test_demo.py -k DEMO_001_1 -q"]

    heat_style = {
        "id": "DEMO-001.1",
        "targetArtifacts": {"test": "tests/test_demo.py"},
        "verification": ["pytest tests/ -k toggle_domain -v"],
    }
    sibling_task = {
        "id": "DEMO-001",
        "subtasks": [
            heat_style,
            {
                "id": "DEMO-001.2",
                "targetArtifacts": {"test": "tests/test_demo.py"},
                "verification": ["pytest tests/ -k toggle_api -v"],
            },
        ],
    }
    assert complete_subtask_verification(sibling_task, "DEMO-001.1", heat_style) is None


def test_completer_write_back_touches_only_verification() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        plans_dir = Path(temp_dir) / "plans"
        original_requirements = _tasks.REQUIREMENTS_DIR
        original_plans = config.PLANS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        config.PLANS_DIR = plans_dir
        try:
            task = {
                "id": "DEMO-002",
                "summary": "Keep business text",
                "acceptanceCriteria": "Must stay",
                "subtasks": [
                    {
                        "id": "DEMO-002.1",
                        "title": "First",
                        "scope": ["Add app/domain/example.py"],
                        "acceptanceCriteria": "unused",
                        "verification": ["pytest tests/test_demo.py -q"],
                    }
                ],
            }
            _write_task(requirements_dir, task)
            persist_verification_repair(
                "DEMO-002",
                "DEMO-002.1",
                ["pytest tests/test_demo.py -k DEMO_002_1 -q"],
            )
            saved = json.loads((requirements_dir / "DEMO-002.json").read_text(encoding="utf-8"))
            assert saved["acceptanceCriteria"] == "Must stay"
            assert saved["summary"] == "Keep business text"
            assert saved["subtasks"][0]["verification"] == [
                "pytest tests/test_demo.py -k DEMO_002_1 -q"
            ]
            assert saved["subtasks"][0]["title"] == "First"
        finally:
            _tasks.REQUIREMENTS_DIR = original_requirements
            config.PLANS_DIR = original_plans
