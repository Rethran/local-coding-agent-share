from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator.plan_merge import merge_subtask_definition
from orchestrator.spec_readiness import evaluate_spec_readiness
from tools import _tasks


def _write_task(requirements_dir: Path, task: dict) -> None:
    task_id = task["id"]
    (requirements_dir / f"{task_id}.json").write_text(
        json.dumps(task, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_derived_subtask_fails_golden_rules() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-900",
                    "summary": "Derived subtask test",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "scope": ["Create app/domain/example.py"],
                    "verification": ["pytest tests/ -v"],
                },
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-900", "HEAT-900.1")
            assert subtask is not None
            context_pack = {
                "subtaskContext": {"legacySlices": []},
            }
            result = evaluate_spec_readiness(
                "HEAT-900",
                "HEAT-900.1",
                context_pack,
                subtask,
            )
            assert result.ready is False
            assert "GR-DERIVED-001" in result.rule_ids or "GR-SUB-002" in result.rule_ids
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir


def test_missing_legacy_symbols_fails_for_read_legacy_subtask() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-901",
                    "summary": "Legacy subtask test",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "verification": ["pytest tests/ -v"],
                    "subtasks": [
                        {
                            "id": "HEAT-901.1",
                            "title": "Legacy parser",
                            "summary": "Parse legacy",
                            "readLegacy": True,
                            "scope": ["Implement app/domain/example.py"],
                            "verification": ["pytest tests/ -v"],
                        }
                    ],
                },
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-901", "HEAT-901.1")
            assert subtask is not None
            result = evaluate_spec_readiness(
                "HEAT-901",
                "HEAT-901.1",
                {"subtaskContext": {"legacySlices": []}},
                subtask,
            )
            assert result.ready is False
            assert "GR-LEGACY-001" in result.rule_ids
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir


def test_sidecar_enrichments_satisfy_derived_subtask_rules() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-902",
                    "summary": "Planner enrichment test",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "scope": ["Create app/domain/example.py"],
                    "verification": ["pytest tests/ -v"],
                },
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-902", "HEAT-902.1")
            assert subtask is not None
            merged = merge_subtask_definition(
                subtask,
                {
                    "verification": ["python -c \"import app.domain.example\""],
                    "targetArtifacts": {
                        "module": "app/domain/example.py",
                        "function": "example",
                    },
                },
            )
            result = evaluate_spec_readiness(
                "HEAT-902",
                "HEAT-902.1",
                {"subtaskContext": {"legacySlices": []}},
                merged,
            )
            assert result.ready is True
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir


def _two_subtask_task(task_id: str, verification: list[str], *, selector: bool = False) -> dict:
    first_verify = ["pytest tests/test_example.py -k first -v"] if selector else verification
    second_verify = ["pytest tests/test_example.py -k second -v"] if selector else verification
    return {
        "id": task_id,
        "summary": "Shared verification test",
        "acceptanceCriteria": "Works",
        "dependsOn": [],
        "verification": ["pytest tests/ -v"],
        "subtasks": [
            {
                "id": f"{task_id}.1",
                "title": "First slice",
                "summary": "First slice",
                "readLegacy": False,
                "scope": ["Add app/domain/example.py"],
                "targetArtifacts": {
                    "module": "app/domain/example.py",
                    "test": "tests/test_example.py",
                },
                "verification": first_verify,
            },
            {
                "id": f"{task_id}.2",
                "title": "Second slice",
                "summary": "Second slice",
                "readLegacy": False,
                "scope": ["Add app/domain/example.py"],
                "targetArtifacts": {
                    "module": "app/domain/example.py",
                    "test": "tests/test_example.py",
                },
                "verification": second_verify,
            },
        ],
    }


def test_identical_sibling_verification_fails_gr_sub_004() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                _two_subtask_task("HEAT-910", ["pytest tests/test_example.py -q"]),
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-910", "HEAT-910.1")
            assert subtask is not None
            result = evaluate_spec_readiness(
                "HEAT-910",
                "HEAT-910.1",
                {"subtaskContext": {"legacySlices": []}},
                subtask,
            )
            assert result.ready is False
            assert "GR-SUB-004" in result.rule_ids
            assert "GR-SUB-006" in result.rule_ids
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir


def test_unique_k_selectors_pass_shared_test_file_rules() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                _two_subtask_task("HEAT-911", [], selector=True),
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-911", "HEAT-911.1")
            assert subtask is not None
            result = evaluate_spec_readiness(
                "HEAT-911",
                "HEAT-911.1",
                {"subtaskContext": {"legacySlices": []}},
                subtask,
            )
            assert "GR-SUB-004" not in result.rule_ids
            assert "GR-SUB-006" not in result.rule_ids
            assert result.ready is True
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir


def test_prose_only_subtask_verification_fails_gr_sub_005() -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original_dir = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            _write_task(
                requirements_dir,
                {
                    "id": "HEAT-912",
                    "summary": "Prose verification",
                    "acceptanceCriteria": "Works",
                    "dependsOn": [],
                    "subtasks": [
                        {
                            "id": "HEAT-912.1",
                            "title": "Manual check",
                            "summary": "Manual check",
                            "readLegacy": False,
                            "scope": ["Add app/domain/example.py"],
                            "targetArtifacts": {"module": "app/domain/example.py"},
                            "verification": ["Select one slot.", "Toggle it."],
                        }
                    ],
                },
            )
            subtask = _tasks.get_subtask_spec_definition("HEAT-912", "HEAT-912.1")
            assert subtask is not None
            result = evaluate_spec_readiness(
                "HEAT-912",
                "HEAT-912.1",
                {"subtaskContext": {"legacySlices": []}},
                subtask,
            )
            assert result.ready is False
            assert "GR-SUB-005" in result.rule_ids
        finally:
            _tasks.REQUIREMENTS_DIR = original_dir

