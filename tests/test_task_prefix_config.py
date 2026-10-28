from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from tools import _tasks


def _write_task(requirements_dir: Path, task_id: str) -> None:
    (requirements_dir / f"{task_id}.json").write_text(
        json.dumps(
            {
                "id": task_id,
                "summary": f"Task {task_id}",
                "dependsOn": [],
                "acceptanceCriteria": "Works",
                "scope": ["Create src/example.py"],
                "verification": ["python -m pytest tests -q"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def test_graph_task_prefix_parses_and_loads(monkeypatch) -> None:
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        _write_task(requirements_dir, "GRAPH-001")
        _write_task(requirements_dir, "GRAPH-002")
        _write_task(requirements_dir, "HEAT-001")

        monkeypatch.setattr(_tasks, "REQUIREMENTS_DIR", requirements_dir)
        monkeypatch.setattr(_tasks, "TASK_PREFIX", "GRAPH")

        parsed = _tasks.parse_heat_ref("GRAPH-001.2")
        assert parsed["taskId"] == "GRAPH-001"
        assert parsed["subtaskId"] == "GRAPH-001.2"
        assert _tasks.parse_task_number("GRAPH-002") == 2

        loaded = _tasks.load_task_definitions()
        assert sorted(loaded) == ["GRAPH-001", "GRAPH-002"]
        assert list(_tasks.filter_task_range(loaded, "GRAPH-001", "GRAPH-001")) == [
            "GRAPH-001"
        ]
