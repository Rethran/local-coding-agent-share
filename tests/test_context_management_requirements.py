"""CTX packets stay valid for the local-agent runner."""

from __future__ import annotations

import json
import re
from pathlib import Path

from requirements_orchestrator.validation import validate_task

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements" / "context-management"
PROFILE = ROOT / "projects" / "context-management.json"
PREFIXES = ["src/", "tests/", "static/"]
REQUIRED_IDS = [
    "FR-C1-01", "FR-C1-02", "FR-C1-03", "FR-C1-04", "FR-C1-05", "FR-C1-06", "FR-C1-08",
    "FR-C2-01", "FR-C2-02", "FR-C2-03", "FR-C2-04", "FR-C2-05", "FR-C2-06", "FR-C2-07",
    "FR-C3-01", "FR-C3-02", "FR-C3-03", "FR-C3-04", "FR-C3-05", "FR-C3-06", "FR-C3-07", "FR-C3-08",
    "FR-C4-01", "FR-C4-02", "FR-C4-03", "FR-C4-04", "FR-C4-05", "FR-C4-06",
    "FR-C5-01", "FR-C5-02", "FR-C5-03", "FR-C5-04",
    "FR-C6-01", "FR-C6-02", "FR-C6-03",
    "NFR-01", "NFR-02", "NFR-04", "NFR-05",
    "FR-C1-07", "FR-C1-09", "FR-C1-10",
    "FR-C4-07", "FR-C4-08", "FR-C4-09",
    *[f"FR-C7-{index:02d}" for index in range(1, 19)],
    *[f"SEC-{index:02d}" for index in range(1, 32)],
    *[f"DEP-{index:02d}" for index in range(1, 12)],
    *[f"OWN-{index:02d}" for index in range(1, 6)],
    "NFR-06",
    "T1.", "T3.", "T5.", "T6.", "T7.", "T8.", "T9.", "T10.",
    "T8-1", "T8-2", "T8-3", "T8-4", "T8-5", "T8-6",
    "P2", "P4", "P7", "P8", "P9",
    "OD-5", "OD-6", "OD-7", "OD-8",
]
TASK_FILE = re.compile(r"CTX-\d{3}\.json$")


def test_context_management_packets_pass_validation() -> None:
    paths = sorted(path for path in REQUIREMENTS.glob("CTX-*.json") if TASK_FILE.fullmatch(path.name))
    assert [path.name for path in paths] == [f"CTX-{index:03d}.json" for index in range(1, 70)]

    prior: set[str] = set()
    seen_tests: set[str] = set()
    blob: list[str] = []
    for index, path in enumerate(paths, start=1):
        task = json.loads(path.read_text(encoding="utf-8"))
        blob.append(path.read_text(encoding="utf-8"))
        result = validate_task(
            task,
            prefix="CTX",
            task_number=index,
            implementation_prefixes=PREFIXES,
            acceptance_ids={"ACCEPT-001"},
            covers=["ACCEPT-001"],
            prior_task_ids=set(prior),
        )
        assert result.valid, result.errors
        assert task["dependsOn"] == sorted(task["dependsOn"])
        for subtask in task["subtasks"]:
            assert subtask["readLegacy"] is False
            assert "legacySymbols" not in subtask
            test_path = subtask["targetArtifacts"]["test"]
            assert test_path not in seen_tests
            seen_tests.add(test_path)
            command = subtask["verification"][0]
            assert test_path in command
            assert " -k CTX_" in command
        prior.add(task["id"])

    text = "\n".join(blob)
    missing = [item for item in REQUIRED_IDS if item not in text]
    assert missing == []


def test_context_management_profile_points_at_ctx_packets() -> None:
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["taskPrefix"] == "CTX"
    assert profile["defaultModel"] == "night-runner:latest"
    assert profile["defaultFromId"] == "CTX-001"
    assert profile["defaultToId"] == "CTX-004"
    assert profile["autoPush"] is False
    assert profile["implementationPrefixes"] == PREFIXES
    assert profile["legacyPrefixes"] == []
    assert Path(profile["requirementsDir"]) == REQUIREMENTS.resolve()
    assert profile["workspaceRoot"].endswith("context-management")
