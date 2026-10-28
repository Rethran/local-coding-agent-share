from __future__ import annotations

import json
from pathlib import Path

from orchestrator.context_pack import build_context_pack

AGENT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_DIR = AGENT_ROOT / "requirements"


def test_context_pack_includes_legacy_slices_and_target_snapshot() -> None:
    task_id = "HEAT-003"
    subtask_id = "HEAT-003.2"
    pack = build_context_pack(task_id, subtask_id)

    assert pack["taskId"] == task_id
    assert pack["subtaskId"] == subtask_id
    assert pack["subtaskContext"]["subtaskId"] == subtask_id
    assert pack["subtaskContext"]["legacySlices"]
    assert pack["verificationChecklist"]["taskId"] == task_id
    assert "dependencyStatus" in pack
    snapshot = pack["targetFileSnapshot"]
    assert snapshot["path"] == "app/domain/prices.py"
    assert "explorationPolicy" in pack
    assert "referenceSnippets" in pack


def test_context_pack_heat_009_2_includes_related_snippets_and_policy() -> None:
    pack = build_context_pack("HEAT-009", "HEAT-009.2")

    assert pack["referenceSnippets"]
    paths = {item["path"] for item in pack["referenceSnippets"]}
    assert "app/services/schedule_store.py" in paths
    assert "app/api/schedule.py" in paths
    schedule_store_snippet = next(
        item for item in pack["referenceSnippets"]
        if item["path"] == "app/services/schedule_store.py"
    )
    schedule_store_source = schedule_store_snippet["source"]
    assert "import threading" in schedule_store_source
    assert "build_schedule" in schedule_store_source
    assert "class ScheduleStore" in schedule_store_source
    policy = pack["explorationPolicy"]
    assert policy["preflightSatisfied"] is True
    assert "get_subtask_context" in policy["skipTools"]
    assert pack["targetFileSnapshot"].get("excerpt")
    assert "repoContext" in pack
    assert isinstance(pack["repoContext"].get("matches"), list)
    repo_names = {item["name"] for item in pack["repoContext"]["matches"]}
    assert "toggle_slot" in repo_names
    assert "get_repo_context" in policy["skipTools"]


def test_context_pack_uses_pinned_subtask_not_active_state() -> None:
    pack = build_context_pack("HEAT-003", "HEAT-003.1")
    assert pack["subtaskContext"]["title"] == "Create prices module skeleton only"
    assert pack["subtaskContext"]["legacySlices"] == []


def test_context_pack_directory_target_expands_to_files() -> None:
    pack = build_context_pack("HEAT-027", "HEAT-027.3")

    paths = {item["path"] for item in pack["referenceSnippets"]}
    assert "deploy/systemd/heating-controller.service" in paths
    unit_snippet = next(
        item
        for item in pack["referenceSnippets"]
        if item["path"] == "deploy/systemd/heating-controller.service"
    )
    assert "Restart=" in unit_snippet["source"]
    assert "python -m app.main" in unit_snippet["source"]

    snapshot = pack["targetFileSnapshot"]
    assert snapshot["exists"] is True
    assert snapshot.get("isDirectory") is True
    assert "deploy/systemd/heating-controller.service" in snapshot.get("files", [])
    assert "Restart=" in snapshot.get("excerpt", "")
