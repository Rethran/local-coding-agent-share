"""Deterministic dependency inference from generated artifact references."""

from __future__ import annotations

from typing import Any


def normalize_artifact_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./")


def referenced_artifacts(task: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for subtask in task.get("subtasks") or []:
        if not isinstance(subtask, dict):
            continue
        targets = subtask.get("targetArtifacts") or {}
        if isinstance(targets, dict):
            for key, value in targets.items():
                if key != "function" and normalize_artifact_path(value):
                    paths.add(normalize_artifact_path(value))
        for related in subtask.get("relatedArtifacts") or []:
            if isinstance(related, dict) and normalize_artifact_path(related.get("path")):
                paths.add(normalize_artifact_path(related["path"]))
    return paths


def produced_artifacts(task: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for subtask in task.get("subtasks") or []:
        if not isinstance(subtask, dict):
            continue
        targets = subtask.get("targetArtifacts") or {}
        if not isinstance(targets, dict):
            continue
        for key, value in targets.items():
            if key != "function" and normalize_artifact_path(value):
                paths.add(normalize_artifact_path(value))
    return paths


def build_artifact_frontier(accepted: list[dict[str, Any]]) -> dict[str, str]:
    frontier: dict[str, str] = {}
    for entry in accepted:
        task = entry.get("task") if isinstance(entry, dict) else None
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("id") or "")
        for path in produced_artifacts(task):
            frontier[path] = task_id
    return frontier


def wire_dependencies(
    task: dict[str, Any],
    accepted: list[dict[str, Any]],
) -> dict[str, Any]:
    """Set dependencies to prior latest producers referenced by this task."""
    frontier = build_artifact_frontier(accepted)
    dependencies = {
        frontier[path]
        for path in referenced_artifacts(task)
        if path in frontier
    }
    ordered_ids = [
        str(entry.get("task", {}).get("id") or "")
        for entry in accepted
        if isinstance(entry, dict)
    ]
    task["dependsOn"] = [task_id for task_id in ordered_ids if task_id in dependencies]
    return task
