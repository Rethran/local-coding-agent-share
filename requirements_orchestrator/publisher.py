"""All-or-nothing publication into the existing orchestrator's input folders."""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from requirements_orchestrator.state import GenerationState
from requirements_orchestrator.validation import load_task_schema


class PublishError(RuntimeError):
    """The generated project package cannot be published safely."""


@dataclass(frozen=True)
class PublishResult:
    profile_name: str
    profile_path: Path
    requirements_dir: Path
    first_task_id: str
    last_task_id: str


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise PublishError("Project name does not produce a usable profile slug.")
    return slug


def _profile_payload(state: GenerationState, requirements_dir: Path) -> dict[str, object]:
    fields = state.fields
    prefix = str(fields["taskPrefix"])
    last_id = str(state.accepted[-1]["task"]["id"])
    implementation_prefixes = [str(item) for item in fields["implementationPrefixes"]]
    return {
        "name": str(fields["projectName"]),
        "projectName": str(fields["projectName"]),
        "workspaceRoot": str(fields["workspaceRoot"]),
        "requirementsDir": str(requirements_dir.resolve()),
        "taskPrefix": prefix,
        "defaultModel": str(fields.get("model") or "qwen3:8b"),
        "defaultFromId": f"{prefix}-001",
        "defaultToId": last_id,
        "autoPush": False,
        "implementationPrefixes": implementation_prefixes,
        "readFileAllowedPrefixes": implementation_prefixes,
        "readFileAllowedSuffixes": [
            ".py", ".robot", ".resource", ".json", ".md", ".txt", ".toml",
            ".yaml", ".yml", ".js", ".jsx", ".ts", ".tsx", ".html", ".css",
            ".go", ".rs", ".java", ".cs",
        ],
        "legacyPrefixes": [],
        "promptGuidance": (
            f"Implement exactly the active {prefix} subtask. Follow its constraints, "
            "target artifacts, and focused verification command."
        ),
        "taskPromptGuidance": [],
        "knowledgePacks": [],
    }


def publish(state: GenerationState, *, repo_root: Path) -> PublishResult:
    if state.stage != "READY_TO_PUBLISH":
        raise PublishError(f"State must be READY_TO_PUBLISH, not {state.stage}.")
    if not state.accepted:
        raise PublishError("No accepted tasks to publish.")

    workspace = Path(str(state.fields.get("workspaceRoot") or ""))
    if not workspace.is_absolute() or not workspace.is_dir():
        raise PublishError(f"Target workspace must already exist and be a directory: {workspace}")

    root = Path(repo_root).resolve()
    profile_name = slugify(str(state.fields["projectName"]))
    requirements_parent = root / "requirements"
    profiles_dir = root / "projects"
    destination = requirements_parent / profile_name
    profile_path = profiles_dir / f"{profile_name}.json"
    if destination.exists():
        raise PublishError(f"Requirements destination already exists: {destination}")
    if profile_path.exists():
        raise PublishError(f"Project profile already exists: {profile_path}")

    token = uuid.uuid4().hex
    staged_requirements = requirements_parent / f".{profile_name}.{token}.tmp"
    staged_profile = profiles_dir / f".{profile_name}.{token}.tmp.json"
    requirements_parent.mkdir(parents=True, exist_ok=True)
    profiles_dir.mkdir(parents=True, exist_ok=True)
    schema_validator = Draft202012Validator(load_task_schema())

    try:
        staged_requirements.mkdir()
        for entry in state.accepted:
            task = entry.get("task")
            if not isinstance(task, dict):
                raise PublishError("Accepted entry has no task object.")
            errors = list(schema_validator.iter_errors(task))
            if errors:
                raise PublishError(f"Final task schema validation failed: {errors[0].message}")
            task_id = str(task["id"])
            task_path = staged_requirements / f"{task_id}.json"
            task_path.write_text(
                json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

        profile = _profile_payload(state, destination)
        staged_profile.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(staged_requirements, destination)
        try:
            os.replace(staged_profile, profile_path)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    except Exception:
        shutil.rmtree(staged_requirements, ignore_errors=True)
        staged_profile.unlink(missing_ok=True)
        raise

    state.stage = "PUBLISHED"
    state.published_profile = profile_name
    return PublishResult(
        profile_name=profile_name,
        profile_path=profile_path,
        requirements_dir=destination,
        first_task_id=str(state.accepted[0]["task"]["id"]),
        last_task_id=str(state.accepted[-1]["task"]["id"]),
    )
