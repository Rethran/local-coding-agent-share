"""Compatibility checks for generated local-agent task packets."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).parent / "schemas" / "task.schema.json"
COMMAND_RUNNERS = {
    "python", "python3", "py", "pytest", "npm", "npx", "pnpm", "yarn",
    "cargo", "go", "unittest", "node", "uv", "poetry", "dotnet", "mvn",
    "gradle", "make",
}
NEGATIVE_RE = re.compile(r"\b(do not|must not|never|avoid|without|only)\b", re.IGNORECASE)
FOCUS_MARKERS = (" -k ", "::", "--filter", "--testpathpattern", "--test-name-pattern")


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    errors: list[str]


def load_task_schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def proposal_task_schema() -> dict[str, Any]:
    """Return the publish schema minus deterministic dependency wiring."""
    schema = copy.deepcopy(load_task_schema())
    schema["required"].remove("dependsOn")
    schema["properties"].pop("dependsOn")
    return schema


def decomposition_response_schema() -> dict[str, Any]:
    task_schema = proposal_task_schema()
    definitions = task_schema.pop("$defs")
    task_schema.pop("$schema", None)
    task_schema.pop("$id", None)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["status", "task", "covers", "reason"],
        "properties": {
            "status": {"type": "string", "enum": ["task", "done"]},
            "task": {"anyOf": [task_schema, {"type": "null"}]},
            "covers": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "reason": {"type": "string"},
        },
        "$defs": definitions,
    }


def _normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/").lstrip("./")


def _valid_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


def _matches_prefix(path: str, prefixes: list[str]) -> bool:
    return any(path.startswith(_normalized_path(prefix).rstrip("/") + "/") for prefix in prefixes)


def _schema_errors(task: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(load_task_schema())
    errors = sorted(validator.iter_errors(task), key=lambda item: list(item.absolute_path))
    rendered: list[str] = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<task>"
        rendered.append(f"{location}: {error.message}")
    return rendered


def _commands(task: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for subtask in task.get("subtasks") or []:
        if not isinstance(subtask, dict):
            continue
        for command in subtask.get("verification") or []:
            rows.append((str(command).strip(), subtask))
    return rows


def validate_task(
    task: dict[str, Any],
    *,
    prefix: str,
    task_number: int,
    implementation_prefixes: list[str],
    acceptance_ids: set[str],
    covers: list[str],
    prior_task_ids: set[str],
) -> ValidationResult:
    errors = _schema_errors(task)
    expected_id = f"{prefix}-{task_number:03d}"
    if task.get("id") != expected_id:
        errors.append(f"id: expected {expected_id}")
    if task.get("type") != "task":
        errors.append("type: must be 'task'")

    expected_subtask_ids = [
        f"{expected_id}.{index}"
        for index in range(1, len(task.get("subtasks") or []) + 1)
    ]
    actual_subtask_ids = [
        str(item.get("id") or "") for item in task.get("subtasks") or [] if isinstance(item, dict)
    ]
    if actual_subtask_ids != expected_subtask_ids:
        errors.append(f"subtasks: expected sequential ids {expected_subtask_ids}")

    dependencies = task.get("dependsOn") or []
    unknown_dependencies = [item for item in dependencies if item not in prior_task_ids]
    if unknown_dependencies:
        errors.append(f"dependsOn: unknown or forward dependencies {unknown_dependencies}")
    if expected_id in dependencies:
        errors.append("dependsOn: task cannot depend on itself")

    if not covers:
        errors.append("covers: task must cover at least one acceptance signal")
    unknown_coverage = sorted(set(covers) - acceptance_ids)
    if unknown_coverage:
        errors.append(f"covers: unknown acceptance signal ids {unknown_coverage}")

    constraints = [str(item) for item in task.get("constraints") or []]
    if constraints and not any(NEGATIVE_RE.search(item) for item in constraints):
        errors.append("constraints: include at least one explicit negative constraint")

    test_owners: dict[str, str] = {}
    verification_by_test: dict[str, list[str]] = {}
    for subtask in task.get("subtasks") or []:
        if not isinstance(subtask, dict):
            continue
        subtask_id = str(subtask.get("id") or "<unknown>")
        targets = subtask.get("targetArtifacts") or {}
        if isinstance(targets, dict):
            for kind, raw_path in targets.items():
                if kind == "function":
                    continue
                path = _normalized_path(raw_path)
                if not _valid_relative_path(path):
                    errors.append(f"{subtask_id}.targetArtifacts.{kind}: must be a safe relative path")
                elif not _matches_prefix(path, implementation_prefixes):
                    errors.append(
                        f"{subtask_id}.targetArtifacts.{kind}: outside implementation prefixes "
                        f"{implementation_prefixes}"
                    )
            test_path = _normalized_path(targets.get("test"))
            if test_path:
                previous = test_owners.get(test_path)
                commands = [str(item).strip() for item in subtask.get("verification") or []]
                prior_commands = verification_by_test.get(test_path) or []
                if previous and (
                    prior_commands == commands
                    or not all(any(marker in f" {cmd.lower()} " for marker in FOCUS_MARKERS) for cmd in prior_commands + commands)
                ):
                    errors.append(
                        f"{subtask_id}.verification: shared test {test_path} with {previous} "
                        "requires distinct focused selectors"
                    )
                test_owners[test_path] = subtask_id
                verification_by_test[test_path] = commands

        read_legacy = subtask.get("readLegacy")
        legacy_symbols = subtask.get("legacySymbols") or []
        if read_legacy is True and not legacy_symbols:
            errors.append(f"{subtask_id}.legacySymbols: required when readLegacy is true")
        if read_legacy is False and legacy_symbols:
            errors.append(f"{subtask_id}.legacySymbols: forbidden when readLegacy is false")

    for command, subtask in _commands(task):
        subtask_id = str(subtask.get("id") or "<unknown>")
        first = command.split(maxsplit=1)[0].lower() if command else ""
        if first not in COMMAND_RUNNERS:
            errors.append(f"{subtask_id}.verification: unsupported command runner in {command!r}")
            continue
        targets = subtask.get("targetArtifacts") or {}
        test_path = _normalized_path(targets.get("test")) if isinstance(targets, dict) else ""
        lowered = f" {command.lower()} "
        focused = (
            bool(test_path and (test_path.lower() in lowered or PurePosixPath(test_path).name.lower() in lowered))
            or any(marker in lowered for marker in FOCUS_MARKERS)
        )
        if not focused:
            errors.append(
                f"{subtask_id}.verification: must target its test artifact or include a test selector"
            )

    return ValidationResult(valid=not errors, errors=errors)
