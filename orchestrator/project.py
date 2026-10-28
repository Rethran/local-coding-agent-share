from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AGENT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKSPACE_ROOT = AGENT_ROOT.parent
PROFILES_DIR = AGENT_ROOT / "projects"


@dataclass(frozen=True)
class ProjectProfile:
    name: str
    workspace_root: Path
    requirements_dir: Path
    task_prefix: str
    default_model: str
    default_from_id: str
    default_to_id: str
    prompt_guidance: str
    task_prompt_guidance: tuple[dict[str, str], ...]
    knowledge_packs: tuple[dict[str, str], ...]
    implementation_prefixes: tuple[str, ...]
    read_file_allowed_prefixes: tuple[str, ...]
    read_file_allowed_suffixes: tuple[str, ...]
    legacy_prefixes: tuple[str, ...]
    auto_push: bool


def _as_tuple(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item).replace("\\", "/") for item in value if str(item).strip())
    return default


def _as_object_tuple(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        return ()
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append({str(key): str(val) for key, val in item.items() if val is not None})
    return tuple(rows)


def _resolve_profile_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.suffix.lower() == ".json" or candidate.is_absolute():
        return candidate
    return PROFILES_DIR / f"{value}.json"


def _read_profile_data() -> dict[str, Any]:
    profile = os.environ.get("LOCAL_AGENT_PROFILE", "").strip()
    if not profile:
        return {}
    path = _resolve_profile_path(profile)
    if not path.exists():
        raise FileNotFoundError(f"Local agent profile not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: Any, default: Path) -> Path:
    if not value:
        return default.resolve()
    return Path(str(value)).expanduser().resolve()


def load_project_profile() -> ProjectProfile:
    data = _read_profile_data()
    name = str(
        os.environ.get("LOCAL_AGENT_PROJECT_NAME")
        or data.get("projectName")
        or data.get("name")
        or "HEAT"
    )
    workspace_root = _resolve_path(
        os.environ.get("LOCAL_AGENT_WORKSPACE_ROOT") or data.get("workspaceRoot"),
        DEFAULT_WORKSPACE_ROOT,
    )
    requirements_dir = _resolve_path(
        os.environ.get("LOCAL_AGENT_REQUIREMENTS_DIR") or data.get("requirementsDir"),
        AGENT_ROOT / "requirements",
    )
    task_prefix = str(
        os.environ.get("LOCAL_AGENT_TASK_PREFIX") or data.get("taskPrefix") or "HEAT"
    ).strip()
    default_model = str(
        os.environ.get("LOCAL_AGENT_MODEL") or data.get("defaultModel") or "qwen3:8b"
    ).strip()
    default_from = str(
        os.environ.get("LOCAL_AGENT_DEFAULT_FROM")
        or data.get("defaultFromId")
        or f"{task_prefix}-001"
    )
    default_to = str(
        os.environ.get("LOCAL_AGENT_DEFAULT_TO")
        or data.get("defaultToId")
        or f"{task_prefix}-005"
    )
    default_allowed_prefixes = (
        "app/",
        "tests/",
        "local-coding-agent/requirements/",
        "local-coding-agent/tests/",
    )
    default_suffixes = (
        ".xml",
        ".json",
        ".md",
        ".txt",
        ".csv",
        ".robot",
        ".resource",
        ".toml",
        ".yaml",
        ".yml",
        ".ini",
        ".cfg",
    )
    implementation_prefixes = _as_tuple(
        data.get("implementationPrefixes"),
        ("app/", "tests/"),
    )
    read_prefixes = _as_tuple(
        data.get("readFileAllowedPrefixes"),
        default_allowed_prefixes,
    )
    read_suffixes = _as_tuple(
        data.get("readFileAllowedSuffixes"),
        default_suffixes,
    )
    legacy_prefixes = _as_tuple(data.get("legacyPrefixes"), ("old_entsoes/",))
    auto_push = bool(data.get("autoPush", True))
    return ProjectProfile(
        name=name,
        workspace_root=workspace_root,
        requirements_dir=requirements_dir,
        task_prefix=task_prefix,
        default_model=default_model,
        default_from_id=default_from,
        default_to_id=default_to,
        prompt_guidance=str(data.get("promptGuidance") or ""),
        task_prompt_guidance=_as_object_tuple(data.get("taskPromptGuidance")),
        knowledge_packs=_as_object_tuple(data.get("knowledgePacks")),
        implementation_prefixes=implementation_prefixes,
        read_file_allowed_prefixes=read_prefixes,
        read_file_allowed_suffixes=read_suffixes,
        legacy_prefixes=legacy_prefixes,
        auto_push=auto_push,
    )


PROJECT_PROFILE = load_project_profile()
