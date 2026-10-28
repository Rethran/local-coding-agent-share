"""Named prompt clauses with file-based overrides.

Every sentence block in the worker prompts has a clause name. A prompt variant
is a JSON file under ``orchestrator/prompt_variants/<name>.json`` that maps
clause names to replacement text. ``baseline.json`` is empty, so the default
variant renders exactly the hardcoded text.

Select a variant with ``LOCAL_AGENT_PROMPT_VARIANT=<name>``. The override text
may reference ``{subtask_id}``, ``{legacy_paths}``, ``{project_name}``,
``{task_prefix}``, ``{implementation_prefixes}``, ``{task_id}``; unknown
placeholders are left untouched.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

VARIANTS_DIR = Path(__file__).resolve().parent / "prompt_variants"
DEFAULT_VARIANT = "baseline"


class _SafeFormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def active_variant_name() -> str:
    return os.environ.get("LOCAL_AGENT_PROMPT_VARIANT", "").strip() or DEFAULT_VARIANT


def variant_path(name: str) -> Path:
    return VARIANTS_DIR / f"{name}.json"


@lru_cache(maxsize=8)
def load_variant(name: str) -> dict[str, str]:
    path = variant_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt variant '{name}' not found at {path}. "
            f"Available: {sorted(p.stem for p in VARIANTS_DIR.glob('*.json'))}"
        )
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Prompt variant {path} must be a JSON object")
    clauses = data.get("clauses", data)
    if not isinstance(clauses, dict):
        raise ValueError(f"Prompt variant {path}: 'clauses' must be an object")
    return {str(k): str(v) for k, v in clauses.items() if k != "description"}


def clause(name: str, default: str, **variables: Any) -> str:
    """Return the override for ``name`` in the active variant, else ``default``."""
    overrides = load_variant(active_variant_name())
    if name not in overrides:
        return default
    return overrides[name].format_map(_SafeFormatDict(variables))


def list_variants() -> list[str]:
    return sorted(p.stem for p in VARIANTS_DIR.glob("*.json"))
