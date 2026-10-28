"""Disk-backed handoff records for fragile local-model tool chains."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools._common import WORKSPACE_ROOT

_AGENT_ROOT = Path(__file__).resolve().parent.parent


def runtime_root() -> Path:
    """Directory that owns agent runtime artifacts (handoffs, recovery).

    Defaults to the agent repository root. ``LOCAL_AGENT_RUNTIME_DIR`` overrides
    it so that eval runs can give every sample a fresh, isolated state.
    """
    override = os.environ.get("LOCAL_AGENT_RUNTIME_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return _AGENT_ROOT


HANDOFFS_DIR = runtime_root() / "handoffs"
CONSUMED_DIR = HANDOFFS_DIR / ".consumed"

SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_safe_id(value: str, field_name: str) -> str:
    if not value or not SAFE_ID_PATTERN.match(value):
        raise ValueError(f"Invalid {field_name}: {value}")
    return value


def slug(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return safe or "handoff"


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
    except ValueError:
        return str(resolved).replace("\\", "/")


def parse_json_object(value: str, field_name: str) -> dict[str, Any]:
    if not value:
        return {}

    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return parsed


def parse_json_value(value: str) -> Any:
    if not value:
        return {}
    return json.loads(value)


def chain_dir(chain_id: str) -> Path:
    return HANDOFFS_DIR / ensure_safe_id(chain_id, "chain_id")


def consumed_chain_dir(chain_id: str) -> Path:
    return CONSUMED_DIR / ensure_safe_id(chain_id, "chain_id")


def handoff_path_from_id(handoff_id: str) -> Path:
    ensure_safe_id(handoff_id, "handoff_id")

    active_matches = sorted(HANDOFFS_DIR.glob(f"*/{handoff_id}.json"))
    if active_matches:
        return active_matches[0]

    consumed_matches = sorted(CONSUMED_DIR.glob(f"*/{handoff_id}.json"))
    if consumed_matches:
        return consumed_matches[0]

    raise ValueError(f"Handoff not found: {handoff_id}")


def handoff_path_from_user_input(
    handoff_id: str = "",
    handoff_path: str = "",
) -> Path:
    if handoff_path:
        candidate = (WORKSPACE_ROOT / handoff_path).resolve()
        if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
            raise ValueError(f"Path outside workspace: {handoff_path}")
        if not candidate.exists():
            raise ValueError(f"Handoff path not found: {handoff_path}")
        return candidate

    if handoff_id:
        return handoff_path_from_id(handoff_id)

    raise ValueError("Provide handoff_id or handoff_path")


def read_handoff_record(
    handoff_id: str = "",
    handoff_path: str = "",
) -> dict[str, Any]:
    path = handoff_path_from_user_input(handoff_id, handoff_path)
    return json.loads(path.read_text(encoding="utf-8"))


def next_step_index(chain_id: str) -> int:
    directory = chain_dir(chain_id)
    consumed_directory = consumed_chain_dir(chain_id)
    max_index = 0

    for root in (directory, consumed_directory):
        if not root.exists():
            continue
        for path in root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            max_index = max(max_index, int(record.get("stepIndex", 0)))

    return max_index + 1


def compact_handoff(record: dict[str, Any], path: Path) -> dict[str, Any]:
    return {
        "handoffId": record["handoffId"],
        "handoffPath": relative_path(path),
        "chainId": record["chainId"],
        "stepIndex": record["stepIndex"],
        "summary": record.get("summary", ""),
        "nextAction": record.get("nextAction"),
        "copyPastePrompt": record.get("copyPastePrompt", ""),
    }


def set_copy_paste_prompt(
    handoff_bundle: dict[str, Any],
    copy_paste_prompt: str,
) -> dict[str, Any]:
    record = handoff_bundle["record"]
    path = handoff_bundle["path"]
    record["copyPastePrompt"] = copy_paste_prompt
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    handoff_bundle["compact"] = compact_handoff(record, path)
    return handoff_bundle


def record_handoff(
    *,
    task_id: str,
    tool: str,
    input_data: dict[str, Any],
    result: Any,
    summary: str,
    next_action: dict[str, Any] | None = None,
    previous_handoff_id: str = "",
    chain_id: str = "",
    copy_paste_prompt: str = "",
) -> dict[str, Any]:
    if previous_handoff_id:
        previous_record = read_handoff_record(handoff_id=previous_handoff_id)
        chain_id = chain_id or previous_record["chainId"]
        step_index = int(previous_record["stepIndex"]) + 1
    else:
        chain_id = chain_id or task_id
        step_index = next_step_index(chain_id)

    ensure_safe_id(chain_id, "chain_id")
    handoff_id = f"{chain_id}-{step_index:04d}"
    directory = chain_dir(chain_id)
    directory.mkdir(parents=True, exist_ok=True)

    record = {
        "handoffId": handoff_id,
        "previousHandoffId": previous_handoff_id or None,
        "stepIndex": step_index,
        "chainId": chain_id,
        "createdAt": utc_now_iso(),
        "taskId": task_id,
        "tool": tool,
        "input": input_data,
        "result": result,
        "summary": summary,
        "nextAction": next_action,
        "chainStatus": "active",
        "copyPastePrompt": copy_paste_prompt,
    }

    path = directory / f"{handoff_id}.json"
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "record": record,
        "path": path,
        "compact": compact_handoff(record, path),
    }


def consume_handoff(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["chainStatus"] = "consumed"
    record["consumedAt"] = utc_now_iso()

    destination_dir = consumed_chain_dir(record["chainId"])
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / path.name
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    shutil.move(str(path), str(destination))
    return {
        "handoffId": record["handoffId"],
        "mode": "consume",
        "handoffPath": relative_path(destination),
    }


def delete_or_consume_handoff(
    *,
    handoff_id: str = "",
    handoff_path: str = "",
    mode: str = "consume",
) -> dict[str, Any]:
    path = handoff_path_from_user_input(handoff_id, handoff_path)

    if mode == "consume":
        return consume_handoff(path)
    if mode == "delete":
        record = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
        return {
            "handoffId": record["handoffId"],
            "mode": "delete",
            "handoffPath": relative_path(path),
        }

    raise ValueError("mode must be 'consume' or 'delete'")


def delete_or_consume_chain(chain_id: str, mode: str = "consume") -> dict[str, Any]:
    ensure_safe_id(chain_id, "chain_id")
    directory = chain_dir(chain_id)
    if not directory.exists():
        return {
            "chainId": chain_id,
            "mode": mode,
            "count": 0,
            "handoffs": [],
        }

    results = []
    for path in sorted(directory.glob("*.json")):
        if mode == "consume":
            results.append(consume_handoff(path))
        elif mode == "delete":
            record = json.loads(path.read_text(encoding="utf-8"))
            path.unlink()
            results.append({
                "handoffId": record["handoffId"],
                "mode": "delete",
                "handoffPath": relative_path(path),
            })
        else:
            raise ValueError("mode must be 'consume' or 'delete'")

    return {
        "chainId": chain_id,
        "mode": mode,
        "count": len(results),
        "handoffs": results,
    }
