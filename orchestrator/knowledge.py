"""Project knowledge packs and task-scoped prompt guidance."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from orchestrator.project import AGENT_ROOT, PROJECT_PROFILE

_TASK_NUMBER = re.compile(r"^[A-Z]+-(\d+)")


def task_number(task_or_subtask_id: str | None) -> int | None:
    if not task_or_subtask_id:
        return None
    match = _TASK_NUMBER.match(task_or_subtask_id.strip())
    if not match:
        return None
    return int(match.group(1))


def _range_includes(entry: dict[str, Any], number: int) -> bool:
    start = task_number(str(entry.get("from") or entry.get("fromId") or ""))
    end = task_number(str(entry.get("to") or entry.get("toId") or ""))
    if start is None:
        return False
    if end is None:
        end = start
    return start <= number <= end


def guidance_for(task_or_subtask_id: str | None) -> str:
    number = task_number(task_or_subtask_id)
    default = str(PROJECT_PROFILE.prompt_guidance or "").strip()
    if number is None:
        return default
    for entry in PROJECT_PROFILE.task_prompt_guidance:
        if isinstance(entry, dict) and _range_includes(entry, number):
            text = str(entry.get("guidance") or "").strip()
            if text:
                return text
    return default


def knowledge_pack_text(task_or_subtask_id: str | None) -> str:
    number = task_number(task_or_subtask_id)
    if number is None:
        return ""
    chunks: list[str] = []
    for entry in PROJECT_PROFILE.knowledge_packs:
        if not isinstance(entry, dict) or not _range_includes(entry, number):
            continue
        relative = str(entry.get("path") or "").strip()
        if not relative:
            continue
        path = Path(relative)
        if not path.is_absolute():
            path = AGENT_ROOT / relative
        try:
            text = path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        if text:
            pack_id = str(entry.get("id") or path.stem)
            chunks.append(f"[{pack_id}]\n{text}")
    return "\n\n".join(chunks)
