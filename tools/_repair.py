"""Parse, repair, and learn from malformed <tool_call> XML in tool arguments."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOL_CALL_MARKERS = ("<tool_call>", "<function=", "<parameter=")

# Known wrong names Continue models invent — extended by the learning store.
STATIC_TOOL_ALIASES: dict[str, str] = {
    "local_agent_read_file": "read_file",
    "local_agent_write_file": "write_file",
    "local_agent_create_folder": "create_folder",
    "local_agent_list_directory": "list_directory",
    "local_agent_search_files": "search_files",
    "local_agent_find_symbol": "find_symbol",
    "local_agent_extract_symbol": "extract_symbol",
    "local_agent_analyze_project": "analyze_project",
    "local_agent_run_command": "run_command",
    "local_agent_extract_structure": "extract_structure",
    "local_agent_map_summarize_modules": "map_summarize_modules",
    "local_agent_reduce_architecture_review": "reduce_architecture_review",
    "local_agent_analyze_architecture": "analyze_architecture",
    "local-agent-read_file": "read_file",
    "codebase_search": "search_files",
}

FUNCTION_PATTERN = re.compile(r"<function=([^>\s]+)>", re.IGNORECASE)
PARAMETER_PATTERN = re.compile(
    r"<parameter=(?:name=)?[\"']?([^>\s\"']+)[\"']?>\s*(.*?)\s*</parameter>",
    re.IGNORECASE | re.DOTALL,
)
# Continue models often omit </parameter> and close with </tool_call> instead.
PARAMETER_LOOSE_PATTERN = re.compile(
    r"<parameter=(?:name=)?[\"']?([^>\s\"']+)[\"']?>\s*(.*?)(?=</(?:parameter|function|tool_call)>|$)",
    re.IGNORECASE | re.DOTALL,
)
PARAMETER_NAME_ATTR_PATTERN = re.compile(
    r"<parameter\s+name=[\"']([^\"']+)[\"']\s*>(.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)

MEMORY_PATH = Path(__file__).resolve().parent.parent / ".tool_repair_memory.json"
MAX_LEARNED_ALIASES = 200


@dataclass
class ParsedToolCall:
    tool_name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class RepairResult:
    tool_name: str
    arguments: dict[str, Any]
    was_repaired: bool = False
    redirect_tool: bool = False
    repair_notes: list[str] = field(default_factory=list)
    source: str = "original"


def looks_like_xml_tool_call(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    stripped = text.lstrip()
    return any(marker in stripped for marker in TOOL_CALL_MARKERS)


def normalize_tool_name(name: str, memory: RepairMemory | None = None) -> str:
    cleaned = name.strip().strip("<>/")
    aliases = STATIC_TOOL_ALIASES.copy()
    if memory is not None:
        aliases.update(memory.learned_aliases())
    return aliases.get(cleaned, cleaned)


def parse_xml_tool_call(text: str, memory: RepairMemory | None = None) -> ParsedToolCall | None:
    if not looks_like_xml_tool_call(text):
        return None

    function_match = FUNCTION_PATTERN.search(text)
    if not function_match:
        return None

    tool_name = normalize_tool_name(function_match.group(1), memory)
    arguments: dict[str, Any] = {}

    for match in PARAMETER_PATTERN.finditer(text):
        param_name = match.group(1).strip()
        param_value = match.group(2).strip()
        arguments[param_name] = param_value

    if not arguments:
        for match in PARAMETER_LOOSE_PATTERN.finditer(text):
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()
            if param_value:
                arguments[param_name] = param_value

    for match in PARAMETER_NAME_ATTR_PATTERN.finditer(text):
        param_name = match.group(1).strip()
        param_value = match.group(2).strip()
        arguments[param_name] = param_value

    if not arguments:
        return None

    return ParsedToolCall(tool_name=tool_name, arguments=arguments, raw=text)


def _coerce_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def repair_arguments(
    tool_name: str,
    arguments: dict[str, Any] | None,
    memory: RepairMemory | None = None,
) -> RepairResult:
    args = dict(arguments or {})
    memory = memory or RepairMemory.load()

    learned = memory.match_argument_repair(tool_name, args)
    if learned is not None:
        return RepairResult(
            tool_name=learned["tool_name"],
            arguments=learned["arguments"],
            was_repaired=True,
            redirect_tool=learned["tool_name"] != tool_name,
            repair_notes=[f"Applied learned repair ({learned['hit_count']} prior hits)"],
            source="learned",
        )

    xml_blob = _find_xml_blob(args)
    if xml_blob is None:
        return RepairResult(tool_name=tool_name, arguments=args)

    parsed = parse_xml_tool_call(xml_blob, memory)
    if parsed is None:
        return RepairResult(
            tool_name=tool_name,
            arguments=args,
            was_repaired=False,
            repair_notes=["Detected XML tool-call text but could not parse parameters"],
        )

    merged = {**args, **parsed.arguments}
    for key, value in list(merged.items()):
        if isinstance(value, str) and looks_like_xml_tool_call(value):
            del merged[key]
        else:
            merged[key] = _coerce_json_value(value)

    redirect = parsed.tool_name != normalize_tool_name(tool_name, memory)
    notes = ["Parsed <tool_call> XML from malformed argument input"]
    if redirect:
        notes.append(f"Redirecting {tool_name!r} -> {parsed.tool_name!r}")

    return RepairResult(
        tool_name=parsed.tool_name,
        arguments=merged,
        was_repaired=True,
        redirect_tool=redirect,
        repair_notes=notes,
        source="parsed_xml",
    )


def repair_output_text(text: str, memory: RepairMemory | None = None) -> RepairResult | None:
    """When tool/chat output is XML instead of real data, parse intended call."""
    if not looks_like_xml_tool_call(text):
        return None

    parsed = parse_xml_tool_call(text, memory)
    if parsed is None:
        return None

    return RepairResult(
        tool_name=parsed.tool_name,
        arguments=parsed.arguments,
        was_repaired=True,
        redirect_tool=True,
        repair_notes=["Output contained <tool_call> XML; extracted intended tool invocation"],
        source="output_xml",
    )


def _find_xml_blob(arguments: dict[str, Any]) -> str | None:
    for value in arguments.values():
        if isinstance(value, str) and looks_like_xml_tool_call(value):
            return value

    combined = json.dumps(arguments)
    if looks_like_xml_tool_call(combined):
        return combined
    return None


class RepairMemory:
    """Persistent store of successful repairs for pattern reuse."""

    def __init__(self, data: dict[str, Any] | None = None):
        self.data = data or {"aliases": {}, "repairs": []}

    @classmethod
    def load(cls, path: Path = MEMORY_PATH) -> RepairMemory:
        if not path.exists():
            return cls()
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return cls()

    def save(self, path: Path = MEMORY_PATH) -> None:
        path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def learned_aliases(self) -> dict[str, str]:
        return dict(self.data.get("aliases", {}))

    def record_success(self, repair: RepairResult, original_tool: str, original_args: dict) -> None:
        if not repair.was_repaired:
            return

        raw_name = _extract_raw_tool_name(original_args) or original_tool
        normalized = repair.tool_name
        if raw_name and raw_name != normalized:
            aliases = self.data.setdefault("aliases", {})
            aliases[raw_name] = normalized
            if len(aliases) > MAX_LEARNED_ALIASES:
                for key in list(aliases)[: len(aliases) - MAX_LEARNED_ALIASES]:
                    del aliases[key]

        fingerprint = _repair_fingerprint(original_tool, original_args)
        repairs: list[dict] = self.data.setdefault("repairs", [])
        for entry in repairs:
            if entry.get("fingerprint") == fingerprint:
                entry["hit_count"] = entry.get("hit_count", 0) + 1
                entry["last_success"] = _now_iso()
                entry["tool_name"] = repair.tool_name
                entry["arguments"] = repair.arguments
                entry["notes"] = repair.repair_notes
                self.save()
                return

        repairs.append({
            "id": str(uuid.uuid4()),
            "fingerprint": fingerprint,
            "tool_name": repair.tool_name,
            "arguments": repair.arguments,
            "original_tool": original_tool,
            "hit_count": 1,
            "last_success": _now_iso(),
            "notes": repair.repair_notes,
        })
        if len(repairs) > MAX_LEARNED_ALIASES:
            self.data["repairs"] = repairs[-MAX_LEARNED_ALIASES :]
        self.save()

    def match_argument_repair(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any] | None:
        fingerprint = _repair_fingerprint(tool_name, arguments)
        for entry in reversed(self.data.get("repairs", [])):
            if entry.get("fingerprint") == fingerprint:
                return {
                    "tool_name": entry["tool_name"],
                    "arguments": entry["arguments"],
                    "hit_count": entry.get("hit_count", 1),
                }
        return None


def _extract_raw_tool_name(arguments: dict[str, Any]) -> str | None:
    blob = _find_xml_blob(arguments)
    if not blob:
        return None
    match = FUNCTION_PATTERN.search(blob)
    return match.group(1).strip() if match else None


def _repair_fingerprint(tool_name: str, arguments: dict[str, Any]) -> str:
    blob = _find_xml_blob(arguments)
    if blob:
        parsed = parse_xml_tool_call(blob)
        if parsed:
            return f"xml:{parsed.tool_name}:{sorted(parsed.arguments.items())}"
        return f"xml-raw:{hash(blob)}"
    return f"args:{tool_name}:{sorted((k, str(v)[:120]) for k, v in arguments.items())}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
