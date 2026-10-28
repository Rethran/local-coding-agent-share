"""Create a HEAT bug ticket from a symptom + verified heating-app symbols."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from orchestrator.config import DEFAULT_MODEL, DEFAULT_NUM_CTX
from orchestrator.heating_map import get_heating_repo_map
from orchestrator.ollama_client import ollama_chat
from tools import _tasks

BLAST_RADIUS_HOPS = 2
PROPOSE_NUM_PREDICT = 256

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]")


class BugTicketError(Exception):
    """User-facing failure: no verified symbols, or ticket could not be written."""


def next_heat_id() -> str:
    """Allocate HEAT-NNN as max existing task number + 1."""
    numbers = [_tasks.parse_task_number(task_id) for task_id in _tasks.load_task_definitions()]
    nxt = (max(numbers) + 1) if numbers else 1
    return f"HEAT-{nxt:03d}"


def _simple_name(qualified: str) -> str:
    return qualified.rsplit(".", 1)[-1]


def resolve_candidate(needle: str, symbols: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only real symbols. Exact name first; unique simple-name; else empty (ambiguous)."""
    raw = (needle or "").strip()
    if not raw:
        return []

    exact = [entry for entry in symbols if entry.get("name") == raw]
    if exact:
        return exact

    lowered = raw.lower()
    exact_ci = [entry for entry in symbols if str(entry.get("name", "")).lower() == lowered]
    if exact_ci:
        return exact_ci

    simple_hits = [
        entry
        for entry in symbols
        if _simple_name(str(entry.get("name", ""))).lower() == lowered
    ]
    distinct = {str(entry.get("name")) for entry in simple_hits}
    if len(distinct) == 1:
        return simple_hits
    return []


def guard_candidates(needles: list[str], repo_map: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve needles against the map. Deduplicate by symbol_id, highest score first."""
    symbols = list(repo_map.get("symbols") or [])
    by_id: dict[str, dict[str, Any]] = {}
    for needle in needles:
        for entry in resolve_candidate(needle, symbols):
            by_id[str(entry["symbol_id"])] = entry
    verified = list(by_id.values())
    verified.sort(key=lambda e: (-float(e.get("score") or 0.0), str(e.get("symbol_id"))))
    return verified


def _extract_name_list(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = _JSON_ARRAY_RE.search(cleaned)
        if match is None:
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []

    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]
    if isinstance(payload, dict):
        for key in ("names", "symbols", "candidates"):
            value = payload.get(key)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
    return []


def propose_symbol_names(
    summary: str,
    repo_map: dict[str, Any],
    *,
    model: str,
    num_ctx: int = DEFAULT_NUM_CTX,
) -> list[str]:
    """Ask the local model for existing names only. Return parsed strings; caller guards."""
    names = sorted({str(entry.get("name", "")) for entry in repo_map.get("symbols") or [] if entry.get("name")})
    if not names:
        return []

    response = ollama_chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You pick Python function names from a provided list. "
                    "Return a JSON array of strings. No markdown, no objects, no new names."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Bug report:\n{summary}\n\n"
                    "Existing function names (use only these):\n"
                    + "\n".join(names)
                    + "\n\nReturn a JSON array of 1-12 names that are most likely involved."
                ),
            },
        ],
        num_ctx=num_ctx,
        num_predict=PROPOSE_NUM_PREDICT,
        think=False,
    )
    content = ""
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        content = str(message.get("content") or "")
    elif message is not None:
        content = str(getattr(message, "content", "") or "")
    return _extract_name_list(content)


def _import_blast_radius():
    from repomap.blast_radius import blast_radius
    import networkx as nx

    return blast_radius, nx


def callers_of(symbol: dict[str, Any], repo_map: dict[str, Any], *, max_hops: int = BLAST_RADIUS_HOPS) -> list[dict[str, Any]]:
    blast_radius, nx = _import_blast_radius()
    graph = nx.DiGraph()
    graph.add_edges_from(repo_map.get("edges") or [])
    by_id = {str(entry["symbol_id"]): entry for entry in repo_map.get("symbols") or []}
    radius_ids = blast_radius(graph, str(symbol["symbol_id"]), max_hops=max_hops)
    callers = [by_id[rid] for rid in radius_ids if rid in by_id]
    callers.sort(key=lambda e: (-float(e.get("score") or 0.0), str(e.get("symbol_id"))))
    return callers


def _test_path_for(module: str, explicit: str | None) -> str:
    if explicit:
        return str(explicit).replace("\\", "/")
    stem = Path(str(module).replace("\\", "/")).stem
    return f"tests/test_{stem}.py"


def _related_artifacts(verified: list[dict[str, Any]], repo_map: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str] = set()
    artifacts: list[dict[str, str]] = []
    target_ids = {str(entry["symbol_id"]) for entry in verified}
    for symbol in verified:
        for caller in callers_of(symbol, repo_map):
            sid = str(caller["symbol_id"])
            if sid in target_ids or sid in seen:
                continue
            seen.add(sid)
            artifacts.append(
                {
                    "path": str(caller["file"]).replace("\\", "/"),
                    "name": str(caller["name"]),
                    "purpose": (
                        f"Caller of {symbol['name']} "
                        f"(blast radius, {BLAST_RADIUS_HOPS} hops)."
                    ),
                }
            )
    return artifacts


def build_bug_task(
    *,
    task_id: str,
    summary: str,
    verified: list[dict[str, Any]],
    repo_map: dict[str, Any],
    test_path: str | None = None,
) -> dict[str, Any]:
    target = verified[0]
    module = str(target["file"]).replace("\\", "/")
    function_name = str(target["name"])
    test_file = _test_path_for(module, test_path)
    related = _related_artifacts(verified, repo_map)
    verify_cmd = f"pytest {test_file} -v"
    target_artifacts = {
        "function": function_name,
        "module": module,
        "test": test_file,
    }
    involved = ", ".join(entry["name"] for entry in verified)
    caller_names = ", ".join(item["name"] for item in related) or "(none found)"

    subtask_one = {
        "id": f"{task_id}.1",
        "title": "Add a failing test that reproduces the bug",
        "summary": (
            f"Write a failing test in {test_file} that reproduces: {summary}. "
            f"Target {function_name} in {module}. Do not implement the fix yet."
        ),
        "readLegacy": False,
        "targetArtifacts": target_artifacts,
        "relatedArtifacts": related,
        "scope": [
            f"Add or extend {test_file} to reproduce: {summary}",
            f"Cover {function_name} in {module}",
        ],
        "verification": [verify_cmd],
        "instructions": [
            "contextPack already includes target and related caller snippets — do not rediscover.",
            f"Write a failing test for: {summary}",
            f"Run {verify_cmd} and leave the production code unchanged.",
        ],
    }
    subtask_two = {
        "id": f"{task_id}.2",
        "title": "Fix the bug and keep related callers working",
        "summary": (
            f"Fix {function_name} so the bug is gone without breaking callers: {caller_names}."
        ),
        "readLegacy": False,
        "targetArtifacts": target_artifacts,
        "relatedArtifacts": related,
        "scope": [
            f"Fix {function_name} in {module}",
            f"Keep related callers working: {caller_names}",
            f"Keep {test_file} passing",
        ],
        "verification": [verify_cmd],
        "instructions": [
            "contextPack already includes target and related caller snippets — do not rediscover.",
            f"Fix {function_name} for: {summary}",
            f"Run {verify_cmd} and mark DONE.",
        ],
    }

    return {
        "id": task_id,
        "type": "bug",
        "summary": summary,
        "dependsOn": [],
        "description": (
            f"Investigate and fix: {summary}. "
            f"Verified symbols: {involved}. Callers: {caller_names}."
        ),
        "scope": [
            f"Reproduce the bug in {test_file}",
            f"Fix {function_name} in {module}",
            "Do not change unrelated heating behavior",
        ],
        "acceptanceCriteria": (
            "A failing test reproduces the reported bug, the fix makes that test pass, "
            "and related in-app callers of the verified symbols still behave correctly."
        ),
        "constraints": [
            "Do not change unrelated heating behavior.",
            "Stay within app/ and tests/.",
            "Do not read old_entsoes/.",
            "Do not invent symbols that are not in the heating-app map.",
        ],
        "verification": [verify_cmd],
        "subtasks": [subtask_one, subtask_two],
    }


def collect_verified_symbols(
    summary: str,
    symbols: list[str],
    repo_map: dict[str, Any],
    *,
    model: str,
    num_ctx: int,
) -> list[dict[str, Any]]:
    needles = [item.strip() for item in symbols if item and item.strip()]
    if not needles:
        needles = propose_symbol_names(summary, repo_map, model=model, num_ctx=num_ctx)
    verified = guard_candidates(needles, repo_map)
    if not verified:
        raise BugTicketError(
            "No verified heating-app symbols. Pass --symbol with a real function "
            "name from app/ (the model may only propose names that already exist)."
        )
    return verified


def write_bug_task(task: dict[str, Any]) -> Path:
    task_id = str(task["id"])
    path = _tasks.task_path(task_id)
    if path.exists():
        raise BugTicketError(f"Task file already exists: {path}")
    path.write_text(json.dumps(task, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _tasks.ensure_subtask_state(task_id)
    return path


def create_bug_ticket(
    *,
    summary: str,
    symbols: list[str] | None = None,
    test_path: str | None = None,
    model: str = DEFAULT_MODEL,
    num_ctx: int = DEFAULT_NUM_CTX,
    repo_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    text = (summary or "").strip()
    if not text:
        raise BugTicketError("--summary is required.")

    heating_map = repo_map if repo_map is not None else get_heating_repo_map()
    verified = collect_verified_symbols(
        text,
        list(symbols or []),
        heating_map,
        model=model,
        num_ctx=num_ctx,
    )
    task_id = next_heat_id()
    task = build_bug_task(
        task_id=task_id,
        summary=text,
        verified=verified,
        repo_map=heating_map,
        test_path=test_path,
    )
    path = write_bug_task(task)
    return {
        "taskId": task_id,
        "path": str(path),
        "verified": [
            {"name": entry["name"], "file": entry["file"], "symbol_id": entry["symbol_id"]}
            for entry in verified
        ],
        "relatedArtifacts": task["subtasks"][0].get("relatedArtifacts") or [],
    }


def run_create_bug(args: Any) -> int:
    try:
        result = create_bug_ticket(
            summary=getattr(args, "summary", ""),
            symbols=list(getattr(args, "symbol", None) or []),
            test_path=getattr(args, "test", None) or None,
            model=getattr(args, "model", DEFAULT_MODEL),
            num_ctx=int(getattr(args, "num_ctx", DEFAULT_NUM_CTX)),
        )
    except BugTicketError as exc:
        print(str(exc))
        return 2

    print(f"Created {result['taskId']}")
    print(f"Path: {result['path']}")
    print("Verified symbols:")
    for entry in result["verified"]:
        print(f"  {entry['name']}  {entry['file']}")
    print("Callers (blast radius):")
    related = result["relatedArtifacts"]
    if not related:
        print("  (none)")
    for item in related:
        print(f"  {item['name']}  {item['path']}")
    print(f"Run: python main.py --from {result['taskId']} --to {result['taskId']}")
    return 0
