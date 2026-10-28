"""Prefixed, HEAD-cached repo maps for a subtree (e.g. app/).

`build_repo_map(scan_root)` stores files relative to that root. Callers that
need workspace-relative paths (`app/domain/schedule.py`) rewrite through
`prefix_repo_map` and cache the result keyed by HEAD.
"""
from __future__ import annotations

import json
from pathlib import Path

from repomap.mcp_tool import _head_commit
from repomap.query import build_repo_map


def _posix(path: str) -> str:
    return str(path).replace("\\", "/").strip("/")


def prefix_repo_map(repo_map: dict, prefix: str) -> dict:
    """Rewrite file paths and symbol_ids so they are workspace-relative."""
    prefix = _posix(prefix)
    old_to_new: dict[str, str] = {}
    symbols: list[dict] = []
    for entry in repo_map.get("symbols") or []:
        new_file = _posix(f"{prefix}/{entry['file']}") if prefix else _posix(entry["file"])
        new_id = f"{new_file}:{entry['name']}:{entry['start_line']}"
        old_to_new[str(entry["symbol_id"])] = new_id
        symbols.append({**entry, "file": new_file, "symbol_id": new_id})

    edges = []
    for edge in repo_map.get("edges") or []:
        src, dst = edge[0], edge[1]
        edges.append([old_to_new.get(str(src), src), old_to_new.get(str(dst), dst)])
    edges.sort()
    symbols.sort(key=lambda e: (-e["score"], e["symbol_id"]))
    return {"symbols": symbols, "edges": edges}


def get_or_build_prefixed_map(
    scan_root: Path,
    prefix: str,
    cache_dir: Path,
) -> dict:
    """Build a prefixed map of `scan_root`, cached under cache_dir by HEAD + prefix."""
    scan_root = Path(scan_root)
    cache_dir = Path(cache_dir)
    commit = _head_commit(scan_root) or "nohead"
    cache_file = cache_dir / f"{_posix(prefix) or 'root'}-{commit}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    prefixed = prefix_repo_map(build_repo_map(scan_root), prefix)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(prefixed, sort_keys=True), encoding="utf-8")
    return prefixed
