"""
Phase 5 — expose repomap as a FastMCP tool.

Mirrors a per-tool module layout: one file, one tool, plugging into
the same ValidationMiddleware rather than inventing a second validation
path. Cache the repo_map on disk keyed by HEAD commit so repeated
queries in one session don't re-walk the repo every call.
"""
from __future__ import annotations

import json
from pathlib import Path

from repomap.query import build_repo_map, query

_CACHE_DIR = Path(".repomap_cache")


def _head_commit(repo_root: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_or_build_repo_map(repo_root: Path) -> dict:
    """Cache repo_map to disk keyed by HEAD commit; rebuild only when HEAD moves."""
    repo_root = Path(repo_root)
    commit = _head_commit(repo_root)
    cache_dir = repo_root / _CACHE_DIR
    cache_file = cache_dir / f"{commit}.json" if commit else None

    if cache_file and cache_file.exists():
        return json.loads(cache_file.read_text())

    repo_map = build_repo_map(repo_root)

    if cache_file:
        cache_dir.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps(repo_map, sort_keys=True))

    return repo_map


def register_repo_context_tool(mcp_server, repo_root: Path, validation_middleware=None) -> None:
    """
    Call from your existing mcp_server.py, e.g.:

        from repomap.mcp_tool import register_repo_context_tool
        register_repo_context_tool(mcp, repo_root=PROJECT_ROOT,
                                    validation_middleware=ValidationMiddleware(...))

    Registers a single tool: get_repo_context(query, max_hops, token_budget).
    """

    @mcp_server.tool()
    def get_repo_context(query_text: str, max_hops: int = 1, token_budget: int = 2000) -> str:
        """
        Return ranked symbols matching `query_text`, with call relationships,
        churn, and blast radius, capped at token_budget. Deterministic for a
        given HEAD commit — safe to call repeatedly within one agent session.
        """
        repo_map = get_or_build_repo_map(repo_root)
        result = query(repo_map, query_text, max_hops=max_hops, token_budget=token_budget)
        payload = json.dumps(result, sort_keys=True)

        if validation_middleware is not None:
            payload = validation_middleware.validate_output(payload)

        return payload
