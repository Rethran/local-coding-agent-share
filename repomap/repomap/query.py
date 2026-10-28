"""
Phase 4 — the interface an agent actually calls.

`build_repo_map` does the (relatively) expensive one-time work per repo
snapshot and returns a plain, JSON-serializable dict — safe to cache to
disk and diff for the determinism guarantee ripwire leans on. `query`
is cheap and repo_map-only: no filesystem or git access, so it's safe
to call repeatedly per agent turn.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from repomap.blast_radius import blast_radius
from repomap.churn import compute_churn
from repomap.graph import build_call_graph, rank_symbols
from repomap.symbols import extract_symbols_from_repo


def build_repo_map(repo_root: Path) -> dict:
    """One-time (per snapshot) build: symbols + ranking + churn + call edges."""
    repo_root = Path(repo_root)
    symbols = extract_symbols_from_repo(repo_root)
    graph = build_call_graph(symbols, repo_root)
    scores = {sym.symbol_id: score for sym, score in rank_symbols(graph, symbols)}
    churn_by_file = compute_churn(repo_root)

    entries = []
    for sym in symbols:
        try:
            rel = sym.file.relative_to(repo_root)
        except ValueError:
            rel = sym.file
        entries.append(
            {
                "symbol_id": sym.symbol_id,
                "name": sym.name,
                "signature": sym.signature,
                "file": str(rel),
                "start_line": sym.start_line,
                "end_line": sym.end_line,
                "score": scores.get(sym.symbol_id, 0.0),
                "churn": churn_by_file.get(rel, 0),
            }
        )

    # deterministic order regardless of ranking ties or filesystem walk order
    entries.sort(key=lambda e: (-e["score"], e["symbol_id"]))
    edges = sorted(graph.edges())

    return {"symbols": entries, "edges": edges}


def _estimate_tokens(signature: str) -> int:
    """Rough, stable token estimate — good enough for budget accounting, not billing."""
    return max(1, len(signature) // 4)


def query(repo_map: dict, name_substring: str, max_hops: int = 1, token_budget: int = 2000) -> dict:
    """
    Filter repo_map's symbols by a case-insensitive substring match on
    name, rank-order the matches, and fill each with its blast radius —
    stopping once token_budget is spent so the response stays bounded.
    """
    graph = nx.DiGraph()
    graph.add_edges_from(repo_map["edges"])

    name_by_id = {e["symbol_id"]: e["name"] for e in repo_map["symbols"]}
    needle = name_substring.lower()

    candidates = [e for e in repo_map["symbols"] if needle in e["name"].lower()]
    candidates.sort(key=lambda e: (-e["score"], e["symbol_id"]))

    matches = []
    tokens_used = 0
    for entry in candidates:
        cost = _estimate_tokens(entry["signature"])
        if tokens_used + cost > token_budget:
            break
        radius_ids = blast_radius(graph, entry["symbol_id"], max_hops=max_hops)
        radius_names = sorted(name_by_id[rid] for rid in radius_ids if rid in name_by_id)
        matches.append({**entry, "blast_radius": radius_names})
        tokens_used += cost

    return {"matches": matches, "token_budget_used": tokens_used}
