"""
Phase 4a — blast radius.

Given the call graph (edge caller -> callee) and a symbol that's
changing, blast radius answers "who is affected", which means walking
edges *backward*: everyone who (transitively, up to max_hops) calls
this symbol.
"""
from __future__ import annotations

from collections import deque

import networkx as nx


def blast_radius(graph: nx.DiGraph, symbol_id: str, max_hops: int = 2) -> set[str]:
    """
    BFS over reversed call edges starting at symbol_id, up to max_hops.
    Returns the set of affected symbol_ids, excluding symbol_id itself.
    Safe against cycles (each node visited once).
    """
    if symbol_id not in graph:
        return set()

    visited = {symbol_id}
    affected: set[str] = set()
    queue = deque([(symbol_id, 0)])

    while queue:
        node, depth = queue.popleft()
        if depth >= max_hops:
            continue
        for caller in graph.predecessors(node):
            if caller not in visited:
                visited.add(caller)
                affected.add(caller)
                queue.append((caller, depth + 1))

    return affected
