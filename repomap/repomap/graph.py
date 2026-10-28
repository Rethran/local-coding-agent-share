"""
Phase 2 — call graph + ranking.

Builds a directed graph (caller -> callee) over the symbol table using
simple name resolution (no full import/type resolution — good enough
for ranking, not a type checker), then ranks symbols with PageRank so
"what matters" reflects how central a symbol is to the codebase, not
just where it sits alphabetically.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from repomap.symbols import Symbol, _PARSER


def _simple_name(qualified: str) -> str:
    """'Cache.get' -> 'get'; 'compute_key' -> 'compute_key'."""
    return qualified.rsplit(".", 1)[-1]


def _find_function_node(root, start_line: int, end_line: int):
    """Locate the tree-sitter node for a symbol by its 1-indexed line range."""
    target = None

    def visit(node):
        nonlocal target
        if node.type == "function_definition":
            if node.start_point[0] + 1 == start_line and node.end_point[0] + 1 == end_line:
                target = node
                return
        for child in node.children:
            if target is not None:
                return
            visit(child)

    visit(root)
    return target


def _extract_call_names(func_node, source: bytes) -> list[str]:
    """Names referenced in `foo(...)` or `obj.foo(...)` call expressions inside func_node."""
    names: list[str] = []

    def visit(node):
        if node.type == "call":
            fn = node.child_by_field_name("function")
            if fn is not None:
                if fn.type == "identifier":
                    names.append(source[fn.start_byte:fn.end_byte].decode("utf-8"))
                elif fn.type == "attribute":
                    attr = fn.child_by_field_name("attribute")
                    if attr is not None:
                        names.append(source[attr.start_byte:attr.end_byte].decode("utf-8"))
        for child in node.children:
            visit(child)

    body = func_node.child_by_field_name("body")
    if body is not None:
        visit(body)
    return names


def build_call_graph(symbols: list[Symbol], repo_root: Path) -> nx.DiGraph:
    """
    Build a directed graph over symbol_ids: an edge caller -> callee means
    caller's body contains a call that resolves (by simple name) to callee.
    """
    graph = nx.DiGraph()
    for sym in symbols:
        graph.add_node(sym.symbol_id)

    # index candidate callees by their simple (unqualified) name
    by_simple_name: dict[str, list[Symbol]] = {}
    for sym in symbols:
        by_simple_name.setdefault(_simple_name(sym.name), []).append(sym)

    # group symbols by file so we parse each file once
    by_file: dict[Path, list[Symbol]] = {}
    for sym in symbols:
        by_file.setdefault(sym.file, []).append(sym)

    for file_path, file_symbols in by_file.items():
        try:
            source = file_path.read_bytes()
            tree = _PARSER.parse(source)
        except Exception:
            continue

        for sym in file_symbols:
            func_node = _find_function_node(tree.root_node, sym.start_line, sym.end_line)
            if func_node is None:
                continue
            for called_name in _extract_call_names(func_node, source):
                for callee in by_simple_name.get(called_name, []):
                    if callee.symbol_id == sym.symbol_id:
                        continue  # skip self-recursion for ranking purposes
                    graph.add_edge(sym.symbol_id, callee.symbol_id)

    return graph


def rank_symbols(graph: nx.DiGraph, symbols: list[Symbol]) -> list[tuple[Symbol, float]]:
    """
    Rank symbols by PageRank over the call graph. Returns (symbol, score)
    pairs sorted descending by score, then by symbol_id for a fully
    deterministic tie-break.
    """
    by_id = {s.symbol_id: s for s in symbols}

    if graph.number_of_nodes() == 0:
        return []

    scores = nx.pagerank(graph)

    ranked = [(by_id[node_id], score) for node_id, score in scores.items() if node_id in by_id]
    ranked.sort(key=lambda pair: (-pair[1], pair[0].symbol_id))
    return ranked
