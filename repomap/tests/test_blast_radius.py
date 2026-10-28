"""
Phase 4a: blast radius — if this symbol changes, what else is reachable
by walking "who calls this" edges outward, capped at N hops.
"""
from pathlib import Path

import networkx as nx

from repomap.blast_radius import blast_radius

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _graph_from_edges(edges):
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


def test_blast_radius_includes_direct_callers():
    # a -> b means "a calls b". Blast radius of b (if b changes) is a.
    graph = _graph_from_edges([("a", "b")])

    radius = blast_radius(graph, "b", max_hops=1)

    assert radius == {"a"}


def test_blast_radius_excludes_the_symbol_itself():
    graph = _graph_from_edges([("a", "b")])

    radius = blast_radius(graph, "b", max_hops=5)

    assert "b" not in radius


def test_blast_radius_respects_max_hops():
    # c -> b -> a  (c calls b, b calls a). Blast radius of a at 1 hop = {b} only.
    graph = _graph_from_edges([("c", "b"), ("b", "a")])

    radius_1hop = blast_radius(graph, "a", max_hops=1)
    radius_2hop = blast_radius(graph, "a", max_hops=2)

    assert radius_1hop == {"b"}
    assert radius_2hop == {"b", "c"}


def test_blast_radius_of_symbol_with_no_callers_is_empty():
    graph = _graph_from_edges([("a", "b")])

    radius = blast_radius(graph, "a", max_hops=5)

    assert radius == set()


def test_blast_radius_handles_cycles_without_infinite_loop():
    graph = _graph_from_edges([("a", "b"), ("b", "a")])

    radius = blast_radius(graph, "a", max_hops=10)

    assert radius == {"b"}


def test_blast_radius_on_fixture_repo():
    from repomap.symbols import extract_symbols_from_repo
    from repomap.graph import build_call_graph

    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)
    compute_key_id = next(s.symbol_id for s in symbols if s.name == "compute_key")
    invalidate_id = next(s.symbol_id for s in symbols if s.name == "invalidate")

    radius = blast_radius(graph, compute_key_id, max_hops=1)

    assert invalidate_id in radius
