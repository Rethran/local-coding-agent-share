"""
Phase 2: build a call graph from a symbol table and rank symbols
with PageRank. A symbol that's called by many important symbols
should rank higher than a leaf function nobody calls.
"""
from pathlib import Path

from repomap.symbols import Symbol, extract_symbols_from_repo
from repomap.graph import build_call_graph, rank_symbols

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def _sym(name, file="mod.py", start=1):
    return Symbol(name=name, file=Path(file), signature=f"def {name}()", start_line=start, end_line=start + 1)


def test_build_call_graph_creates_edge_for_direct_call(tmp_path):
    caller_file = tmp_path / "caller.py"
    callee_file = tmp_path / "callee.py"
    callee_file.write_text("def target():\n    pass\n")
    caller_file.write_text("from callee import target\n\n\ndef caller():\n    target()\n")

    symbols = extract_symbols_from_repo(tmp_path)
    graph = build_call_graph(symbols, tmp_path)

    caller_id = next(s.symbol_id for s in symbols if s.name == "caller")
    target_id = next(s.symbol_id for s in symbols if s.name == "target")

    assert graph.has_edge(caller_id, target_id)


def test_build_call_graph_has_no_edge_for_unrelated_functions(tmp_path):
    f = tmp_path / "mod.py"
    f.write_text("def alone_one():\n    pass\n\n\ndef alone_two():\n    pass\n")

    symbols = extract_symbols_from_repo(tmp_path)
    graph = build_call_graph(symbols, tmp_path)

    a = next(s.symbol_id for s in symbols if s.name == "alone_one")
    b = next(s.symbol_id for s in symbols if s.name == "alone_two")
    assert not graph.has_edge(a, b)
    assert not graph.has_edge(b, a)


def test_build_call_graph_on_fixture_repo_links_service_to_cache():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)

    update_item = next(s.symbol_id for s in symbols if s.name == "update_item")
    invalidate = next(s.symbol_id for s in symbols if s.name == "invalidate")

    assert graph.has_edge(update_item, invalidate)


def test_rank_symbols_returns_all_symbols_scored():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)

    ranked = rank_symbols(graph, symbols)

    assert len(ranked) == len(symbols)
    ids = {s.symbol_id for s, _score in ranked}
    assert ids == {s.symbol_id for s in symbols}


def test_rank_symbols_is_sorted_descending_by_score():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)

    ranked = rank_symbols(graph, symbols)
    scores = [score for _sym, score in ranked]

    assert scores == sorted(scores, reverse=True)


def test_called_symbol_outranks_a_top_level_orchestrator_no_one_calls():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)
    ranked = rank_symbols(graph, symbols)

    ranked_names = [s.name for s, _score in ranked]
    # update_item is an entry point: nothing in the fixture calls it, so it
    # receives no PageRank weight. compute_key and _persist are both callees
    # of other symbols and should rank ahead of it.
    assert ranked_names.index("compute_key") < ranked_names.index("update_item")
    assert ranked_names.index("_persist") < ranked_names.index("update_item")


def test_rank_symbols_is_deterministic_across_runs():
    symbols = extract_symbols_from_repo(FIXTURE_REPO)
    graph = build_call_graph(symbols, FIXTURE_REPO)

    first = [s.symbol_id for s, _ in rank_symbols(graph, symbols)]
    second = [s.symbol_id for s, _ in rank_symbols(graph, symbols)]

    assert first == second
