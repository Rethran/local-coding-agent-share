"""
Phase 4: the query interface an agent actually calls — ties symbols,
ranking, churn and blast radius together into one deterministic,
token-budgeted result. This is repomap's equivalent of ripwire's
"what matters for X" response.
"""
import json
from pathlib import Path

from repomap.query import build_repo_map, query

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_build_repo_map_produces_all_symbols_with_scores_and_churn():
    repo_map = build_repo_map(FIXTURE_REPO)

    names = {entry["name"] for entry in repo_map["symbols"]}
    assert "invalidate" in names

    invalidate_entry = next(e for e in repo_map["symbols"] if e["name"] == "invalidate")
    assert "score" in invalidate_entry
    assert "churn" in invalidate_entry
    assert invalidate_entry["churn"] == 2  # pkg/cache.py touched twice


def test_build_repo_map_is_byte_identical_across_runs():
    first = json.dumps(build_repo_map(FIXTURE_REPO), sort_keys=True)
    second = json.dumps(build_repo_map(FIXTURE_REPO), sort_keys=True)

    assert first == second


def test_query_filters_by_symbol_name_substring_and_returns_blast_radius():
    repo_map = build_repo_map(FIXTURE_REPO)

    result = query(repo_map, "invalidate", max_hops=1, token_budget=2000)

    matched_names = {m["name"] for m in result["matches"]}
    assert "invalidate" in matched_names
    assert "invalidate_namespace" in matched_names

    invalidate_match = next(m for m in result["matches"] if m["name"] == "invalidate")
    assert "update_item" in invalidate_match["blast_radius"]


def test_query_respects_token_budget_by_truncating_matches():
    repo_map = build_repo_map(FIXTURE_REPO)

    result = query(repo_map, "e", max_hops=1, token_budget=1)  # tiny budget

    # with a near-zero budget we should get very few (possibly zero) matches,
    # never an error, and never the full unfiltered set.
    assert len(result["matches"]) < len(repo_map["symbols"])


def test_query_with_no_matches_returns_empty_list_not_error():
    repo_map = build_repo_map(FIXTURE_REPO)

    result = query(repo_map, "no_such_symbol_exists", max_hops=1, token_budget=2000)

    assert result["matches"] == []


def test_query_orders_matches_by_rank_score_descending():
    repo_map = build_repo_map(FIXTURE_REPO)

    result = query(repo_map, "", max_hops=1, token_budget=10000)  # match everything

    scores = [m["score"] for m in result["matches"]]
    assert scores == sorted(scores, reverse=True)
