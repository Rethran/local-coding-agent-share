"""
Phase 6 — the determinism gate.

ripwire's own CI runs the binary twice and diffs stdout byte-for-byte.
Same idea here: build the repo_map three times over the fixture repo
and require byte-identical JSON every time. This should live in your
CI pipeline as a required check, not just a local test.
"""
import json
from pathlib import Path

from repomap.query import build_repo_map

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_repo_map_is_byte_identical_across_three_runs():
    runs = [json.dumps(build_repo_map(FIXTURE_REPO), sort_keys=True) for _ in range(3)]

    assert runs[0] == runs[1] == runs[2]


def test_repo_map_json_round_trips_without_loss():
    original = build_repo_map(FIXTURE_REPO)
    round_tripped = json.loads(json.dumps(original, sort_keys=True))

    # edges come back as lists (JSON has no tuples) - normalize before comparing
    original_edges = [list(e) for e in original["edges"]]
    assert round_tripped["edges"] == original_edges
    assert round_tripped["symbols"] == original["symbols"]
