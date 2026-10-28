"""Prefixed repo maps expose workspace-relative files and rewritten edges."""
from pathlib import Path

from repomap.query import build_repo_map
from repomap.scoped import get_or_build_prefixed_map, prefix_repo_map

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_prefix_repo_map_rewrites_files_and_edges() -> None:
    raw = build_repo_map(FIXTURE_REPO)
    prefixed = prefix_repo_map(raw, "app")

    files = {entry["file"] for entry in prefixed["symbols"]}
    assert "app/pkg/cache.py" in files
    assert "app/pkg/service.py" in files
    assert all(entry["symbol_id"].startswith("app/") for entry in prefixed["symbols"])
    assert prefixed["edges"]
    for src, dst in prefixed["edges"]:
        assert src.startswith("app/")
        assert dst.startswith("app/")

    by_name = {entry["name"]: entry for entry in prefixed["symbols"]}
    update_id = by_name["update_item"]["symbol_id"]
    invalidate_id = by_name["invalidate"]["symbol_id"]
    assert [update_id, invalidate_id] in prefixed["edges"]


def test_prefixed_map_cache_round_trip(tmp_path: Path) -> None:
    first = get_or_build_prefixed_map(FIXTURE_REPO, "app", tmp_path)
    cached_files = list(tmp_path.glob("app-*.json"))
    assert cached_files
    second = get_or_build_prefixed_map(FIXTURE_REPO, "app", tmp_path)
    assert first == second
