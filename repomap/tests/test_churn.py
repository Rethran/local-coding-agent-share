"""
Phase 3: git churn + change amplification.

Churn = how often a file has changed. Change amplification = when this
file changes, how many *other* files tend to change in the same commit
(a proxy for "touching this ripples outward").
"""
from pathlib import Path

from repomap.churn import compute_churn, compute_change_amplification

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


def test_compute_churn_counts_commits_touching_each_file():
    churn = compute_churn(FIXTURE_REPO)

    # pkg/cache.py was touched in the initial commit + the later tweak commit = 2
    assert churn[Path("pkg/cache.py")] == 2
    # pkg/service.py was only added once
    assert churn[Path("pkg/service.py")] == 1


def test_compute_churn_returns_zero_for_untouched_file(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path)

    churn = compute_churn(tmp_path)

    assert churn.get(Path("nonexistent.py"), 0) == 0


def test_compute_churn_on_non_git_directory_returns_empty(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")

    churn = compute_churn(tmp_path)

    assert churn == {}


def test_change_amplification_counts_distinct_cochanged_files():
    amp = compute_change_amplification(FIXTURE_REPO)

    # cache.py was committed alongside __init__.py once (the initial commit)
    # and alone once (the tweak commit) -> exactly 1 distinct co-changed file.
    assert amp[Path("pkg/cache.py")] == 1
    # service.py has never shared a commit with anything else.
    assert amp[Path("pkg/service.py")] == 0


def test_change_amplification_is_symmetric(tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path)
    (tmp_path / "a.py").write_text("1\n")
    (tmp_path / "b.py").write_text("2\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path)
    subprocess.run(["git", "commit", "-q", "-m", "both together"], cwd=tmp_path)

    amp = compute_change_amplification(tmp_path)

    assert amp[Path("a.py")] == amp[Path("b.py")] == 1
