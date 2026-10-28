"""
Phase 3 — git churn + change amplification.

Both metrics are computed straight from `git log --name-only`, no
external dependency beyond the git binary already on the machine.
"""
from __future__ import annotations

import subprocess
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


def _commit_file_lists(repo_root: Path) -> list[list[Path]]:
    """Return one list of touched file paths per commit, oldest-independent order."""
    try:
        result = subprocess.run(
            ["git", "log", "--name-only", "--pretty=format:--commit--"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    commits: list[list[Path]] = []
    current: list[Path] = []
    for line in result.stdout.splitlines():
        if line == "--commit--":
            if current:
                commits.append(current)
            current = []
        elif line.strip():
            current.append(Path(line.strip()))
    if current:
        commits.append(current)
    return commits


def compute_churn(repo_root: Path) -> dict[Path, int]:
    """Number of commits that touched each file."""
    repo_root = Path(repo_root)
    counts: Counter[Path] = Counter()
    for files in _commit_file_lists(repo_root):
        for f in files:
            counts[f] += 1
    return dict(counts)


def compute_change_amplification(repo_root: Path) -> dict[Path, int]:
    """
    For each file, the number of *distinct* other files it has ever been
    committed alongside. High amplification means touching this file
    correlates historically with touching many other files too.
    """
    repo_root = Path(repo_root)
    partners: dict[Path, set[Path]] = defaultdict(set)

    for files in _commit_file_lists(repo_root):
        unique_files = set(files)
        if len(unique_files) < 2:
            for f in unique_files:
                partners.setdefault(f, set())
            continue
        for a, b in combinations(sorted(unique_files), 2):
            partners[a].add(b)
            partners[b].add(a)
        for f in unique_files:
            partners.setdefault(f, set())

    return {f: len(others) for f, others in partners.items()}
