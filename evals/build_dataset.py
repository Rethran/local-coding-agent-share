"""Build eval cases from Living Graph subtask commits.

Each historically accepted `GRAPH-N.M: title` commit is one case:
input = the requirement spec, base_commit = parent SHA (the repo the agent
starts from), accepted_commit = the SHA that landed, reference_tests = tests/
paths changed in that commit.

Latest commit wins when the same subtask id appears twice.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = Path(__file__).resolve().parent / "cases" / "living-graph.jsonl"
SUBJECT_RE = re.compile(r"^(GRAPH-\d+)\.(\d+):\s*(.+)$")

# Plumbing: three small, diverse subtasks. Run these first.
SMOKE_IDS = ("GRAPH-011.1", "GRAPH-013.1", "GRAPH-047.1")

# Overnight comparison set. GRAPH-051.1 is historically BLOCKED and has no
# landing commit, so it is not in the git-derived dataset.
REGRESSION_IDS = (
    "GRAPH-003.1",
    "GRAPH-006.1",
    "GRAPH-011.1",
    "GRAPH-013.1",
    "GRAPH-031.1",
    "GRAPH-046.1",
    "GRAPH-047.1",
    "GRAPH-048.1",
    "GRAPH-049.1",
    "GRAPH-050.1",
    "GRAPH-052.1",
    "GRAPH-054.1",
)

# GRAPH-051.1 is historically BLOCKED and never produced a GRAPH-051.1 commit,
# so it cannot be a git-derived case until it lands.


def default_workspace() -> Path:
    profile = json.loads(
        (AGENT_ROOT / "projects" / "living-graph.json").read_text(encoding="utf-8")
    )
    return Path(profile["workspaceRoot"])


def default_requirements_dir() -> Path:
    profile = json.loads(
        (AGENT_ROOT / "projects" / "living-graph.json").read_text(encoding="utf-8")
    )
    return Path(profile["requirementsDir"])


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def iter_subtask_commits(repo: Path) -> list[dict[str, Any]]:
    raw = _git(
        repo,
        "log",
        "--pretty=format:%H%x09%P%x09%s",
        "--name-only",
    )
    blocks = raw.split("\n\n")
    seen: dict[str, dict[str, Any]] = {}
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        sha, parents, subject = lines[0].split("\t", 2)
        match = SUBJECT_RE.match(subject.strip())
        if not match:
            continue
        task_id, suffix, title = match.group(1), match.group(2), match.group(3).strip()
        sample_id = f"{task_id}.{suffix}"
        if sample_id in seen:
            continue  # git log is newest-first; keep the latest landing
        parent = parents.split()[0] if parents.strip() else ""
        files = [path.replace("\\", "/") for path in lines[1:]]
        tests = [path for path in files if path.startswith("tests/")]
        seen[sample_id] = {
            "id": sample_id,
            "task_id": task_id,
            "subtask_id": sample_id,
            "title": title,
            "accepted_commit": sha,
            "base_commit": parent,
            "changed_files": files,
            "reference_tests": tests,
        }
    return list(seen.values())


def _load_spec(requirements_dir: Path, task_id: str) -> dict[str, Any]:
    path = requirements_dir / f"{task_id}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _subtask_summary(spec: dict[str, Any], subtask_id: str) -> str:
    for item in spec.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return str(item.get("summary") or item.get("title") or "")
    return str(spec.get("summary") or "")


def tags_for(sample_id: str) -> list[str]:
    tags: list[str] = []
    if sample_id in SMOKE_IDS:
        tags.append("smoke")
    if sample_id in REGRESSION_IDS:
        tags.append("regression")
    return tags


def build_cases(
    *,
    workspace: Path | None = None,
    requirements_dir: Path | None = None,
) -> list[dict[str, Any]]:
    repo = workspace or default_workspace()
    reqs = requirements_dir or default_requirements_dir()
    cases: list[dict[str, Any]] = []
    for commit in iter_subtask_commits(repo):
        spec = _load_spec(reqs, commit["task_id"])
        summary = _subtask_summary(spec, commit["subtask_id"])
        tags = tags_for(commit["id"])
        cases.append({
            "id": commit["id"],
            "input": (
                f"Implement {commit['subtask_id']}: {commit['title']}. "
                f"{summary}"
            ).strip(),
            "target": "",
            "metadata": {
                **commit,
                "tags": tags,
                "summary": summary,
            },
        })
    cases.sort(key=lambda row: row["id"])
    return cases


def write_cases(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    return path


def load_cases(
    path: Path | None = None,
    *,
    tags: str = "",
) -> list[dict[str, Any]]:
    source = path or DEFAULT_CASES
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wanted = {item.strip() for item in tags.split(",") if item.strip()}
    if not wanted:
        return rows
    return [
        row
        for row in rows
        if wanted.intersection(row.get("metadata", {}).get("tags") or [])
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build living-graph eval cases from git history.")
    parser.add_argument("--workspace", default="", help="Living Graph git repo (default: profile workspaceRoot).")
    parser.add_argument("--requirements-dir", default="", help="requirements/living-graph directory.")
    parser.add_argument("--out", default=str(DEFAULT_CASES))
    args = parser.parse_args(argv)
    cases = build_cases(
        workspace=Path(args.workspace) if args.workspace else None,
        requirements_dir=Path(args.requirements_dir) if args.requirements_dir else None,
    )
    out = write_cases(Path(args.out), cases)
    smoke = sum(1 for case in cases if "smoke" in case["metadata"]["tags"])
    regression = sum(1 for case in cases if "regression" in case["metadata"]["tags"])
    print(f"Wrote {len(cases)} cases ({smoke} smoke, {regression} regression) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
