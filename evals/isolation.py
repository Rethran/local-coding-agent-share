"""Run one TDD-coder sample in an isolated worktree and runtime dir.

Every sample and every epoch must start from a clean checkout. Otherwise run 2
sees run 1's edits, continuity packets, and handoffs — and the pass rate is a
lie.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AGENT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = "living-graph"
DEFAULT_MODEL = "night-runner:latest"
DEFAULT_TIMEOUT = 2400


class IsolationError(RuntimeError):
    pass


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and completed.returncode != 0:
        raise IsolationError(
            f"git {' '.join(args)} failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout)[-2000:]}"
        )
    return completed


@dataclass
class IsolatedRun:
    sample_id: str
    worktree: Path
    requirements_dir: Path
    runtime_dir: Path
    repo: Path
    run_summary_path: Path
    state_path: Path
    exit_code: int | None = None
    timed_out: bool = False
    wall_seconds: float = 0.0
    stdout_tail: str = ""
    stderr_tail: str = ""
    extra_temps: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        excerpt = summary_excerpt(self.run_summary_path)
        return {
            "sampleId": self.sample_id,
            "worktree": str(self.worktree),
            "requirementsDir": str(self.requirements_dir),
            "runtimeDir": str(self.runtime_dir),
            "runSummaryPath": str(self.run_summary_path),
            "statePath": str(self.state_path),
            "exitCode": self.exit_code,
            "timedOut": self.timed_out,
            "wallSeconds": round(self.wall_seconds, 3),
            "stdoutTail": self.stdout_tail,
            "stderrTail": self.stderr_tail,
            "totals": excerpt["totals"],
            "stopReasons": excerpt["stopReasons"],
        }

    def cleanup(self) -> None:
        _run_git(self.repo, "worktree", "remove", "--force", str(self.worktree), check=False)
        _run_git(self.repo, "worktree", "prune", check=False)
        for path in (self.requirements_dir, self.runtime_dir, *self.extra_temps):
            shutil.rmtree(path, ignore_errors=True)
        shutil.rmtree(self.worktree, ignore_errors=True)


def prepare_isolated_run(
    *,
    sample_id: str,
    task_id: str,
    base_commit: str,
    repo: Path,
    source_requirements: Path,
) -> IsolatedRun:
    if not base_commit:
        raise IsolationError(f"{sample_id}: missing base_commit")
    work = Path(tempfile.mkdtemp(prefix=f"eval-{sample_id}-work-"))
    shutil.rmtree(work, ignore_errors=True)
    _run_git(repo, "worktree", "add", "--detach", str(work), base_commit)

    reqs = Path(tempfile.mkdtemp(prefix=f"eval-{sample_id}-reqs-"))
    shutil.copytree(source_requirements, reqs, dirs_exist_ok=True)
    state_path = reqs / f"{task_id}.state.json"
    if state_path.exists():
        state_path.unlink()

    runtime = Path(tempfile.mkdtemp(prefix=f"eval-{sample_id}-runtime-"))
    summary = runtime / "run_summary.json"
    return IsolatedRun(
        sample_id=sample_id,
        worktree=work,
        requirements_dir=reqs,
        runtime_dir=runtime,
        repo=repo,
        run_summary_path=summary,
        state_path=state_path,
    )


def run_orchestrator(
    isolated: IsolatedRun,
    *,
    subtask_id: str,
    model_tag: str = DEFAULT_MODEL,
    prompt_variant: str = "baseline",
    timeout: int = DEFAULT_TIMEOUT,
    profile: str = DEFAULT_PROFILE,
) -> IsolatedRun:
    env = os.environ.copy()
    env["LOCAL_AGENT_PROFILE"] = profile
    env["LOCAL_AGENT_WORKSPACE_ROOT"] = str(isolated.worktree)
    env["LOCAL_AGENT_REQUIREMENTS_DIR"] = str(isolated.requirements_dir)
    env["LOCAL_AGENT_RUNTIME_DIR"] = str(isolated.runtime_dir)
    env["LOCAL_AGENT_PROMPT_VARIANT"] = prompt_variant
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        str(AGENT_ROOT / "main.py"),
        "--from",
        subtask_id,
        "--to",
        subtask_id,
        "--no-auto-push",
        "--hide-model-output",
        "--run-summary",
        str(isolated.run_summary_path),
        "--model",
        model_tag,
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=AGENT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        isolated.exit_code = completed.returncode
        isolated.stdout_tail = (completed.stdout or "")[-4000:]
        isolated.stderr_tail = (completed.stderr or "")[-4000:]
    except subprocess.TimeoutExpired as exc:
        isolated.timed_out = True
        isolated.exit_code = None
        isolated.stdout_tail = str(exc.stdout or "")[-4000:]
        isolated.stderr_tail = str(exc.stderr or "")[-4000:]
    isolated.wall_seconds = time.perf_counter() - started
    return isolated


def overlay_reference_tests(
    worktree: Path,
    repo: Path,
    accepted_commit: str,
    paths: list[str],
) -> list[str]:
    """Replace test files in the worktree with the accepted-commit versions."""
    overlaid: list[str] = []
    for relative in paths:
        shown = _run_git(repo, "show", f"{accepted_commit}:{relative}", check=False)
        if shown.returncode != 0:
            continue
        dest = worktree / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(shown.stdout, encoding="utf-8")
        overlaid.append(relative)
    return overlaid


_TOTAL_KEYS = (
    "steps",
    "toolCalls",
    "toolErrors",
    "attempts",
    "repeatedIdenticalCalls",
    "promptTokens",
    "completionTokens",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def summary_excerpt(path: Path) -> dict[str, Any]:
    """Compact numbers that must survive worktree cleanup.

    Scorers read ``run_summary.json`` while the sample is alive. The report
    reads the ``.eval`` log later, after that file is gone.
    """
    data = load_json(path)
    totals = data.get("totals") if isinstance(data.get("totals"), dict) else {}
    compact = {key: totals.get(key, 0) for key in _TOTAL_KEYS} if totals else {}
    stop_reasons = [
        str(attempt.get("stopReason"))
        for attempt in data.get("attempts") or []
        if isinstance(attempt, dict) and attempt.get("stopReason")
    ]
    return {"totals": compact, "stopReasons": stop_reasons}
