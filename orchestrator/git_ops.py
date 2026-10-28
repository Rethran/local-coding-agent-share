from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any

from tools import _recovery, _tasks
from tools._common import WORKSPACE_ROOT

from orchestrator.config import DEFAULT_GIT_TIMEOUT_SECONDS


class GitOperationError(RuntimeError):
    """Raised when subtask auto-commit cannot safely complete."""


@dataclass(frozen=True)
class GitCommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class AttemptSnapshot:
    files: _recovery.Snapshot
    dirty_git_paths: frozenset[str]


def run_git(args: list[str]) -> GitCommandResult:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=WORKSPACE_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=DEFAULT_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return GitCommandResult(
            returncode=127,
            stdout="",
            stderr="git executable was not found on PATH",
        )
    except subprocess.TimeoutExpired as exc:
        return GitCommandResult(
            returncode=124,
            stdout=str(exc.stdout or "").strip(),
            stderr=f"git command timed out after {DEFAULT_GIT_TIMEOUT_SECONDS} seconds",
        )
    return GitCommandResult(
        returncode=result.returncode,
        stdout=result.stdout.rstrip("\r\n"),
        stderr=result.stderr.strip(),
    )


def git_error_message(action: str, result: GitCommandResult) -> str:
    details = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if not details:
        details = f"git exited with code {result.returncode}"
    return f"{action} failed: {details}"


def require_git_success(action: str, result: GitCommandResult) -> GitCommandResult:
    if result.returncode != 0:
        raise GitOperationError(git_error_message(action, result))
    return result


def _porcelain_entry_path(entry: str) -> str | None:
    entry = entry.rstrip("\r\n")
    if len(entry) < 3:
        return None
    if entry[2:3] == " ":
        return entry[3:]
    # Defensive: a leading space on XY status must not be stripped before parsing.
    if entry[1:2] == " ":
        return entry[2:]
    return None


def parse_porcelain_paths(output: str) -> set[str]:
    paths: set[str] = set()
    entries = output.split("\0")
    index = 0
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue

        status = entry[:2]
        path = _porcelain_entry_path(entry)
        if path:
            paths.add(path)

        if status[:1] in {"R", "C"} and index < len(entries):
            previous_path = entries[index]
            index += 1
            if previous_path:
                paths.add(previous_path)

    return paths


def git_dirty_paths(paths: list[str] | None = None) -> set[str]:
    if paths is not None and not paths:
        return set()
    args = ["status", "--porcelain=v1", "-z"]
    if paths is not None:
        args.extend(["--", *paths])
    result = require_git_success("Inspecting git status", run_git(args))
    return parse_porcelain_paths(result.stdout)


def capture_attempt_snapshot() -> AttemptSnapshot:
    return AttemptSnapshot(
        files=_recovery.take_snapshot(WORKSPACE_ROOT),
        dirty_git_paths=frozenset(git_dirty_paths()),
    )


def changed_paths_since_snapshot(snapshot: AttemptSnapshot) -> list[str]:
    after_snapshot = _recovery.take_snapshot(WORKSPACE_ROOT)
    return sorted(_recovery.diff_snapshots(snapshot.files, after_snapshot))


def ensure_upstream_configured() -> None:
    result = run_git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if result.returncode != 0:
        raise GitOperationError(
            "Cannot auto-push because the current branch has no upstream. "
            "Set one with git push -u origin HEAD, then rerun the orchestrator."
        )


def clean_commit_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def subtask_commit_subject(task_id: str, subtask_id: str) -> str:
    spec = _tasks.get_subtask_spec_definition(task_id, subtask_id) or {}
    title = clean_commit_text(spec.get("title"))
    if not title:
        title = "Complete subtask"
    if title.lower().startswith(subtask_id.lower()):
        return title
    return f"{subtask_id}: {title}"


def subtask_commit_body(task_id: str, subtask_id: str) -> str:
    state = _tasks.load_task_state(task_id) or {}
    for item in state.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            summary = clean_commit_text(item.get("summary"))
            if summary:
                return summary

    spec = _tasks.get_subtask_spec_definition(task_id, subtask_id) or {}
    return clean_commit_text(spec.get("summary"))


def mark_auto_commit_failed(task_id: str, subtask_id: str, message: str) -> None:
    _tasks.update_task_state(
        task_id,
        "FAILED",
        summary=f"Subtask {subtask_id} completed, but auto commit/push failed.",
        blockers=[message],
        last_failure_reason=message,
        recommended_resume_step=(
            f"Resolve the git auto-commit/push failure for {subtask_id}, "
            "verify the completed work, then rerun the orchestrator."
        ),
    )


def _auto_commit_scoped_changes(
    scope_label: str,
    snapshot: AttemptSnapshot,
    *,
    subject: str,
    body: str,
    push: bool,
) -> None:
    candidate_paths = changed_paths_since_snapshot(snapshot)
    if not candidate_paths:
        print(f"{scope_label}: no subtask-scoped git changes to commit")
        return
    dirty_candidate_paths = git_dirty_paths(candidate_paths)
    if not dirty_candidate_paths:
        print(f"{scope_label}: no subtask-scoped git changes to commit")
        return

    preexisting_overlap = sorted(dirty_candidate_paths & set(snapshot.dirty_git_paths))
    if preexisting_overlap:
        raise GitOperationError(
            "Refusing auto-commit because these subtask-touched paths were already "
            f"dirty before the subtask attempt: {', '.join(preexisting_overlap)}"
        )

    if push:
        ensure_upstream_configured()

    paths_to_commit = sorted(dirty_candidate_paths)
    require_git_success("Staging subtask changes", run_git(["add", "--", *paths_to_commit]))

    diff_result = run_git(["diff", "--cached", "--quiet", "--", *paths_to_commit])
    if diff_result.returncode == 0:
        print(f"{scope_label}: no staged subtask-scoped changes to commit")
        return
    if diff_result.returncode != 1:
        raise GitOperationError(
            git_error_message("Inspecting staged subtask changes", diff_result)
        )

    commit_args = ["commit", "--only", "-m", subject]
    if body:
        commit_args.extend(["-m", body])
    commit_args.extend(["--", *paths_to_commit])
    require_git_success("Creating subtask commit", run_git(commit_args))
    print(f"{scope_label}: created git commit: {subject}")

    if push:
        require_git_success("Pushing subtask commit", run_git(["push"]))
        print(f"{scope_label}: pushed git commit")


def auto_commit_completed_subtask(
    task_id: str,
    subtask_id: str,
    snapshot: AttemptSnapshot,
    *,
    push: bool = True,
) -> None:
    subject = subtask_commit_subject(task_id, subtask_id)
    body = subtask_commit_body(task_id, subtask_id)
    _auto_commit_scoped_changes(
        f"{task_id}.{subtask_id}",
        snapshot,
        subject=subject,
        body=body,
        push=push,
    )
