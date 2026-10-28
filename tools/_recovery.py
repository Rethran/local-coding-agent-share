"""Task-attempt checkpoints and safe rollback helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from tools._common import WORKSPACE_ROOT, is_within_ignored_directory

EXCLUDED_SUFFIXES = (".pyc", ".log")
EXCLUDED_PARTS = frozenset({
    ".git",
    ".pytest_cache",
    "__pycache__",
    "handoffs",
})


@dataclass(frozen=True)
class FileRecord:
    """A file's content at one point in time."""

    digest: str
    content: bytes


@dataclass(frozen=True)
class RestoreResult:
    """Summary of a rollback attempt."""

    restored: list[str]
    deleted: list[str]
    skipped: list[str]


Snapshot = dict[str, FileRecord]


def should_snapshot(path: Path, root: Path = WORKSPACE_ROOT) -> bool:
    """Return True when a workspace file should be protected by recovery."""
    if not path.is_file():
        return False
    if is_within_ignored_directory(path):
        return False
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return False
    if path.name.endswith(".state.json"):
        return False
    if path.suffix in EXCLUDED_SUFFIXES:
        return False

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _record(path: Path) -> FileRecord:
    content = path.read_bytes()
    return FileRecord(
        digest=hashlib.sha256(content).hexdigest(),
        content=content,
    )


def take_snapshot(root: Path = WORKSPACE_ROOT) -> Snapshot:
    """Capture rollback-safe workspace file contents."""
    snapshot: Snapshot = {}
    for path in root.rglob("*"):
        if should_snapshot(path, root):
            snapshot[path.relative_to(root).as_posix()] = _record(path)
    return snapshot


def diff_snapshots(before: Snapshot, after: Snapshot) -> list[str]:
    """Return files added, removed, or modified between two snapshots."""
    changed: list[str] = []
    for rel_path in sorted(set(before) | set(after)):
        before_record = before.get(rel_path)
        after_record = after.get(rel_path)
        if before_record is None or after_record is None:
            changed.append(rel_path)
        elif before_record.digest != after_record.digest:
            changed.append(rel_path)
    return changed


def restore_changes(
    *,
    root: Path,
    before: Snapshot,
    after_failure: Snapshot,
    changed_files: list[str],
) -> RestoreResult:
    """Restore files changed during an attempt if they have not changed again.

    `after_failure` is the snapshot taken immediately after the failed attempt.
    If current file content differs from that snapshot, recovery skips the file
    to avoid overwriting unrelated user edits made after failure handling began.
    """
    restored: list[str] = []
    deleted: list[str] = []
    skipped: list[str] = []

    for rel_path in changed_files:
        path = root / rel_path
        before_record = before.get(rel_path)
        failed_record = after_failure.get(rel_path)

        if path.exists():
            current_record = _record(path)
            if failed_record is None or current_record.digest != failed_record.digest:
                skipped.append(rel_path)
                continue
        elif failed_record is not None:
            skipped.append(rel_path)
            continue

        if before_record is None:
            if path.exists():
                path.unlink()
                deleted.append(rel_path)
            continue

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(before_record.content)
        restored.append(rel_path)

    return RestoreResult(restored=restored, deleted=deleted, skipped=skipped)
