from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from orchestrator import git_ops
from tools import _recovery, _tasks


def _init_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    readme = root / "README.md"
    readme.write_text("initial\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_parse_porcelain_paths_preserves_leading_index_space_status() -> None:
    # Git reports worktree-only modifications as " M path". Stripping stdout breaks this.
    output = " M app/domain/schedule.py\x00 M tests/test_schedule.py\x00\r\n"
    assert git_ops.parse_porcelain_paths(output) == {
        "app/domain/schedule.py",
        "tests/test_schedule.py",
    }


def test_parse_porcelain_paths_recovers_single_char_status_after_strip() -> None:
    corrupted = "M app/domain/schedule.py\x00"
    assert git_ops.parse_porcelain_paths(corrupted) == {"app/domain/schedule.py"}


def test_changed_paths_since_snapshot_detects_new_file() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _init_git_repo(root)
        with patch.object(_recovery, "WORKSPACE_ROOT", root), patch.object(
            git_ops, "WORKSPACE_ROOT", root
        ):
            snapshot = git_ops.capture_attempt_snapshot()
            (root / "app_example.py").write_text("print('ok')\n", encoding="utf-8")

            changed = git_ops.changed_paths_since_snapshot(snapshot)

            assert changed == ["app_example.py"]


def test_subtask_commit_subject_uses_spec_title() -> None:
    subject = git_ops.subtask_commit_subject("HEAT-005", "HEAT-005.2")
    assert subject == "HEAT-005.2: Implement select_cheapest_indices pure function"


def test_auto_commit_completed_subtask_creates_commit_without_push() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _init_git_repo(root)

        requirements_dir = root / "local-coding-agent" / "requirements"
        requirements_dir.mkdir(parents=True)
        task = json.loads(
            (_tasks.REQUIREMENTS_DIR / "HEAT-005.json").read_text(encoding="utf-8")
        )
        (requirements_dir / "HEAT-005.json").write_text(
            json.dumps(task, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        original_requirements = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            with patch.object(_recovery, "WORKSPACE_ROOT", root), patch.object(
                git_ops, "WORKSPACE_ROOT", root
            ):
                snapshot = git_ops.capture_attempt_snapshot()
                target = root / "app" / "domain" / "schedule.py"
                target.parent.mkdir(parents=True)
                target.write_text("DEFAULT_HEATING_SLOT_COUNT = 48\n", encoding="utf-8")

                git_ops.auto_commit_completed_subtask(
                    "HEAT-005",
                    "HEAT-005.1",
                    snapshot,
                    push=False,
                )

                log = subprocess.run(
                    ["git", "log", "-1", "--pretty=%s"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                assert log.stdout.strip() == (
                    "HEAT-005.1: Create schedule module skeleton only"
                )

                show = subprocess.run(
                    ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                committed_paths = {
                    line.strip()
                    for line in show.stdout.splitlines()
                    if line.strip()
                }
                assert committed_paths == {"app/domain/schedule.py"}
        finally:
            _tasks.REQUIREMENTS_DIR = original_requirements


def test_auto_commit_refuses_preexisting_dirty_overlap() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _init_git_repo(root)

        dirty_file = root / "already_dirty.py"
        dirty_file.write_text("v1\n", encoding="utf-8")

        with patch.object(_recovery, "WORKSPACE_ROOT", root), patch.object(
            git_ops, "WORKSPACE_ROOT", root
        ):
            snapshot = git_ops.capture_attempt_snapshot()
            dirty_file.write_text("v2\n", encoding="utf-8")

            with pytest.raises(git_ops.GitOperationError, match="already dirty"):
                git_ops.auto_commit_completed_subtask(
                    "HEAT-005",
                    "HEAT-005.1",
                    snapshot,
                    push=False,
                )


def test_auto_commit_ignores_preexisting_dirty_files_when_subtask_made_no_changes() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _init_git_repo(root)

        dirty_file = root / "already_dirty.py"
        dirty_file.write_text("unrelated\n", encoding="utf-8")

        with patch.object(_recovery, "WORKSPACE_ROOT", root), patch.object(
            git_ops, "WORKSPACE_ROOT", root
        ):
            snapshot = git_ops.capture_attempt_snapshot()
            git_ops.auto_commit_completed_subtask(
                "HEAT-005",
                "HEAT-005.1",
                snapshot,
                push=False,
            )

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            assert "already_dirty.py" in status.stdout
            log = subprocess.run(
                ["git", "log", "-1", "--pretty=%s"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            assert log.stdout.strip() == "Initial commit"


def test_auto_commit_no_changes_is_noop() -> None:
    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        _init_git_repo(root)

        with patch.object(_recovery, "WORKSPACE_ROOT", root), patch.object(
            git_ops, "WORKSPACE_ROOT", root
        ):
            snapshot = git_ops.capture_attempt_snapshot()
            git_ops.auto_commit_completed_subtask(
                "HEAT-005",
                "HEAT-005.1",
                snapshot,
                push=False,
            )

            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
            assert status.stdout.strip() == ""
