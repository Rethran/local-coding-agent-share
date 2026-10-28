"""Tests for deterministic focused file replacement."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.replace_in_file import replace_in_file


def _patch_workspace(monkeypatch: pytest.MonkeyPatch, root: Path, file: Path) -> None:
    monkeypatch.setattr("tools.replace_in_file.WORKSPACE_ROOT", root)
    monkeypatch.setattr("tools.replace_in_file.safe_path", lambda path: file)


def test_replace_in_file_replaces_one_unique_block_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    file = tmp_path / "module.py"
    file.write_text("before\nold block\nafter\n", encoding="utf-8")
    _patch_workspace(monkeypatch, tmp_path, file)

    result = replace_in_file("module.py", "old block", "new block")

    assert file.read_text(encoding="utf-8") == "before\nnew block\nafter\n"
    assert result == "Replaced text in: module.py"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize(
    ("content", "old_text", "expected_count"),
    [
        ("only one block", "missing", 0),
        ("same same", "same", 2),
    ],
)
def test_replace_in_file_rejects_non_unique_match(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content: str,
    old_text: str,
    expected_count: int,
) -> None:
    file = tmp_path / "module.py"
    file.write_text(content, encoding="utf-8")
    _patch_workspace(monkeypatch, tmp_path, file)

    with pytest.raises(ValueError, match=f"found {expected_count}"):
        replace_in_file("module.py", old_text, "replacement")

    assert file.read_text(encoding="utf-8") == content
