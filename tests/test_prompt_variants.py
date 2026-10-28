"""Prompt variants: baseline is byte-identical, overrides apply per clause.

The snapshot fixture was captured from the pre-refactor ``system_prompt``
with ``LOCAL_AGENT_PROFILE=living-graph``. The snapshot test runs in a fresh
interpreter so that the profile-dependent module constants are rebuilt.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchestrator import prompt_variants
from orchestrator.prompt_variants import clause, list_variants, load_variant

AGENT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = AGENT_ROOT / "tests" / "fixtures" / "prompt_baseline_snapshot.json"

_RENDER = (
    "import json\n"
    "from orchestrator.prompts import system_prompt\n"
    "print(json.dumps({\n"
    "  'system_implementer_none': system_prompt(None, role='implementer'),\n"
    "  'system_implementer_sub': system_prompt('GRAPH-047.1', role='implementer'),\n"
    "  'system_test_writer_sub': system_prompt('GRAPH-047.1', role='test_writer'),\n"
    "}))\n"
)


def _render(env_override: dict[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "LOCAL_AGENT_PROMPT_VARIANT"}
    env.update(env_override)
    completed = subprocess.run(
        [sys.executable, "-c", _RENDER],
        cwd=AGENT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_baseline_variant_renders_identical_system_prompts() -> None:
    expected = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    actual = _render({"LOCAL_AGENT_PROFILE": "living-graph"})
    for key, text in expected.items():
        assert actual[key] == text, f"{key} drifted from baseline snapshot"


def test_variant_override_replaces_only_named_clause(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    variants_dir = tmp_path / "variants"
    variants_dir.mkdir()
    (variants_dir / "baseline.json").write_text("{}", encoding="utf-8")
    (variants_dir / "v2.json").write_text(
        json.dumps({"clauses": {"system.tdd": "Keep it simple for {subtask_id}. "}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_variants, "VARIANTS_DIR", variants_dir)
    load_variant.cache_clear()

    monkeypatch.setenv("LOCAL_AGENT_PROMPT_VARIANT", "v2")
    assert clause("system.tdd", "default", subtask_id="X-1.1") == "Keep it simple for X-1.1. "
    assert clause("system.intro", "default intro") == "default intro"
    assert clause("system.tdd", "default", subtask_id="X-1.1", unused="ignored").endswith("X-1.1. ")

    monkeypatch.setenv("LOCAL_AGENT_PROMPT_VARIANT", "baseline")
    assert clause("system.tdd", "default") == "default"
    load_variant.cache_clear()


def test_unknown_placeholder_is_left_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    variants_dir = tmp_path / "variants"
    variants_dir.mkdir()
    (variants_dir / "odd.json").write_text(
        json.dumps({"system.intro": "Literal {not_a_var} and {subtask_id}"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_variants, "VARIANTS_DIR", variants_dir)
    monkeypatch.setenv("LOCAL_AGENT_PROMPT_VARIANT", "odd")
    load_variant.cache_clear()
    assert clause("system.intro", "d", subtask_id="S") == "Literal {not_a_var} and S"
    load_variant.cache_clear()


def test_missing_variant_reports_available_names(monkeypatch: pytest.MonkeyPatch) -> None:
    load_variant.cache_clear()
    with pytest.raises(FileNotFoundError, match="baseline"):
        load_variant("does-not-exist")


def test_repo_ships_baseline_and_v2_variants() -> None:
    assert "baseline" in list_variants()
    assert "v2" in list_variants()
    load_variant.cache_clear()
    assert load_variant("baseline") == {}
    assert "system.tdd" in load_variant("v2")
