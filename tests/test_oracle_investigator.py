"""Investigator names a pass condition; the developer runs it."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.oracle_investigator import (
    develop_pass_condition,
    investigate_pass_condition,
    recover_unexecutable_duplicate,
)


OMITTED = {
    "targetArtifacts": {
        "module": "src/context_mgmt/events.py",
        "function": "to_stored_event",
    },
    "dataExamples": [{"omitted": ["tokens_potential_est", "source"]}],
}

KEEPS_OPTIONAL = '''
def to_stored_event(payload, salt):
    stored = {"org_id": payload["org_id"]}
    if "source" in payload:
        stored["source"] = payload["source"]
    if "tokens_potential_est" in payload:
        stored["tokens_potential_est"] = payload["tokens_potential_est"]
    return stored
'''

FORCES_OPTIONAL = '''
def to_stored_event(payload, salt):
    return {"org_id": payload["org_id"], "source": "gateway", "tokens_potential_est": 0}
'''


def test_investigator_names_omitted_keys_condition() -> None:
    condition = investigate_pass_condition(OMITTED)
    assert condition is not None
    assert condition.omitted == ("tokens_potential_est", "source")
    assert "absent from the stored result" in condition.summary


def test_investigator_ignores_a_normal_call_example() -> None:
    assert investigate_pass_condition({"dataExamples": [{"action": "none"}]}) is None


def test_developer_passes_when_omitted_keys_stay_out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "events.py"
    module_path.write_text(KEEPS_OPTIONAL, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    condition = investigate_pass_condition(OMITTED)
    assert condition is not None
    assert develop_pass_condition("CTX-022.CTX-022.3", OMITTED, condition) == []


def test_developer_rejects_forced_optional_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "events.py"
    module_path.write_text(FORCES_OPTIONAL, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    condition = investigate_pass_condition(OMITTED)
    assert condition is not None
    blockers = develop_pass_condition("CTX-022.CTX-022.3", OMITTED, condition)
    assert any("omitted key 'source'" in item for item in blockers)
    assert any("omitted key 'tokens_potential_est'" in item for item in blockers)


def test_stub_failure_is_not_recovered() -> None:
    recovery = recover_unexecutable_duplicate(
        "CTX-022",
        "CTX-022.3",
        OMITTED,
        ["CTX-022.CTX-022.3 declared tests still pass when to_stored_event is stubbed."],
    )
    assert recovery is None


def test_unknown_example_stays_blocked() -> None:
    recovery = recover_unexecutable_duplicate(
        "CTX-022",
        "CTX-022.3",
        {"dataExamples": [{"action": "none"}], "targetArtifacts": OMITTED["targetArtifacts"]},
        ["CTX-022.CTX-022.3 dataExamples are not executable by the coordinator oracle."],
    )
    assert recovery is not None
    assert recovery.passed is False
