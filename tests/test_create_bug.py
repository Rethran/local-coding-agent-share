"""Bug-ticket CLI: symbol guard, ticket shape, and create-bug dispatch."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest

AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.bug_ticket import (  # noqa: E402
    BugTicketError,
    build_bug_task,
    collect_verified_symbols,
    create_bug_ticket,
    guard_candidates,
    next_heat_id,
    propose_symbol_names,
    resolve_candidate,
    run_create_bug,
)
from orchestrator.cli import parse_args  # noqa: E402
from orchestrator.context_pack import build_repo_context  # noqa: E402
from orchestrator.spec_readiness import evaluate_spec_readiness  # noqa: E402
from tools import _tasks  # noqa: E402

SAMPLE_MAP = {
    "symbols": [
        {
            "symbol_id": "app/domain/schedule.py:build_schedule:60",
            "name": "build_schedule",
            "file": "app/domain/schedule.py",
            "signature": "def build_schedule(price_slots, selected_count=48)",
            "start_line": 60,
            "end_line": 90,
            "score": 0.25,
            "churn": 0,
        },
        {
            "symbol_id": "app/services/schedule_store.py:ScheduleStore.replace:10",
            "name": "ScheduleStore.replace",
            "file": "app/services/schedule_store.py",
            "signature": "def replace(self, schedule)",
            "start_line": 10,
            "end_line": 20,
            "score": 0.15,
            "churn": 0,
        },
        {
            "symbol_id": "app/domain/prices.py:parse_entsoe_prices:1",
            "name": "parse_entsoe_prices",
            "file": "app/domain/prices.py",
            "signature": "def parse_entsoe_prices(xml)",
            "start_line": 1,
            "end_line": 20,
            "score": 0.1,
            "churn": 0,
        },
        {
            "symbol_id": "app/other.py:Helper.get:1",
            "name": "Helper.get",
            "file": "app/other.py",
            "signature": "def get(self)",
            "start_line": 1,
            "end_line": 2,
            "score": 0.05,
            "churn": 0,
        },
        {
            "symbol_id": "app/alt.py:Other.get:1",
            "name": "Other.get",
            "file": "app/alt.py",
            "signature": "def get(self)",
            "start_line": 1,
            "end_line": 2,
            "score": 0.04,
            "churn": 0,
        },
    ],
    "edges": [
        [
            "app/services/schedule_store.py:ScheduleStore.replace:10",
            "app/domain/schedule.py:build_schedule:60",
        ],
    ],
}


@pytest.fixture
def isolated_requirements():
    with TemporaryDirectory() as temp_dir:
        requirements_dir = Path(temp_dir)
        original = _tasks.REQUIREMENTS_DIR
        _tasks.REQUIREMENTS_DIR = requirements_dir
        try:
            yield requirements_dir
        finally:
            _tasks.REQUIREMENTS_DIR = original


def test_parse_args_run_defaults_unchanged() -> None:
    args = parse_args([])
    assert args.command == "run"
    assert args.from_id == "HEAT-001"
    assert args.to_id == "HEAT-005"
    assert args.prompt == "Implement HEAT-001 through HEAT-005"


def test_parse_args_prompt_follows_from_to_range() -> None:
    args = parse_args(
        [
            "--model",
            "night-runner:latest",
            "--from",
            "CTX-005",
            "--to",
            "CTX-010",
            "--auto-push",
        ]
    )
    assert args.prompt == "Implement CTX-005 through CTX-010"


def test_parse_args_keeps_explicit_prompt() -> None:
    args = parse_args(["Ship only the storage layer", "--from", "CTX-005", "--to", "CTX-010"])
    assert args.prompt == "Ship only the storage layer"


def test_parse_args_create_bug() -> None:
    args = parse_args(
        ["create-bug", "--summary", "schedule stuck", "--symbol", "build_schedule"]
    )
    assert args.command == "create-bug"
    assert args.summary == "schedule stuck"
    assert args.symbol == ["build_schedule"]


def test_guard_exact_and_unique_simple_name() -> None:
    exact = resolve_candidate("build_schedule", SAMPLE_MAP["symbols"])
    assert [item["name"] for item in exact] == ["build_schedule"]

    unique = resolve_candidate("replace", SAMPLE_MAP["symbols"])
    assert [item["name"] for item in unique] == ["ScheduleStore.replace"]


def test_guard_rejects_unknown_and_ambiguous_simple_name() -> None:
    assert resolve_candidate("does_not_exist", SAMPLE_MAP["symbols"]) == []
    assert resolve_candidate("get", SAMPLE_MAP["symbols"]) == []


def test_guard_drops_hallucinated_llm_names() -> None:
    verified = guard_candidates(
        ["not_a_real_function", "build_schedule", "invented"],
        SAMPLE_MAP,
    )
    assert [item["name"] for item in verified] == ["build_schedule"]


def test_collect_verified_symbols_uses_llm_then_guard() -> None:
    with patch(
        "orchestrator.bug_ticket.propose_symbol_names",
        return_value=["hallucinated", "build_schedule"],
    ) as propose:
        verified = collect_verified_symbols(
            "schedule not updating",
            [],
            SAMPLE_MAP,
            model="heat-agent:latest",
            num_ctx=2048,
        )
    propose.assert_called_once()
    assert [item["name"] for item in verified] == ["build_schedule"]


def test_collect_verified_symbols_aborts_when_nothing_survives() -> None:
    with pytest.raises(BugTicketError, match="No verified"):
        collect_verified_symbols(
            "mystery",
            ["nope"],
            SAMPLE_MAP,
            model="heat-agent:latest",
            num_ctx=2048,
        )


def test_propose_symbol_names_parses_json_array() -> None:
    response = SimpleNamespace(message={"content": '["build_schedule", "nope"]'})
    with patch("orchestrator.bug_ticket.ollama_chat", return_value=response):
        names = propose_symbol_names(
            "schedule",
            SAMPLE_MAP,
            model="heat-agent:latest",
            num_ctx=1024,
        )
    assert names == ["build_schedule", "nope"]


def test_next_heat_id_increments(isolated_requirements: Path) -> None:
    (isolated_requirements / "HEAT-041.json").write_text(
        json.dumps({"id": "HEAT-041", "summary": "prior"}),
        encoding="utf-8",
    )
    assert next_heat_id() == "HEAT-042"


def test_create_bug_ticket_writes_ready_spec(isolated_requirements: Path) -> None:
    result = create_bug_ticket(
        summary="tomorrow schedule not updating",
        symbols=["build_schedule"],
        repo_map=SAMPLE_MAP,
    )
    assert result["taskId"] == "HEAT-001"
    task = json.loads((isolated_requirements / "HEAT-001.json").read_text(encoding="utf-8"))
    assert task["dependsOn"] == []
    assert task["type"] == "bug"
    assert "legacySymbols" not in task["subtasks"][0]
    assert task["subtasks"][0]["readLegacy"] is False
    assert task["subtasks"][0]["targetArtifacts"]["function"] == "build_schedule"
    assert task["subtasks"][0]["targetArtifacts"]["module"] == "app/domain/schedule.py"
    related_names = {item["name"] for item in task["subtasks"][0]["relatedArtifacts"]}
    assert "ScheduleStore.replace" in related_names
    assert (isolated_requirements / "HEAT-001.state.json").exists()

    for suffix in (".1", ".2"):
        subtask_id = f"HEAT-001{suffix}"
        subtask = _tasks.get_subtask_spec_definition("HEAT-001", subtask_id)
        assert subtask is not None
        readiness = evaluate_spec_readiness(
            "HEAT-001",
            subtask_id,
            {"subtaskContext": {"legacySlices": []}},
            subtask,
        )
        assert readiness.ready, readiness.blockers


def test_run_create_bug_prints_id(isolated_requirements: Path, capsys) -> None:
    args = parse_args(
        ["create-bug", "--summary", "schedule stuck", "--symbol", "build_schedule"]
    )
    with patch("orchestrator.bug_ticket.get_heating_repo_map", return_value=SAMPLE_MAP):
        code = run_create_bug(args)
    assert code == 0
    output = capsys.readouterr().out
    assert "Created HEAT-001" in output
    assert "build_schedule" in output


def test_run_create_bug_refuses_unverified_symbol(isolated_requirements: Path) -> None:
    args = parse_args(["create-bug", "--summary", "mystery", "--symbol", "nope"])
    with patch("orchestrator.bug_ticket.get_heating_repo_map", return_value=SAMPLE_MAP):
        code = run_create_bug(args)
    assert code == 2
    assert list(isolated_requirements.glob("HEAT-*.json")) == []


def test_build_repo_context_from_spec_names() -> None:
    subtask = {
        "targetArtifacts": {
            "function": "build_schedule",
            "module": "app/domain/schedule.py",
        },
        "relatedArtifacts": [
            {
                "name": "ScheduleStore.replace",
                "path": "app/services/schedule_store.py",
            }
        ],
    }
    result = build_repo_context(subtask, repo_map=SAMPLE_MAP)
    names = {item["name"] for item in result["matches"]}
    assert "build_schedule" in names
    build_match = next(item for item in result["matches"] if item["name"] == "build_schedule")
    assert "ScheduleStore.replace" in build_match["blast_radius"]
    assert result["token_budget_used"] > 0


def test_build_bug_task_uses_explicit_test_path() -> None:
    task = build_bug_task(
        task_id="HEAT-099",
        summary="vat missing",
        verified=[SAMPLE_MAP["symbols"][2]],
        repo_map=SAMPLE_MAP,
        test_path="tests/test_prices.py",
    )
    assert task["subtasks"][0]["targetArtifacts"]["test"] == "tests/test_prices.py"
    assert "pytest tests/test_prices.py -v" in task["subtasks"][0]["verification"]
