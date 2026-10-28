from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from requirements_orchestrator.cli import build_parser, run
from requirements_orchestrator.decomposition import build_decomposition_context
from requirements_orchestrator.dependencies import build_artifact_frontier
from requirements_orchestrator.elicitation import elicit_next_question, normalize_answer
from requirements_orchestrator.engine import apply_decomposition_response
from requirements_orchestrator.ollama_client import StructuredOllamaClient, StructuredOutputError
from requirements_orchestrator.publisher import PublishError, publish
from requirements_orchestrator.state import GenerationState, StateStore
from requirements_orchestrator.validation import validate_task


def make_fields(workspace: Path, *, acceptance_count: int = 2) -> dict[str, Any]:
    return {
        "model": "qwen3:8b",
        "projectName": "Demo Project",
        "taskPrefix": "DEMO",
        "workspaceRoot": str(workspace),
        "goal": "Build a small demo service.",
        "constraints": ["Keep functions pure."],
        "nonGoals": ["Do not add a UI."],
        "acceptanceSignals": [
            {"id": f"ACCEPT-{index:03d}", "text": f"Outcome {index}"}
            for index in range(1, acceptance_count + 1)
        ],
        "existingDependencies": ["Python"],
        "technicalContext": "Python package with pytest.",
        "implementationPrefixes": ["src/", "tests/"],
        "verificationConventions": "pytest one explicit test file with -q",
    }


def make_task(number: int, *, related: str | None = None) -> dict[str, Any]:
    task_id = f"DEMO-{number:03d}"
    stem = "app" if number == 1 else f"feature_{number}"
    subtask: dict[str, Any] = {
        "id": f"{task_id}.1",
        "title": f"Implement {stem}",
        "summary": f"Add the {stem} behavior with a focused test.",
        "readLegacy": False,
        "targetArtifacts": {
            "module": f"src/{stem}.py",
            "test": f"tests/test_{stem}.py",
        },
        "scope": [f"Implement src/{stem}.py", f"Test the {stem} behavior"],
        "verification": [f"pytest tests/test_{stem}.py -q"],
        "instructions": [
            "Write the focused test first.",
            f"Run pytest tests/test_{stem}.py -q.",
        ],
    }
    if related:
        subtask["relatedArtifacts"] = [
            {"path": related, "purpose": "Use behavior produced by the earlier task."}
        ]
    return {
        "id": task_id,
        "type": "task",
        "summary": f"Build {stem}",
        "description": f"Implement and verify the {stem} slice.",
        "scope": [f"Build {stem}"],
        "acceptanceCriteria": f"The {stem} focused test passes.",
        "constraints": ["Do not change unrelated behavior."],
        "verification": [f"pytest tests/test_{stem}.py -q"],
        "subtasks": [subtask],
    }


class FakeClient:
    def __init__(self, decompositions: list[dict[str, Any]] | None = None) -> None:
        self.decompositions = list(decompositions or [])

    def generate(self, *, messages: list[dict[str, str]], schema: dict[str, Any], **_: Any) -> dict[str, Any]:
        if "question" in schema.get("properties", {}):
            field = messages[-1]["content"].splitlines()[0].split(": ", 1)[1]
            return {"question": f"What value should be used for {field}?"}
        return self.decompositions.pop(0)


def test_state_store_resumes_and_syncs_drafts(tmp_path: Path) -> None:
    state = GenerationState(
        run_id="demo-project",
        stage="DECOMPOSING",
        fields=make_fields(tmp_path),
        accepted=[{"task": {**make_task(1), "dependsOn": []}, "covers": ["ACCEPT-001"]}],
    )
    store = StateStore(tmp_path / "runs" / "demo-project")
    store.save(state)

    loaded = store.load()

    assert loaded.run_id == "demo-project"
    assert loaded.accepted[0]["task"]["id"] == "DEMO-001"
    assert json.loads((store.drafts_dir / "DEMO-001.json").read_text())["dependsOn"] == []


def test_elicitation_tracks_field_in_code() -> None:
    state = GenerationState()

    field_name, question = elicit_next_question(state, FakeClient()) or ("", "")

    assert field_name == "projectName"
    assert "projectName" in question
    assert normalize_answer("taskPrefix", " demo ") == "DEMO"
    assert normalize_answer("acceptanceSignals", "works; fails safely") == [
        {"id": "ACCEPT-001", "text": "works"},
        {"id": "ACCEPT-002", "text": "fails safely"},
    ]


def test_structured_client_rejects_malformed_json() -> None:
    client = StructuredOllamaClient(
        chat_fn=lambda **_: SimpleNamespace(
            message=SimpleNamespace(content="not-json")
        )
    )

    with pytest.raises(StructuredOutputError, match="malformed JSON"):
        client.generate(
            messages=[{"role": "user", "content": "generate"}],
            schema={
                "type": "object",
                "required": ["value"],
                "properties": {"value": {"type": "string"}},
            },
        )


def test_acceptance_coverage_dependency_wiring_and_done_gate(tmp_path: Path) -> None:
    state = GenerationState(
        run_id="demo-project",
        stage="DECOMPOSING",
        fields=make_fields(tmp_path),
    )
    first = apply_decomposition_response(
        state,
        {"status": "task", "task": make_task(1), "covers": ["ACCEPT-001"], "reason": ""},
    )
    premature = apply_decomposition_response(
        state,
        {"status": "done", "task": None, "covers": [], "reason": "Done."},
    )
    second = apply_decomposition_response(
        state,
        {
            "status": "task",
            "task": make_task(2, related="src/app.py"),
            "covers": ["ACCEPT-002"],
            "reason": "",
        },
    )
    complete = apply_decomposition_response(
        state,
        {"status": "done", "task": None, "covers": [], "reason": "All outcomes covered."},
    )

    assert first.accepted
    assert not premature.accepted
    assert second.accepted
    assert state.accepted[1]["task"]["dependsOn"] == ["DEMO-001"]
    assert build_artifact_frontier(state.accepted)["src/feature_2.py"] == "DEMO-002"
    assert complete.complete
    assert state.stage == "READY_TO_PUBLISH"


def test_rejected_proposal_and_errors_are_preserved_for_retry(tmp_path: Path) -> None:
    state = GenerationState(
        stage="DECOMPOSING",
        fields=make_fields(tmp_path, acceptance_count=1),
    )
    rejected = {
        "status": "task",
        "task": make_task(1),
        "covers": [],
        "reason": "",
    }

    transition = apply_decomposition_response(state, rejected)
    context = build_decomposition_context(state)

    assert not transition.accepted
    assert context["rejectedProposal"] == rejected
    assert context["validatorErrors"] == transition.errors


def test_validator_rejects_broad_verification_and_missing_negative_constraint(tmp_path: Path) -> None:
    task = {**make_task(1), "dependsOn": []}
    task["constraints"] = ["Keep code clear."]
    task["subtasks"][0]["verification"] = ["pytest -q"]

    result = validate_task(
        task,
        prefix="DEMO",
        task_number=1,
        implementation_prefixes=["src/", "tests/"],
        acceptance_ids={"ACCEPT-001"},
        covers=["ACCEPT-001"],
        prior_task_ids=set(),
    )

    assert not result.valid
    assert any("negative constraint" in error for error in result.errors)
    assert any("must target its test artifact" in error for error in result.errors)


def test_validator_accepts_focused_non_pytest_runner(tmp_path: Path) -> None:
    task = make_task(1)
    task["dependsOn"] = []
    task["subtasks"][0]["targetArtifacts"] = {
        "module": "src/widget.ts",
        "test": "tests/widget.test.ts",
    }
    task["subtasks"][0]["verification"] = ["npm test -- tests/widget.test.ts"]
    task["verification"] = ["npm test -- tests/widget.test.ts"]

    result = validate_task(
        task,
        prefix="DEMO",
        task_number=1,
        implementation_prefixes=["src/", "tests/"],
        acceptance_ids={"ACCEPT-001"},
        covers=["ACCEPT-001"],
        prior_task_ids=set(),
    )

    assert result.valid, result.errors


def test_publish_writes_profile_and_tasks_without_generation_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    state = GenerationState(
        run_id="demo-project",
        stage="READY_TO_PUBLISH",
        fields=make_fields(workspace, acceptance_count=1),
        accepted=[
            {
                "task": {**make_task(1), "dependsOn": []},
                "covers": ["ACCEPT-001"],
            }
        ],
    )

    result = publish(state, repo_root=tmp_path)

    task_payload = json.loads((result.requirements_dir / "DEMO-001.json").read_text())
    profile_payload = json.loads(result.profile_path.read_text())
    assert "covers" not in task_payload
    assert profile_payload["taskPrefix"] == "DEMO"
    assert profile_payload["requirementsDir"] == str(result.requirements_dir.resolve())
    assert state.stage == "PUBLISHED"
    with pytest.raises(PublishError):
        publish(
            GenerationState(
                run_id="demo-project",
                stage="READY_TO_PUBLISH",
                fields=make_fields(workspace, acceptance_count=1),
                accepted=state.accepted,
            ),
            repo_root=tmp_path,
        )


def test_cli_runs_interactively_and_hands_off_to_code_orchestrator(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "agent"
    answers = iter(
        [
            "Demo Project",
            "DEMO",
            str(workspace),
            "Build a demo",
            "Keep it small",
            "Do not add UI",
            "The demo works",
            "Python",
            "Python and pytest",
            "src;tests",
            "pytest explicit test file",
        ]
    )
    outputs: list[str] = []
    client = FakeClient(
        [
            {
                "status": "task",
                "task": make_task(1),
                "covers": ["ACCEPT-001"],
                "reason": "",
            },
            {
                "status": "done",
                "task": None,
                "covers": [],
                "reason": "Acceptance is covered.",
            },
        ]
    )
    args = build_parser().parse_args(
        [
            "--runs-dir", str(tmp_path / "runs"),
            "--repo-root", str(repo_root),
            "--yes-publish",
        ]
    )

    code = run(
        args,
        client=client,
        input_fn=lambda _: next(answers),
        output_fn=outputs.append,
    )

    assert code == 0
    assert (repo_root / "projects" / "demo-project.json").exists()
    assert (repo_root / "requirements" / "demo-project" / "DEMO-001.json").exists()
    assert '$env:LOCAL_AGENT_PROFILE = "demo-project"' in outputs
    assert "python main.py --from DEMO-001 --to DEMO-001" in outputs
