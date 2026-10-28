"""Integration tests for scout, implementer, verifier orchestration order."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from orchestrator.loop import (
    _duplicate_resume_step,
    _rollback_candidate_files,
    _run_orchestrator_loop,
    _unpreserve_test_paths,
)
from orchestrator.oracle_investigator import DuplicateRecovery
from orchestrator.verifier import VerifierResult
from orchestrator.worker import WorkerResult


def test_rollback_preserves_continuity_evidence() -> None:
    assert _rollback_candidate_files([
        "app/api/schedule.py",
        "tests/test_schedule.py",
        "local-coding-agent/recovery/continuity/HEAT-009/latest.json",
        r"local-coding-agent\recovery\continuity\HEAT-009\attempt-1.json",
    ]) == ["app/api/schedule.py"]


def test_rollback_unpreserves_declared_test_on_collection_error() -> None:
    changed = [
        "src/living_graph/python_extractor.py",
        "tests/test_python_extractor.py",
        "tests/fixtures/simple_library.py",
    ]
    rolled = _rollback_candidate_files(
        changed,
        unpreserve_paths=("tests/test_python_extractor.py",),
    )
    assert "src/living_graph/python_extractor.py" in rolled
    assert "tests/test_python_extractor.py" in rolled
    assert "tests/fixtures/simple_library.py" not in rolled


def test_unpreserve_test_paths_on_collection_output() -> None:
    merged = {"targetArtifacts": {"test": "tests/test_python_extractor.py"}}
    output = "Exit code: 2\nSyntaxError: invalid syntax\n"
    assert _unpreserve_test_paths(merged, verification_output=output) == (
        "tests/test_python_extractor.py",
    )
    assert _unpreserve_test_paths(merged, verification_output="Exit code: 1\nAssertionError") == ()


def test_generic_subtask_runs_scout_implementer_verifier_then_commit() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=1,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        role = kwargs.get("role") or "implementer"
        events.append("test_writer" if role == "test_writer" else "implementer")
        return WorkerResult(ok=True, state={"status": "IN_PROGRESS"})

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    def commit(*_args, **_kwargs):
        events.append("commit")

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "HEAT-001", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state") as mark_subtask,
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.build_context_pack", return_value={"explorationPolicy": {"preflightSatisfied": True}}),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=[]),
        patch("orchestrator.loop.auto_commit_completed_subtask", side_effect=commit),
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert events == ["scout", "test_writer", "scout", "implementer", "verifier", "commit"]
    mark_subtask.assert_called_once_with(
        "HEAT-001",
        "HEAT-001.1",
        "DONE",
        completion_kind="implemented",
    )


def test_already_green_test_writer_skips_implementer_and_marks_done() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=1,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        role = kwargs.get("role") or "implementer"
        events.append("test_writer" if role == "test_writer" else "implementer")
        return WorkerResult(ok=True, state={"status": "IN_PROGRESS"}, ready_for_verification=True)

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    def commit(*_args, **_kwargs):
        events.append("commit")

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "HEAT-001", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state") as mark_subtask,
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.build_context_pack", return_value={"explorationPolicy": {"preflightSatisfied": True}}),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=[]),
        patch("orchestrator.loop.auto_commit_completed_subtask", side_effect=commit),
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert events == ["scout", "test_writer", "verifier", "commit"]
    mark_subtask.assert_called_once_with(
        "HEAT-001",
        "HEAT-001.1",
        "DONE",
        summary="DUPLICATE: existing implementation already covers this subtask.",
        completion_kind="duplicate",
    )


def test_legacy_already_green_failure_is_treated_as_satisfied() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=1,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        events.append(kwargs.get("role") or "implementer")
        return WorkerResult(
            ok=False,
            state={
                "status": "FAILED",
                "lastFailureReason": (
                    "Declared verification passed during test-writer; "
                    "tests do not prove new behavior."
                ),
            },
        )

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    def commit(*_args, **_kwargs):
        events.append("commit")

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "HEAT-001", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state") as mark_subtask,
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.build_context_pack", return_value={"explorationPolicy": {"preflightSatisfied": True}}),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=[]),
        patch("orchestrator.loop.auto_commit_completed_subtask", side_effect=commit),
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert events == ["scout", "test_writer", "verifier", "commit"]
    mark_subtask.assert_called_once_with(
        "HEAT-001",
        "HEAT-001.1",
        "DONE",
        summary="DUPLICATE: existing implementation already covers this subtask.",
        completion_kind="duplicate",
    )


def test_already_green_without_contract_coverage_is_not_marked_done() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=1,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        events.append(kwargs.get("role") or "implementer")
        return WorkerResult(
            ok=True,
            state={"status": "IN_PROGRESS"},
            ready_for_verification=True,
        )

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            return_value={"taskId": "HEAT-001", "graph": {}},
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state") as mark_subtask,
        patch("orchestrator.loop._tasks.ensure_subtask_state", return_value=active),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch(
            "orchestrator.loop.duplicate_coverage_blockers",
            return_value=["HEAT-001.1 declared tests still pass when decide is stubbed (pytest tests/example.py)."],
        ),
        patch(
            "orchestrator.loop.auto_commit_completed_subtask",
            side_effect=lambda *_a, **_k: events.append("commit"),
        ),
        patch("orchestrator.loop.persist_duplicate_failure") as persist_failure,
        patch("orchestrator.loop.append_attempt_report") as attempt_report,
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=False),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 1
    assert events == ["scout", "test_writer", "verifier"]
    assert "NotImplementedError" in attempt_report.call_args.args[0]["likelyCause"]
    assert mark_subtask.call_args.args[:3] == ("HEAT-001", "HEAT-001.1", "FAILED")
    assert persist_failure.call_args.kwargs["blockers"] == [
        "HEAT-001.1 declared tests still pass when decide is stubbed (pytest tests/example.py)."
    ]
    assert persist_failure.call_args.kwargs["recommended_resume_step"].startswith(
        "Rewrite the declared test"
    )
    assert persist_failure.call_args.kwargs["attempt"] == 1


def test_unexecutable_duplicate_continues_after_developer_pass() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="CTX-022",
        to_id="CTX-022",
        model="night-runner:latest",
        max_tool_steps=10,
        max_subtask_attempts=1,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "CTX-022.3", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={
            "verification": ["python -m pytest tests/test_ctx_022_3_omit.py -q -k CTX_022_3"],
            "dataExamples": [{"omitted": ["source"]}],
            "targetArtifacts": {
                "module": "src/context_mgmt/events.py",
                "function": "to_stored_event",
            },
        },
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        events.append(kwargs.get("role") or "implementer")
        return WorkerResult(
            ok=True,
            state={"status": "IN_PROGRESS"},
            ready_for_verification=True,
        )

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "CTX-022", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state"),
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch(
            "orchestrator.loop.duplicate_coverage_blockers",
            return_value=[
                "CTX-022.CTX-022.3 dataExamples are not executable by the coordinator oracle."
            ],
        ),
        patch(
            "orchestrator.loop.recover_unexecutable_duplicate",
            return_value=DuplicateRecovery(
                passed=True,
                summary="Call the target with a payload that omits source.",
            ),
        ),
        patch(
            "orchestrator.loop.auto_commit_completed_subtask",
            side_effect=lambda *_a, **_k: events.append("commit"),
        ),
        patch("orchestrator.loop.persist_duplicate_failure") as persist_failure,
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert "commit" in events
    persist_failure.assert_not_called()


def test_duplicate_resume_step_names_a_missing_function() -> None:
    step = _duplicate_resume_step(
        [
            "CTX-024.CTX-024.2 cannot load load_license_state from src/context_mgmt/license.py",
            "CTX-024.CTX-024.2 cannot stub missing function load_license_state in src/context_mgmt/license.py.",
        ]
    )
    assert "missing from the module" in step
    assert "different function does not finish" in step

    stubbed = _duplicate_resume_step(
        ["HEAT-001.1 declared tests still pass when decide is stubbed (pytest tests/example.py)."]
    )
    assert stubbed.startswith("Rewrite the declared test")


def test_green_tests_missing_target_retry_before_failing() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="CTX-047",
        to_id="CTX-047",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=2,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "CTX-047.1", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={
            "verification": ["pytest tests/test_ctx_047_1_decide.py -q"],
            "targetArtifacts": {
                "module": "src/context_mgmt/guard.py",
                "test": "tests/test_ctx_047_1_decide.py",
                "function": "same_action",
            },
        },
    )

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        role = kwargs.get("role") or "implementer"
        events.append("test_writer" if role == "test_writer" else "implementer")
        if role == "test_writer" and events.count("test_writer") == 1:
            return WorkerResult(
                ok=True,
                state={"status": "IN_PROGRESS"},
                ready_for_verification=True,
            )
        if role == "test_writer":
            recovery = kwargs.get("recovery_context") or {}
            assert "same_action" in recovery.get("instruction", "")
            return WorkerResult(ok=True, state={"status": "IN_PROGRESS"})
        return WorkerResult(ok=True, state={"status": "IN_PROGRESS"})

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        if events.count("verifier") == 1:
            return VerifierResult(
                passed=False,
                evidence=(),
                blockers=(
                    "CTX-047.CTX-047.1 missing module file: src/context_mgmt/guard.py",
                ),
            )
        return VerifierResult(passed=True, evidence=())

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "CTX-047", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state"),
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.build_context_pack", return_value={"explorationPolicy": {"preflightSatisfied": True}}),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=[]),
        patch(
            "orchestrator.loop.build_failure_progress",
            return_value={"activeSubtaskId": "CTX-047.1"},
        ),
        patch(
            "orchestrator.loop.build_retry_context",
            return_value={"instruction": "rewrite the test to call same_action"},
        ),
        patch(
            "orchestrator.loop.auto_commit_completed_subtask",
            side_effect=lambda *_a, **_k: events.append("commit"),
        ),
        patch("orchestrator.loop.load_continuity_packet", return_value=None),
        patch("orchestrator.loop.load_recent_attempt_reports", return_value=[]),
        patch("orchestrator.loop.append_attempt_report") as attempt_report,
        patch("orchestrator.loop.persist_and_sync") as persist_sync,
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert events == [
        "scout",
        "test_writer",
        "verifier",
        "test_writer",
        "scout",
        "implementer",
        "verifier",
        "commit",
    ]
    assert persist_sync.call_args.kwargs["stop_reason"] == "missing_target"
    assert "same_action" in persist_sync.call_args.kwargs["recommended_resume_step"]
    assert "does not prove" in attempt_report.call_args.args[0]["likelyCause"] or (
        "without the target" in attempt_report.call_args.args[0]["likelyCause"]
    )


def test_unproven_duplicate_retries_then_implementer_can_finish() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=2,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    complete = {"allSubtasksDone": True, "activeSubtask": None}
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )
    missing = [
        "HEAT-001.HEAT-001.1 cannot load decide from app/domain/schedule.py",
        "HEAT-001.HEAT-001.1 cannot stub missing function decide in app/domain/schedule.py.",
    ]

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        role = kwargs.get("role") or "implementer"
        events.append("test_writer" if role == "test_writer" else "implementer")
        if role == "test_writer" and events.count("test_writer") == 1:
            return WorkerResult(
                ok=True,
                state={"status": "IN_PROGRESS"},
                ready_for_verification=True,
            )
        if role == "test_writer":
            assert kwargs.get("recovery_context") == {"instruction": "call the missing function"}
            return WorkerResult(ok=True, state={"status": "IN_PROGRESS"})
        return WorkerResult(ok=True, state={"status": "IN_PROGRESS"})

    def verifier(*_args, **_kwargs):
        events.append("verifier")
        return VerifierResult(passed=True, evidence=())

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            side_effect=[
                {"taskId": "HEAT-001", "graph": {}},
                {"taskId": None, "graph": {}},
            ],
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state"),
        patch(
            "orchestrator.loop._tasks.ensure_subtask_state",
            side_effect=[active, complete],
        ),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.build_context_pack", return_value={"explorationPolicy": {"preflightSatisfied": True}}),
        patch("orchestrator.loop.run_verifier", side_effect=verifier),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=missing),
        patch(
            "orchestrator.loop.build_failure_progress",
            return_value={"activeSubtaskId": "HEAT-001.1"},
        ),
        patch(
            "orchestrator.loop.build_retry_context",
            return_value={"instruction": "call the missing function"},
        ),
        patch(
            "orchestrator.loop.auto_commit_completed_subtask",
            side_effect=lambda *_a, **_k: events.append("commit"),
        ),
        patch("orchestrator.loop.load_continuity_packet", return_value=None),
        patch("orchestrator.loop.load_recent_attempt_reports", return_value=[]),
        patch("orchestrator.loop.append_attempt_report") as attempt_report,
        patch("orchestrator.loop.persist_duplicate_failure") as persist_failure,
        patch("orchestrator.loop._complete_task_if_ready", return_value={"status": "DONE"}),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=True),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 0
    assert events == [
        "scout",
        "test_writer",
        "verifier",
        "test_writer",
        "scout",
        "implementer",
        "verifier",
        "commit",
    ]
    assert persist_failure.call_args.kwargs["recommended_resume_step"].startswith(
        "The declared target function is missing"
    )
    assert "not in the module" in attempt_report.call_args.args[0]["likelyCause"]
    assert "commit" in events


def test_repeated_duplicate_blockers_stop_instead_of_retrying() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=3,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )
    blockers = [
        "HEAT-001.HEAT-001.1 cannot load decide from app/domain/schedule.py",
    ]
    packets = [
        None,
        {"stopReason": "duplicate_contract", "blockers": blockers},
    ]

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        events.append(kwargs.get("role") or "implementer")
        return WorkerResult(
            ok=True,
            state={"status": "IN_PROGRESS"},
            ready_for_verification=True,
        )

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            return_value={"taskId": "HEAT-001", "graph": {}},
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state"),
        patch("orchestrator.loop._tasks.ensure_subtask_state", return_value=active),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.run_verifier", return_value=VerifierResult(passed=True, evidence=())),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=blockers),
        patch("orchestrator.loop.load_continuity_packet", side_effect=packets),
        patch("orchestrator.loop.load_recent_attempt_reports", return_value=[]),
        patch("orchestrator.loop.append_attempt_report"),
        patch(
            "orchestrator.loop.build_failure_progress",
            return_value={"activeSubtaskId": "HEAT-001.1"},
        ),
        patch(
            "orchestrator.loop.build_retry_context",
            return_value={"instruction": "call the missing function"},
        ),
        patch("orchestrator.loop.persist_duplicate_failure"),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=False),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 1
    assert events == ["scout", "test_writer", "test_writer"]


def test_repeated_duplicate_in_attempt_report_stops_after_green_overwrite() -> None:
    events: list[str] = []
    args = Namespace(
        from_id="HEAT-001",
        to_id="HEAT-001",
        model="heat-agent:latest",
        max_tool_steps=10,
        max_subtask_attempts=3,
        max_attempts=None,
        hide_model_output=True,
        num_ctx=24576,
        num_predict=8192,
        context_threshold=0.85,
        max_context_handovers=3,
        no_auto_push=True,
        no_rollback_on_failure=False,
    )
    active = {
        "allSubtasksDone": False,
        "activeSubtask": {"id": "HEAT-001.1", "status": "PENDING"},
    }
    preflight = SimpleNamespace(
        ready=True,
        context_pack={"explorationPolicy": {"preflightSatisfied": True}},
        merged_subtask={"verification": ["pytest tests/example.py"]},
    )
    blockers = [
        "HEAT-001.HEAT-001.1 declared tests still pass when decide is stubbed (pytest tests/example.py).",
    ]

    def scout(*_args, **_kwargs):
        events.append("scout")
        return {
            "explorationPolicy": {"preflightSatisfied": True},
            "scoutBrief": {"findings": [], "parallel": False},
        }

    def worker(*_args, **kwargs):
        events.append(kwargs.get("role") or "implementer")
        return WorkerResult(
            ok=True,
            state={"status": "IN_PROGRESS"},
            ready_for_verification=True,
        )

    with (
        patch(
            "orchestrator.loop._tasks.select_next_task_id",
            return_value={"taskId": "HEAT-001", "graph": {}},
        ),
        patch("orchestrator.loop._tasks.update_task_state", return_value={}),
        patch("orchestrator.loop._tasks.update_subtask_state"),
        patch("orchestrator.loop._tasks.ensure_subtask_state", return_value=active),
        patch("orchestrator.loop.run_preflight", return_value=preflight),
        patch("orchestrator.loop.attach_scout_brief", side_effect=scout),
        patch("orchestrator.loop.capture_attempt_snapshot", return_value=MagicMock()),
        patch("orchestrator.loop.run_worker_for_subtask", side_effect=worker),
        patch("orchestrator.loop.run_verifier", return_value=VerifierResult(passed=True, evidence=())),
        patch("orchestrator.loop.duplicate_coverage_blockers", return_value=blockers),
        patch(
            "orchestrator.loop.load_continuity_packet",
            return_value={"stopReason": "already_satisfied", "blockers": []},
        ),
        patch(
            "orchestrator.loop.load_recent_attempt_reports",
            return_value=[
                {
                    "subtaskId": "HEAT-001.1",
                    "failureReason": "; ".join(blockers),
                }
            ],
        ),
        patch("orchestrator.loop.append_attempt_report"),
        patch("orchestrator.loop.persist_duplicate_failure"),
        patch("orchestrator.loop.finalize_task"),
        patch("orchestrator.loop.all_tasks_done", return_value=False),
    ):
        exit_code = _run_orchestrator_loop(args, MagicMock())

    assert exit_code == 1
    assert events == ["scout", "test_writer"]
