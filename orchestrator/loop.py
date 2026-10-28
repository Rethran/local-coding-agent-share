from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import _recovery, _tasks
from tools._common import WORKSPACE_ROOT

from orchestrator.attempts import (
    all_tasks_done,
    append_attempt_report,
    build_attempt_report,
    build_failure_progress,
    build_retry_context,
    describe_restore_result,
    finalize_task,
    infer_likely_cause_and_fix,
    load_recent_attempt_reports,
    should_retry_task,
    update_failed_attempt_state,
)
from orchestrator.continuity import (
    load_continuity_packet,
    persist_and_sync,
    persist_duplicate_failure,
)
from orchestrator.context_pack import build_context_pack
from orchestrator.git_ops import (
    GitOperationError,
    auto_commit_completed_subtask,
    capture_attempt_snapshot,
    mark_auto_commit_failed,
)
from orchestrator.read_guard import normalize_relative_path
from orchestrator.python_write_gate import refuse_invalid_python_write
from orchestrator.log import configure_logging
from orchestrator.ollama_client import warmup_ollama_model
from orchestrator import run_metrics
from orchestrator.preflight import run_preflight
from orchestrator.project import PROJECT_PROFILE
from orchestrator.scout import attach_scout_brief
from orchestrator.session import SessionStore, set_session
from orchestrator.spec_readiness import is_subtask_done
from orchestrator.contract_gate import duplicate_coverage_blockers
from orchestrator.oracle_investigator import recover_unexecutable_duplicate
from orchestrator.verifier import (
    is_collection_or_syntax_error,
    run_verifier,
)
from orchestrator.worker import WorkerResult, run_worker_for_subtask

CONTINUITY_PATH_PREFIX = "local-coding-agent/recovery/continuity/"


def _target_test_path(merged_subtask: dict[str, Any] | None) -> str:
    target = (merged_subtask or {}).get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return ""
    return normalize_relative_path(str(target.get("test") or ""))


def _python_path_has_syntax_error(relative_path: str) -> bool:
    if not relative_path.lower().endswith(".py"):
        return False
    file_path = WORKSPACE_ROOT / relative_path
    try:
        content = file_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    return refuse_invalid_python_write(relative_path, content) is not None


def _unpreserve_test_paths(
    merged_subtask: dict[str, Any] | None,
    *,
    verification_output: str = "",
) -> tuple[str, ...]:
    test_path = _target_test_path(merged_subtask)
    if not test_path:
        return ()
    if verification_output and is_collection_or_syntax_error(verification_output):
        return (test_path,)
    if _python_path_has_syntax_error(test_path):
        return (test_path,)
    return ()


def _duplicate_failure_repeated(
    task_id: str,
    subtask_id: str,
    failure_reason: str,
    previous_packet: dict[str, Any],
) -> bool:
    """True when this gate failure already happened for the subtask.

    A later green pytest overwrites the continuity packet with already_satisfied,
    so the attempt report is the record that survives that overwrite.
    """
    if (
        previous_packet.get("stopReason") == "duplicate_contract"
        and "; ".join(str(item) for item in (previous_packet.get("blockers") or []))
        == failure_reason
    ):
        return True
    for report in load_recent_attempt_reports(task_id, limit=3):
        report_subtask = str(report.get("subtaskId") or report.get("activeSubtaskId") or "")
        if report_subtask == subtask_id and str(report.get("failureReason") or "") == failure_reason:
            return True
    return False


def _unproven_target_failure(blockers: list[str]) -> bool:
    blob = " ".join(blockers).lower()
    return "missing module file" in blob or "does not call" in blob


def _unproven_target_resume_step(
    merged_subtask: dict[str, Any] | None,
) -> str:
    target = (merged_subtask or {}).get("targetArtifacts") or {}
    function = "the declared function"
    module = "the declared module"
    if isinstance(target, dict):
        function = str(target.get("function") or function)
        module = str(target.get("module") or module)
    return (
        f"The declared tests passed without proving {function} in {module}. "
        f"Rewrite the declared test so it imports and calls {function}. "
        "The test must fail while that module or function is missing. "
        "Then rerun the declared command and stop."
    )


def _duplicate_resume_step(blockers: list[str]) -> str:
    blob = " ".join(blockers).lower()
    if "cannot load" in blob or "cannot stub missing function" in blob:
        return (
            "The declared target function is missing from the module. "
            "Rewrite the declared test so it calls that function and fails "
            "while the function is missing or raises NotImplementedError, "
            "then rerun the declared command and stop. "
            "A passing test of a different function does not finish this subtask."
        )
    if "still pass when" in blob or "stub" in blob:
        return (
            "Rewrite the declared test so it fails when the target function is "
            "replaced with NotImplementedError, then rerun the declared command."
        )
    if "dataexamples" in blob or "example " in blob or "oracle" in blob:
        return (
            "Make the implementation satisfy this subtask's dataExamples, or give "
            "the examples an executable input/output shape."
        )
    return (
        "Prove the already-green tests cover this subtask: dataExamples oracle "
        "must pass and the declared command must fail against a stubbed function."
    )


def _already_satisfied_failure(result: WorkerResult) -> bool:
    """True when a failed attempt already proved the declared tests pass."""
    state = result.state or {}
    blob = " ".join(
        str(state.get(key) or "")
        for key in ("lastFailureReason", "progressSummary", "summary")
    ).lower()
    return "do not prove new behavior" in blob or "tests_already_green" in blob


def _commit_verified_subtask(
    *,
    task_id: str,
    subtask_id: str,
    snapshot: Any,
    push: bool,
    completion_kind: str = "implemented",
    summary: str | None = None,
) -> None:
    done_summary = summary
    if completion_kind == "duplicate" and done_summary is None:
        done_summary = (
            "DUPLICATE: existing implementation already covers this subtask."
        )
    kwargs: dict[str, Any] = {"completion_kind": completion_kind}
    if done_summary is not None:
        kwargs["summary"] = done_summary
    _tasks.update_subtask_state(task_id, subtask_id, "DONE", **kwargs)
    label = "DUPLICATE" if completion_kind == "duplicate" else "DONE"
    print(f"{task_id}.{subtask_id}: verifier PASS; subtask {label}")
    auto_commit_completed_subtask(
        task_id,
        subtask_id,
        snapshot,
        push=push,
    )


def _rollback_candidate_files(
    changed_files: list[str],
    extra_preserve_prefixes: tuple[str, ...] = (),
    unpreserve_paths: tuple[str, ...] = (),
) -> list[str]:
    """Exclude continuity evidence and optional prefixes from attempt rollback."""
    preserved = (CONTINUITY_PATH_PREFIX, "tests/") + extra_preserve_prefixes
    unpreserve = {
        normalize_relative_path(path) for path in unpreserve_paths if path
    }
    candidates: list[str] = []
    for path in changed_files:
        normalized = normalize_relative_path(path)
        if normalized in unpreserve:
            candidates.append(path)
            continue
        if any(normalized.startswith(prefix) for prefix in preserved):
            continue
        candidates.append(path)
    return candidates


def _max_subtask_attempts(args: argparse.Namespace) -> int:
    if getattr(args, "max_subtask_attempts", None) is not None:
        return max(args.max_subtask_attempts, 1)
    if getattr(args, "max_attempts", None) is not None:
        print(
            "Warning: --max-attempts is deprecated; use --max-subtask-attempts instead."
        )
        return max(args.max_attempts, 1)
    return 5


def run_orchestrator(args: argparse.Namespace) -> int:
    configure_logging()
    print(f"Prompt: {args.prompt}")
    print(f"Model: {args.model}")
    print(f"Range: {args.from_id}..{args.to_id}")
    print(f"Max subtask attempts: {_max_subtask_attempts(args)}")
    print(f"Max planner rounds: {args.max_planner_rounds}")
    if args.skip_planner:
        print("Planner: skipped (golden rules only)")
    if args.no_auto_push:
        print("Git: auto-push disabled (commits only)")

    try:
        range_plan = _tasks.apply_cli_heat_range(args.from_id, args.to_id)
    except ValueError as exc:
        print(f"Invalid --from/--to: {exc}")
        return 2

    if range_plan.get("fromSubtaskId"):
        print(f"Pinned start subtask: {range_plan['fromSubtaskId']}")
    if range_plan.get("toSubtaskId"):
        print(f"Stop after subtask: {range_plan['toSubtaskId']} (later subtasks skipped)")

    session = SessionStore.create_and_set()
    print(f"Session: {session.run_id}")

    run_summary_path = getattr(args, "run_summary", None)
    metrics = None
    if run_summary_path:
        metrics = run_metrics.start_run(
            model=args.model,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
            from_id=args.from_id,
            to_id=args.to_id,
            workspace_root=WORKSPACE_ROOT,
            max_tool_steps=args.max_tool_steps,
            max_subtask_attempts=_max_subtask_attempts(args),
        )
        print(f"Run summary: {run_summary_path} (prompt variant {metrics.prompt_variant})")

    try:
        warmup_elapsed = warmup_ollama_model(model=args.model, num_ctx=args.num_ctx)
        print(f"Ollama warmup: {warmup_elapsed:.1f}s (num_ctx={args.num_ctx})")
    except Exception as exc:  # noqa: BLE001 - warmup failure should not abort the run
        print(f"Warning: Ollama warmup failed: {exc}")

    exit_code = 1
    try:
        exit_code = _run_orchestrator_loop(args, session)
        return exit_code
    finally:
        session.persist()
        set_session(None)
        if metrics is not None:
            metrics.finish(exit_code)
            metrics.write(Path(run_summary_path))
            run_metrics.clear()


def _complete_task_if_ready(task_id: str) -> dict[str, Any] | None:
    normalized = _tasks.ensure_subtask_state(task_id)
    if not normalized.get("allSubtasksDone"):
        return None
    return _tasks.update_task_state(
        task_id,
        "DONE",
        summary="All subtasks completed by orchestrator.",
    )


def _scout(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict[str, Any],
    *,
    base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context_pack = attach_scout_brief(
        base if base is not None else build_context_pack(task_id, subtask_id),
        merged_subtask,
    )
    scout_brief = context_pack.get("scoutBrief") or {}
    print(
        f"{task_id}.{subtask_id}: scout phase complete "
        f"({len(scout_brief.get('findings') or [])} findings, "
        f"parallel={bool(scout_brief.get('parallel'))})"
    )
    return context_pack


def _recover_failed_attempt(
    *,
    args: argparse.Namespace,
    task_id: str,
    subtask_id: str,
    attempt_number: int,
    max_subtask_attempts: int,
    result: WorkerResult,
    attempt_before_snapshot: Any,
    session: SessionStore,
    merged_subtask: dict[str, Any] | None,
    context_pack: dict[str, Any],
    verifier_failed: bool,
    verification_output: str = "",
) -> dict[str, Any] | None:
    after_failure_snapshot = _recovery.take_snapshot(WORKSPACE_ROOT)
    changed_files = _recovery.diff_snapshots(
        attempt_before_snapshot.files,
        after_failure_snapshot,
    )
    if verifier_failed:
        persist_and_sync(
            task_id=task_id,
            subtask_id=subtask_id,
            attempt=attempt_number,
            status="failed",
            merged_subtask=merged_subtask,
            session=session,
            stop_reason="verification_failed",
            blockers=list(result.state.get("blockers") or []),
            last_handoff_id=result.previous_handoff_id,
            changed_files=changed_files,
            recommended_resume_step=(
                "Fix the preserved implementation using verifier output, "
                "then rerun the exact declared verification command."
            ),
        )
    failure_progress = build_failure_progress(
        task_id=task_id,
        state=result.state,
        changed_files=changed_files,
        previous_handoff_id=result.previous_handoff_id,
    )
    failure_progress["activeSubtaskId"] = subtask_id

    restore_result = None
    unpreserve = _unpreserve_test_paths(
        merged_subtask,
        verification_output=verification_output,
    )
    if verifier_failed:
        rollback_files = _rollback_candidate_files(
            changed_files,
            unpreserve_paths=unpreserve,
        )
        rollback_files = [
            path
            for path in rollback_files
            if normalize_relative_path(path) in {
                normalize_relative_path(item) for item in unpreserve
            }
        ]
        if unpreserve and not args.no_rollback_on_failure and rollback_files:
            restore_result = _recovery.restore_changes(
                root=WORKSPACE_ROOT,
                before=attempt_before_snapshot.files,
                after_failure=after_failure_snapshot,
                changed_files=rollback_files,
            )
            recovery_action = (
                describe_restore_result(restore_result)
                + " Restored uncollectable declared test file after collection error."
            )
        else:
            recovery_action = (
                "Rollback skipped after verifier failure; implementation "
                "and continuity were preserved for the next attempt."
            )
    elif not args.no_rollback_on_failure:
        rollback_files = _rollback_candidate_files(
            changed_files,
            unpreserve_paths=unpreserve,
        )
        if rollback_files:
            restore_result = _recovery.restore_changes(
                root=WORKSPACE_ROOT,
                before=attempt_before_snapshot.files,
                after_failure=after_failure_snapshot,
                changed_files=rollback_files,
            )
            recovery_action = describe_restore_result(restore_result)
            protected_count = len(changed_files) - len(rollback_files)
            if protected_count:
                recovery_action += (
                    f" Preserved {protected_count} test or continuity artifact(s)."
                )
        else:
            recovery_action = describe_restore_result(restore_result)
    else:
        recovery_action = "Rollback skipped by configuration."
    updated_state = update_failed_attempt_state(
        task_id=task_id,
        state=result.state,
        failure_progress=failure_progress,
        recovery_action=recovery_action,
    )
    attempt_report = build_attempt_report(
        task_id=task_id,
        attempt_number=attempt_number,
        updated_state=updated_state,
        failure_progress=failure_progress,
        recovery_action=recovery_action,
        restore_result=restore_result,
    )
    attempt_report["subtaskId"] = subtask_id
    append_attempt_report(attempt_report)

    if updated_state.get("status") == "BLOCKED":
        finalize_task(task_id, updated_state)
        return None
    if not should_retry_task(
        updated_state.get("status", "FAILED"),
        attempt_number,
        max_subtask_attempts,
    ):
        finalize_task(task_id, updated_state)
        return None

    print(
        f"{task_id}.{subtask_id}: retrying after failed attempt "
        f"{attempt_number}. {recovery_action}"
    )
    return build_retry_context(
        task_id=task_id,
        failure_progress=failure_progress,
        recovery_action=recovery_action,
        session=session,
        include_cached_context=not bool(
            (context_pack.get("explorationPolicy") or {}).get("preflightSatisfied")
        ),
    )


def _run_orchestrator_loop(args: argparse.Namespace, session: SessionStore) -> int:
    max_subtask_attempts = _max_subtask_attempts(args)

    while True:
        selection = _tasks.select_next_task_id(args.from_id, args.to_id)
        graph = selection["graph"]
        task_id = selection["taskId"]

        if task_id is None:
            if all_tasks_done(graph):
                print(f"All selected {PROJECT_PROFILE.task_prefix} tasks are DONE.")
                return 0
            print("No active or READY task found.")
            print(json.dumps(graph, indent=2, ensure_ascii=False))
            return 1

        _tasks.update_task_state(
            task_id,
            "IN_PROGRESS",
            summary="Orchestrator processing task subtasks.",
        )

        while True:
            normalized = _tasks.ensure_subtask_state(task_id)
            active = normalized.get("activeSubtask")
            if normalized.get("allSubtasksDone") or active is None:
                break

            subtask_id = str(active["id"])
            if active.get("status") == "DONE":
                continue

            preflight = run_preflight(task_id, subtask_id, args=args, session=session)
            if not preflight.ready:
                finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                return 1

            context_pack = _scout(
                task_id,
                subtask_id,
                preflight.merged_subtask or {},
                base=preflight.context_pack or {},
            )

            recovery_context: dict[str, Any] | None = None
            subtask_before_snapshot = capture_attempt_snapshot()
            test_writer_done = False
            duplicate_committed = False
            for attempt_number in range(1, max_subtask_attempts + 1):
                attempt_before_snapshot = capture_attempt_snapshot()
                result = run_worker_for_subtask(
                    task_id,
                    subtask_id,
                    args.model,
                    args.max_tool_steps,
                    show_model_output=not args.hide_model_output,
                    attempt_number=attempt_number,
                    recovery_context=recovery_context,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    context_threshold=args.context_threshold,
                    max_context_handovers=args.max_context_handovers,
                    session=session,
                    context_pack=context_pack,
                    merged_subtask=preflight.merged_subtask,
                    role="test_writer",
                )
                already_green = bool(result.ok and result.ready_for_verification)
                if result.ok and not already_green:
                    print(
                        f"{task_id}.{subtask_id}: test-writer yielded red verification"
                    )
                    test_writer_done = True
                    break
                if not result.ok and not _already_satisfied_failure(result):
                    recovery_context = _recover_failed_attempt(
                        args=args,
                        task_id=task_id,
                        subtask_id=subtask_id,
                        attempt_number=attempt_number,
                        max_subtask_attempts=max_subtask_attempts,
                        result=result,
                        attempt_before_snapshot=attempt_before_snapshot,
                        session=session,
                        merged_subtask=preflight.merged_subtask,
                        context_pack=context_pack,
                        verifier_failed=False,
                    )
                    if recovery_context is None:
                        return 1
                    continue
                if already_green:
                    print(
                        f"{task_id}.{subtask_id}: test-writer verification already green; "
                        "running verifier"
                    )
                else:
                    print(
                        f"{task_id}.{subtask_id}: tests already pass on existing implementation; "
                        "running verifier"
                    )
                verification = run_verifier(
                    task_id,
                    subtask_id,
                    preflight.merged_subtask or {},
                )
                for evidence in verification.evidence:
                    print(
                        f"{task_id}.{subtask_id}: verifier "
                        f"{'PASS' if evidence.passed else 'FAIL'}: {evidence.command}"
                    )
                run_metrics.record_verifier(
                    task_id,
                    subtask_id,
                    passed=verification.passed,
                    commands=[
                        {"command": item.command, "passed": item.passed}
                        for item in verification.evidence
                    ],
                )
                if not verification.passed:
                    blockers = list(verification.blockers)
                    if not _unproven_target_failure(blockers):
                        _tasks.update_subtask_state(
                            task_id,
                            subtask_id,
                            "FAILED",
                            summary="Independent verifier rejected the already-green test.",
                        )
                        _tasks.update_task_state(
                            task_id,
                            "FAILED",
                            summary=f"Verifier failed for {subtask_id}.",
                            blockers=blockers,
                            last_failure_reason="; ".join(blockers),
                            recommended_resume_step=(
                                "Fix the verifier failure, rerun the declared command, "
                                "then mark the subtask DONE."
                            ),
                        )
                        finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                        return 1
                    resume_step = _unproven_target_resume_step(preflight.merged_subtask)
                    failure_reason = "; ".join(blockers)
                    likely_cause, possible_fix = infer_likely_cause_and_fix(failure_reason)
                    previous_packet = load_continuity_packet(task_id, subtask_id) or {}
                    repeated = _duplicate_failure_repeated(
                        task_id,
                        subtask_id,
                        failure_reason,
                        previous_packet,
                    )
                    print(
                        f"{task_id}.{subtask_id}: green tests do not prove the declared target"
                    )
                    _tasks.update_subtask_state(
                        task_id,
                        subtask_id,
                        "FAILED",
                        summary="Green tests do not prove the declared target.",
                    )
                    failed_state = _tasks.update_task_state(
                        task_id,
                        "FAILED",
                        summary=f"Verifier failed for {subtask_id}.",
                        blockers=blockers,
                        last_failure_reason=failure_reason,
                        recommended_resume_step=resume_step,
                    )
                    persist_and_sync(
                        task_id=task_id,
                        subtask_id=subtask_id,
                        attempt=attempt_number,
                        status="failed",
                        merged_subtask=preflight.merged_subtask,
                        session=session,
                        stop_reason="missing_target",
                        blockers=blockers,
                        last_handoff_id=result.previous_handoff_id,
                        recommended_resume_step=resume_step,
                    )
                    append_attempt_report(
                        {
                            "taskId": task_id,
                            "attempt": attempt_number,
                            "status": "FAILED",
                            "subtaskId": subtask_id,
                            "activeSubtaskId": subtask_id,
                            "failureReason": failure_reason,
                            "likelyCause": likely_cause,
                            "possibleFix": possible_fix,
                            "recoveryAction": resume_step,
                        }
                    )
                    if repeated or attempt_number >= max_subtask_attempts:
                        finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                        return 1
                    print(
                        f"{task_id}.{subtask_id}: declared target not proven; "
                        f"retrying test writer ({attempt_number}/{max_subtask_attempts})"
                    )
                    _tasks.update_subtask_state(
                        task_id,
                        subtask_id,
                        "IN_PROGRESS",
                        summary=resume_step,
                    )
                    _tasks.update_task_state(
                        task_id,
                        "IN_PROGRESS",
                        summary=f"Retrying {subtask_id} after tests that skip the declared target.",
                        blockers=blockers,
                        last_failure_reason=failure_reason,
                        recommended_resume_step=resume_step,
                    )
                    failure_progress = build_failure_progress(
                        task_id=task_id,
                        state=failed_state or {},
                        changed_files=list(
                            (failed_state or {}).get("changedFilesSinceAttemptStart") or []
                        ),
                        previous_handoff_id=result.previous_handoff_id,
                    )
                    failure_progress["activeSubtaskId"] = subtask_id
                    failure_progress["recommendedResumeStep"] = resume_step
                    failure_progress["blockers"] = blockers
                    failure_progress["lastFailureReason"] = failure_reason
                    failure_progress["likelyCause"] = likely_cause
                    failure_progress["possibleFix"] = possible_fix
                    recovery_context = build_retry_context(
                        task_id=task_id,
                        failure_progress=failure_progress,
                        recovery_action=resume_step,
                        session=session,
                        include_cached_context=not bool(
                            (context_pack.get("explorationPolicy") or {}).get(
                                "preflightSatisfied"
                            )
                        ),
                    )
                    continue
                coverage = duplicate_coverage_blockers(
                    task_id,
                    subtask_id,
                    preflight.merged_subtask or {},
                )
                if coverage:
                    recovery = recover_unexecutable_duplicate(
                        task_id,
                        subtask_id,
                        preflight.merged_subtask or {},
                        coverage,
                    )
                    if recovery is not None and recovery.passed:
                        coverage = []
                    elif recovery is not None:
                        coverage = list(recovery.blockers)
                if not coverage:
                    try:
                        _commit_verified_subtask(
                            task_id=task_id,
                            subtask_id=subtask_id,
                            snapshot=subtask_before_snapshot,
                            push=not args.no_auto_push,
                            completion_kind="duplicate",
                        )
                    except GitOperationError as exc:
                        message = str(exc)
                        print(f"{task_id}.{subtask_id}: auto commit/push failed")
                        print(message)
                        mark_auto_commit_failed(task_id, subtask_id, message)
                        finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                        return 1
                    duplicate_committed = True
                    break
                print(
                    f"{task_id}.{subtask_id}: already-green tests are not a proven duplicate"
                )
                resume_step = _duplicate_resume_step(coverage)
                failure_reason = "; ".join(coverage)
                likely_cause, possible_fix = infer_likely_cause_and_fix(failure_reason)
                previous_packet = load_continuity_packet(task_id, subtask_id) or {}
                repeated_blockers = _duplicate_failure_repeated(
                    task_id,
                    subtask_id,
                    failure_reason,
                    previous_packet,
                )
                _tasks.update_subtask_state(
                    task_id,
                    subtask_id,
                    "FAILED",
                    summary="Green tests are not a proven duplicate of this subtask.",
                )
                failed_state = _tasks.update_task_state(
                    task_id,
                    "FAILED",
                    summary=f"Duplicate contract gate failed for {subtask_id}.",
                    blockers=coverage,
                    last_failure_reason=failure_reason,
                    recommended_resume_step=resume_step,
                )
                persist_duplicate_failure(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    attempt=attempt_number,
                    merged_subtask=preflight.merged_subtask,
                    session=session,
                    blockers=coverage,
                    recommended_resume_step=resume_step,
                )
                append_attempt_report(
                    {
                        "taskId": task_id,
                        "attempt": attempt_number,
                        "status": "FAILED",
                        "subtaskId": subtask_id,
                        "activeSubtaskId": subtask_id,
                        "failureReason": failure_reason,
                        "likelyCause": likely_cause,
                        "possibleFix": possible_fix,
                        "recoveryAction": resume_step,
                    }
                )
                if repeated_blockers:
                    print(
                        f"{task_id}.{subtask_id}: duplicate contract repeated the "
                        "same blockers; not retrying"
                    )
                    finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                    return 1
                if attempt_number >= max_subtask_attempts:
                    finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                    return 1
                print(
                    f"{task_id}.{subtask_id}: duplicate contract not proven; "
                    f"retrying test writer ({attempt_number}/{max_subtask_attempts})"
                )
                _tasks.update_subtask_state(
                    task_id,
                    subtask_id,
                    "IN_PROGRESS",
                    summary=resume_step,
                )
                _tasks.update_task_state(
                    task_id,
                    "IN_PROGRESS",
                    summary=f"Retrying {subtask_id} after an unproven duplicate.",
                    blockers=coverage,
                    last_failure_reason=failure_reason,
                    recommended_resume_step=resume_step,
                )
                failure_progress = build_failure_progress(
                    task_id=task_id,
                    state=failed_state or {},
                    changed_files=list(
                        (failed_state or {}).get("changedFilesSinceAttemptStart") or []
                    ),
                    previous_handoff_id=result.previous_handoff_id,
                )
                failure_progress["activeSubtaskId"] = subtask_id
                failure_progress["recommendedResumeStep"] = resume_step
                failure_progress["blockers"] = coverage
                failure_progress["lastFailureReason"] = failure_reason
                failure_progress["likelyCause"] = likely_cause
                failure_progress["possibleFix"] = possible_fix
                recovery_context = build_retry_context(
                    task_id=task_id,
                    failure_progress=failure_progress,
                    recovery_action=resume_step,
                    session=session,
                    include_cached_context=not bool(
                        (context_pack.get("explorationPolicy") or {}).get(
                            "preflightSatisfied"
                        )
                    ),
                )
            else:
                finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                return 1
            if duplicate_committed:
                continue
            if not test_writer_done:
                finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                return 1

            context_pack = _scout(
                task_id,
                subtask_id,
                preflight.merged_subtask or {},
            )
            recovery_context = None
            for attempt_number in range(1, max_subtask_attempts + 1):
                attempt_before_snapshot = capture_attempt_snapshot()
                verifier_failed = False
                verification_output = ""
                result = run_worker_for_subtask(
                    task_id,
                    subtask_id,
                    args.model,
                    args.max_tool_steps,
                    show_model_output=not args.hide_model_output,
                    attempt_number=attempt_number,
                    recovery_context=recovery_context,
                    num_ctx=args.num_ctx,
                    num_predict=args.num_predict,
                    context_threshold=args.context_threshold,
                    max_context_handovers=args.max_context_handovers,
                    session=session,
                    context_pack=context_pack,
                    merged_subtask=preflight.merged_subtask,
                    role="implementer",
                )

                if result.ok or is_subtask_done(task_id, subtask_id):
                    verification = run_verifier(
                        task_id,
                        subtask_id,
                        preflight.merged_subtask or {},
                    )
                    verification_output = "\n".join(
                        item.output for item in verification.evidence
                    )
                    for evidence in verification.evidence:
                        print(
                            f"{task_id}.{subtask_id}: verifier "
                            f"{'PASS' if evidence.passed else 'FAIL'}: {evidence.command}"
                        )
                    run_metrics.record_verifier(
                        task_id,
                        subtask_id,
                        passed=verification.passed,
                        commands=[
                            {"command": item.command, "passed": item.passed}
                            for item in verification.evidence
                        ],
                    )
                    if verification.passed:
                        try:
                            _commit_verified_subtask(
                                task_id=task_id,
                                subtask_id=subtask_id,
                                snapshot=subtask_before_snapshot,
                                push=not args.no_auto_push,
                                completion_kind="implemented",
                            )
                        except GitOperationError as exc:
                            message = str(exc)
                            print(f"{task_id}.{subtask_id}: auto commit/push failed")
                            print(message)
                            mark_auto_commit_failed(task_id, subtask_id, message)
                            finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                            return 1
                        break

                    blockers = list(verification.blockers)
                    verifier_failed = True
                    _tasks.update_subtask_state(
                        task_id,
                        subtask_id,
                        "FAILED",
                        summary="Independent verifier rejected the implementation.",
                    )
                    failed_state = _tasks.update_task_state(
                        task_id,
                        "FAILED",
                        summary=f"Verifier failed for {subtask_id}.",
                        blockers=blockers,
                        last_failure_reason="; ".join(blockers),
                        recommended_resume_step=(
                            "Fix the verifier failure, rerun the declared command, "
                            "then mark the subtask DONE."
                        ),
                    )
                    result = WorkerResult(
                        ok=False,
                        state=failed_state,
                        previous_handoff_id=result.previous_handoff_id,
                    )

                recovery_context = _recover_failed_attempt(
                    args=args,
                    task_id=task_id,
                    subtask_id=subtask_id,
                    attempt_number=attempt_number,
                    max_subtask_attempts=max_subtask_attempts,
                    result=result,
                    attempt_before_snapshot=attempt_before_snapshot,
                    session=session,
                    merged_subtask=preflight.merged_subtask,
                    context_pack=context_pack,
                    verifier_failed=verifier_failed,
                    verification_output=verification_output,
                )
                if recovery_context is None:
                    return 1
            else:
                finalize_task(task_id, _tasks.load_task_state(task_id) or {})
                return 1

        completed_state = _complete_task_if_ready(task_id)
        if completed_state is None:
            finalize_task(task_id, _tasks.load_task_state(task_id) or {})
            return 1

        finalize_task(task_id, completed_state)
