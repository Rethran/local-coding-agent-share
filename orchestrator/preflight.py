"""Preflight gate: context pack, golden rules, and conditional LLM planner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from tools import _tasks

from orchestrator.context_pack import build_context_pack
from orchestrator.plan_merge import merge_subtask_definition
from orchestrator.planner import load_plan_sidecar, plan_sidecar_path, run_llm_planner
from orchestrator.session import SessionStore
from orchestrator.spec_completer import complete_subtask_verification, persist_verification_repair
from orchestrator.spec_readiness import ReadinessResult, evaluate_spec_readiness


@dataclass(frozen=True)
class PreflightResult:
    ready: bool
    context_pack: dict[str, Any] | None = None
    merged_subtask: dict[str, Any] | None = None
    readiness: ReadinessResult | None = None
    blockers: list[str] | None = None
    plan_sidecar_path: str | None = None


def _persist_preflight_state(
    task_id: str,
    *,
    status: str,
    blockers: list[str] | None = None,
    sidecar_path: str | None = None,
) -> None:
    current = _tasks.load_task_state(task_id) or {}
    _tasks.update_task_state(
        task_id,
        current.get("status", "IN_PROGRESS"),
        preflight_status=status,
        preflight_blockers=blockers or [],
        plan_sidecar_path=sidecar_path,
    )


def run_preflight(
    task_id: str,
    subtask_id: str,
    *,
    args: argparse.Namespace,
    session: SessionStore | None = None,
) -> PreflightResult:
    del session  # reserved for future session-scoped planner hints
    max_rounds = max(getattr(args, "max_planner_rounds", 3), 1)
    enable_planner = bool(getattr(args, "enable_planner", False))
    skip_planner = not enable_planner or bool(getattr(args, "skip_planner", False))
    if getattr(args, "skip_planner", False) and enable_planner:
        print(
            f"Warning: both --enable-planner and --skip-planner set; "
            "planner remains disabled."
        )
        skip_planner = True
    planner_model = getattr(args, "planner_model", None) or args.model

    readiness: ReadinessResult | None = None
    sidecar_path: str | None = None

    for planner_round in range(1, max_rounds + 1):
        context_pack = build_context_pack(task_id, subtask_id)
        spec_subtask = _tasks.get_subtask_spec_definition(task_id, subtask_id)
        if spec_subtask is None:
            blockers = [f"Subtask not found: {subtask_id}"]
            _persist_preflight_state(task_id, status="FAILED", blockers=blockers)
            return PreflightResult(ready=False, blockers=blockers)

        sidecar = load_plan_sidecar(task_id, subtask_id)
        enrichments = (sidecar or {}).get("enrichments") if sidecar else None
        merged_subtask = merge_subtask_definition(spec_subtask, enrichments)
        readiness = evaluate_spec_readiness(
            task_id,
            subtask_id,
            context_pack,
            merged_subtask,
        )

        if not readiness.ready:
            packet = _tasks.get_task_packet(task_id)
            repaired = complete_subtask_verification(
                packet["task"],
                subtask_id,
                merged_subtask,
            )
            if repaired:
                persist_verification_repair(task_id, subtask_id, repaired)
                print(
                    f"{task_id}.{subtask_id}: spec completer updated verification"
                )
                continue

        if readiness.ready:
            sidecar_path = plan_sidecar_path(task_id, subtask_id) if sidecar else None
            _persist_preflight_state(
                task_id,
                status="READY",
                blockers=[],
                sidecar_path=sidecar_path,
            )
            print(f"{task_id}.{subtask_id}: preflight READY")
            return PreflightResult(
                ready=True,
                context_pack=context_pack,
                merged_subtask=merged_subtask,
                readiness=readiness,
                plan_sidecar_path=sidecar_path,
            )

        print(
            f"{task_id}.{subtask_id}: preflight not ready (round {planner_round}/{max_rounds})"
        )
        for blocker in readiness.blockers:
            print(f"  - {blocker}")

        if skip_planner:
            break

        try:
            run_llm_planner(
                task_id,
                subtask_id,
                context_pack,
                readiness,
                model=planner_model,
                planner_round=planner_round,
                num_ctx=getattr(args, "num_ctx", None),
            )
        except Exception as exc:  # noqa: BLE001 - planner failures are retryable
            print(f"{task_id}.{subtask_id}: planner round {planner_round} failed: {exc}")

    blockers = readiness.blockers if readiness else ["Preflight failed with unknown error."]
    sidecar_path = plan_sidecar_path(task_id, subtask_id)
    _persist_preflight_state(
        task_id,
        status="BLOCKED",
        blockers=blockers,
        sidecar_path=sidecar_path if load_plan_sidecar(task_id, subtask_id) else None,
    )
    _tasks.update_task_state(
        task_id,
        "BLOCKED",
        summary=f"Preflight blocked for {subtask_id}.",
        blockers=blockers,
        recommended_resume_step=(
            "Fix the task spec or planner sidecar enrichments, then rerun the orchestrator."
        ),
    )
    return PreflightResult(
        ready=False,
        readiness=readiness,
        blockers=blockers,
        plan_sidecar_path=sidecar_path,
    )
