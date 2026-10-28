"""Inspect AI task: dataset -> isolated TDD-coder solver -> code scorers.

The solver does not call Inspect's generate(). Ollama is invoked inside
`python main.py`. Pass `--model mockllm/model` so Inspect does not load a
second copy of the weights. Use `--max-samples 1` on a single GPU.

    inspect eval evals/tdd_coder_task.py -T tags=smoke -T prompt_variant=baseline --epochs 1 --max-samples 1 --max-connections 1 --model mockllm/model --log-dir evals/logs
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENT_ROOT))

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, stderr, scorer
from inspect_ai.solver import Generate, TaskState, solver

from evals.build_dataset import DEFAULT_CASES, default_requirements_dir, default_workspace, load_cases
from evals.isolation import IsolatedRun, load_json, prepare_isolated_run, run_orchestrator
from evals.scorers import score_process_budget, score_reference_tests, score_subtask_done

import evals.hooks  # noqa: E402,F401  registers the dashboard heartbeat + report hook


def _samples(tags: str) -> MemoryDataset:
    rows = load_cases(DEFAULT_CASES, tags=tags)
    return MemoryDataset(
        samples=[
            Sample(
                id=row["id"],
                input=row["input"],
                target=row.get("target") or "",
                metadata=row.get("metadata") or {},
            )
            for row in rows
        ],
        name="living-graph",
    )


def _result(state: TaskState) -> dict[str, Any]:
    return (state.metadata or {}).get("result") or {}


def _isolated_from_state(state: TaskState) -> IsolatedRun | None:
    payload = _result(state)
    if not payload.get("worktree"):
        return None
    return IsolatedRun(
        sample_id=str(payload.get("sampleId") or state.sample_id),
        worktree=Path(payload["worktree"]),
        requirements_dir=Path(payload["requirementsDir"]),
        runtime_dir=Path(payload["runtimeDir"]),
        repo=default_workspace(),
        run_summary_path=Path(payload["runSummaryPath"]),
        state_path=Path(payload["statePath"]),
        exit_code=payload.get("exitCode"),
        timed_out=bool(payload.get("timedOut")),
        wall_seconds=float(payload.get("wallSeconds") or 0),
        stdout_tail=str(payload.get("stdoutTail") or ""),
        stderr_tail=str(payload.get("stderrTail") or ""),
    )


@solver
def tdd_cli(
    prompt_variant: str = "baseline",
    model_tag: str = "night-runner:latest",
    timeout: int = 2400,
    dry_run: bool = False,
):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        meta = state.metadata or {}
        sample_id = str(state.sample_id)
        task_id = str(meta.get("task_id") or sample_id.rsplit(".", 1)[0])
        subtask_id = str(meta.get("subtask_id") or sample_id)
        isolated = await asyncio.to_thread(
            prepare_isolated_run,
            sample_id=sample_id,
            task_id=task_id,
            base_commit=str(meta.get("base_commit") or ""),
            repo=default_workspace(),
            source_requirements=default_requirements_dir(),
        )
        if dry_run:
            isolated.state_path.write_text(
                '{"taskId": "%s", "status": "FAILED", "lastFailureReason": "dry_run", '
                '"subtasks": [{"id": "%s", "status": "FAILED"}]}' % (task_id, subtask_id),
                encoding="utf-8",
            )
            isolated.run_summary_path.write_text(
                '{"schema": "run-summary/v1", "model": "%s", "promptVariant": "%s", '
                '"totals": {"attempts": 0, "steps": 0, "toolCalls": 0, "toolErrors": 0, '
                '"repeatedIdenticalCalls": 0, "promptTokens": 0, "completionTokens": 0}, '
                '"attempts": []}' % (model_tag, prompt_variant),
                encoding="utf-8",
            )
            isolated.exit_code = 1
        else:
            isolated = await asyncio.to_thread(
                run_orchestrator,
                isolated,
                subtask_id=subtask_id,
                model_tag=model_tag,
                prompt_variant=prompt_variant,
                timeout=timeout,
            )
        state.metadata = {
            **meta,
            "result": isolated.to_dict(),
            "promptVariant": prompt_variant,
            "modelTag": model_tag,
            "dryRun": dry_run,
        }
        completion = (
            f"exit={isolated.exit_code} timeout={isolated.timed_out} "
            f"wall={isolated.wall_seconds:.1f}s"
        )
        state.output = ModelOutput.from_content(model_tag, completion)
        return state

    return solve


async def _cleanup_sample(state: TaskState) -> None:
    isolated = _isolated_from_state(state)
    if isolated is not None:
        await asyncio.to_thread(isolated.cleanup)


@scorer(metrics=[accuracy(), stderr()])
def subtask_done():
    async def score(state: TaskState, target: Target) -> Score:
        isolated = _isolated_from_state(state)
        if isolated is None:
            return Score(value=INCORRECT, explanation="Solver did not produce an isolated run.")
        ok, why = score_subtask_done(
            load_json(isolated.state_path),
            str((state.metadata or {}).get("subtask_id") or state.sample_id),
        )
        if isolated.timed_out:
            ok, why = False, f"Timed out after {isolated.wall_seconds:.0f}s. {why}"
        return Score(value=CORRECT if ok else INCORRECT, explanation=why)

    return score


@scorer(metrics=[accuracy(), stderr()])
def reference_tests_pass():
    async def score(state: TaskState, target: Target) -> Score:
        isolated = _isolated_from_state(state)
        meta = state.metadata or {}
        if isolated is None:
            return Score(value=INCORRECT, explanation="Solver did not produce an isolated run.")
        ok, why = await asyncio.to_thread(
            score_reference_tests,
            isolated,
            repo=default_workspace(),
            accepted_commit=str(meta.get("accepted_commit") or ""),
            reference_tests=list(meta.get("reference_tests") or []),
        )
        return Score(value=CORRECT if ok else INCORRECT, explanation=why[-2000:])

    return score


@scorer(metrics=[accuracy(), stderr()])
def process_budget():
    async def score(state: TaskState, target: Target) -> Score:
        isolated = _isolated_from_state(state)
        if isolated is None:
            return Score(value=INCORRECT, explanation="Solver did not produce an isolated run.")
        ok, why, checks = score_process_budget(load_json(isolated.run_summary_path))
        return Score(
            value=CORRECT if ok else INCORRECT,
            explanation=why,
            metadata={"checks": checks},
        )

    return score


@task
def tdd_coder(
    prompt_variant: str = "baseline",
    model_tag: str = "night-runner:latest",
    tags: str = "smoke",
    timeout: int = 2400,
    epochs: int = 1,
    dry_run: bool = False,
):
    return Task(
        dataset=_samples(tags),
        solver=tdd_cli(
            prompt_variant=prompt_variant,
            model_tag=model_tag,
            timeout=timeout,
            dry_run=dry_run,
        ),
        scorer=[subtask_done(), reference_tests_pass(), process_budget()],
        cleanup=_cleanup_sample,
        epochs=epochs,
        fail_on_error=False,
        name="tdd_coder",
        metadata={
            "prompt_variant": prompt_variant,
            "model_tag": model_tag,
            "tags": tags,
        },
    )
