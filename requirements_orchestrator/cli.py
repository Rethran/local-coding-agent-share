"""Interactive CLI for generating and publishing requirement task graphs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

from requirements_orchestrator.decomposition import decompose_next_task
from requirements_orchestrator.elicitation import (
    elicit_next_question,
    first_missing_field,
    normalize_answer,
)
from requirements_orchestrator.engine import apply_decomposition_response
from requirements_orchestrator.ollama_client import StructuredOllamaClient
from requirements_orchestrator.publisher import PublishError, publish, slugify
from requirements_orchestrator.state import GenerationState, StateStore

InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m requirements_orchestrator",
        description="Turn an interactive project vision into local-agent task requirements.",
    )
    parser.add_argument("--resume", metavar="PROJECT_SLUG", default="")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--num-ctx", type=int, default=16384)
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--yes-publish", action="store_true")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).parent / "runs",
        help="Override resumable draft-state directory.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Repository receiving projects/ and requirements/ output.",
    )
    return parser


def _load_or_create(
    *,
    resume: str,
    runs_dir: Path,
) -> tuple[GenerationState, StateStore | None]:
    if not resume:
        return GenerationState(), None
    store = StateStore(runs_dir / resume)
    if not store.path.exists():
        raise FileNotFoundError(f"No resumable state found: {store.path}")
    return store.load(), store


def _elicit(
    state: GenerationState,
    store: StateStore | None,
    *,
    runs_dir: Path,
    client: StructuredOllamaClient,
    input_fn: InputFn,
    output_fn: OutputFn,
) -> StateStore:
    while first_missing_field(state) is not None:
        next_question = elicit_next_question(state, client)
        if next_question is None:
            break
        field_name, question = next_question
        while True:
            try:
                value = normalize_answer(field_name, input_fn(f"{question}\n> "))
            except ValueError as exc:
                output_fn(f"Invalid answer: {exc}")
                continue
            break
        state.fields[field_name] = value
        if store is None:
            if field_name != "projectName":
                raise RuntimeError("projectName must be the first elicited field.")
            state.run_id = slugify(str(value))
            store = StateStore(runs_dir / state.run_id)
            if store.path.exists():
                raise FileExistsError(
                    f"Run already exists. Resume it with --resume {state.run_id}"
                )
        store.save(state)
    if store is None:
        raise RuntimeError("Elicitation did not create a state store.")
    state.stage = "DECOMPOSING"
    store.save(state)
    return store


def _decompose(
    state: GenerationState,
    store: StateStore,
    *,
    client: StructuredOllamaClient,
    max_tasks: int,
    max_retries: int,
    output_fn: OutputFn,
) -> None:
    retries = 0
    while state.stage not in {"READY_TO_PUBLISH", "PUBLISHED"}:
        response = decompose_next_task(state, client)
        if response.get("status") == "task" and len(state.accepted) >= max_tasks:
            raise RuntimeError(
                f"Model proposed more than --max-tasks={max_tasks}; state remains resumable."
            )
        transition = apply_decomposition_response(state, response)
        store.save(state)
        if not transition.accepted:
            retries += 1
            output_fn("Proposal rejected:")
            for error in transition.errors:
                output_fn(f"  - {error}")
            if retries >= max_retries:
                raise RuntimeError(
                    f"Stopped after {max_retries} rejected proposals; resume after review."
                )
            continue
        retries = 0
        if transition.complete:
            output_fn(f"Decomposition complete with {len(state.accepted)} tasks.")
        else:
            task = state.accepted[-1]["task"]
            output_fn(f"Accepted {task['id']}: {task['summary']}")


def run(
    args: argparse.Namespace,
    *,
    client: StructuredOllamaClient | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> int:
    state, store = _load_or_create(resume=args.resume, runs_dir=args.runs_dir)
    model_client = client or StructuredOllamaClient(
        model=args.model,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
    )
    state.fields.setdefault("model", args.model)

    if state.stage == "PUBLISHED":
        output_fn(f"Already published as profile {state.published_profile}.")
        return 0
    if state.stage == "ELICITING":
        store = _elicit(
            state,
            store,
            runs_dir=args.runs_dir,
            client=model_client,
            input_fn=input_fn,
            output_fn=output_fn,
        )
    if store is None:
        raise RuntimeError("Missing state store.")
    if state.stage in {"DECOMPOSING", "VALIDATING"}:
        _decompose(
            state,
            store,
            client=model_client,
            max_tasks=args.max_tasks,
            max_retries=args.max_retries,
            output_fn=output_fn,
        )

    output_fn("Validated tasks:")
    for entry in state.accepted:
        task = entry["task"]
        output_fn(f"  {task['id']}: {task['summary']}")
    should_publish = args.yes_publish
    if not should_publish:
        answer = input_fn("Publish profile and requirements now? [y/N]\n> ").strip().lower()
        should_publish = answer in {"y", "yes"}
    if not should_publish:
        output_fn(f"Draft retained. Resume with --resume {state.run_id}")
        return 0

    result = publish(state, repo_root=args.repo_root)
    store.save(state)
    output_fn(f"Published profile: {result.profile_path}")
    output_fn(f"Published requirements: {result.requirements_dir}")
    output_fn("Run the code orchestrator:")
    output_fn(f'$env:LOCAL_AGENT_PROFILE = "{result.profile_name}"')
    output_fn(
        f"python main.py --from {result.first_task_id} --to {result.last_task_id}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, FileExistsError, PublishError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
