from __future__ import annotations

import argparse
import sys

from orchestrator.config import (
    DEFAULT_AUTO_PUSH,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_FROM_ID,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_MAX_CONTEXT_HANDOVERS,
    DEFAULT_MAX_PLANNER_ROUNDS,
    DEFAULT_MAX_SUBTASK_ATTEMPTS,
    DEFAULT_MAX_TOOL_STEPS,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_TO_ID,
    PROJECT_NAME,
    TASK_PREFIX,
)


def _parse_create_bug(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python main.py create-bug",
        description=(
            "Create a HEAT bug ticket from a symptom. Function names are "
            "verified against the heating-app symbol table; callers are "
            "filled from blast radius. Does not start the worker."
        ),
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="Bug symptom in plain language.",
    )
    parser.add_argument(
        "--symbol",
        action="append",
        default=[],
        help="Known function name (repeatable). Still verified against app/.",
    )
    parser.add_argument(
        "--test",
        default="",
        help="Optional tests/ path for the reproduce subtask.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX)
    args = parser.parse_args(argv)
    args.command = "create-bug"
    return args


def _parse_run(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run {PROJECT_NAME} task orchestration with Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Range examples:\n"
            f"  python main.py --from {TASK_PREFIX}-003 --to {TASK_PREFIX}-004\n"
            f"  python main.py --from {TASK_PREFIX}-003.4 --to {TASK_PREFIX}-004\n"
            f"  python main.py --from {TASK_PREFIX}-003.4 --to {TASK_PREFIX}-003.6\n"
            "\n"
            "Create a bug ticket (no worker):\n"
            "  python main.py create-bug --summary \"tomorrow schedule not updating\"\n"
            "  python main.py create-bug --summary \"...\" --symbol build_schedule\n"
            "\n"
            f"Subtask refs ({TASK_PREFIX}-003.4) pin/stop work inside the parent task while\n"
            f"the numeric range still uses the parent ids ({TASK_PREFIX}-003..{TASK_PREFIX}-004).\n"
            "\n"
            "Each subtask runs preflight (context pack + golden rules) and one\n"
            "fresh worker session per coding attempt."
        ),
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help=(
            "High-level run description. Defaults to "
            "'Implement <from> through <to>' using the parsed range."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--from",
        dest="from_id",
        default=DEFAULT_FROM_ID,
        help=f"Start task or subtask ({TASK_PREFIX}-003 or {TASK_PREFIX}-003.4).",
    )
    parser.add_argument(
        "--to",
        dest="to_id",
        default=DEFAULT_TO_ID,
        help=f"End task or subtask ({TASK_PREFIX}-004 or {TASK_PREFIX}-003.6).",
    )
    parser.add_argument("--max-tool-steps", type=int, default=DEFAULT_MAX_TOOL_STEPS)
    parser.add_argument(
        "--max-subtask-attempts",
        type=int,
        default=DEFAULT_MAX_SUBTASK_ATTEMPTS,
        help="Maximum worker retries per subtask.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Deprecated alias for --max-subtask-attempts when that flag is omitted.",
    )
    parser.add_argument(
        "--max-planner-rounds",
        type=int,
        default=DEFAULT_MAX_PLANNER_ROUNDS,
        help="Maximum LLM planner rounds during preflight when golden rules fail.",
    )
    parser.add_argument(
        "--planner-model",
        default="",
        help="Model for the read-only preflight planner (defaults to --model).",
    )
    parser.add_argument(
        "--enable-planner",
        action="store_true",
        help="Run the LLM planner to enrich specs when golden rules fail (opt-in).",
    )
    parser.add_argument(
        "--skip-planner",
        action="store_true",
        help="Deprecated alias for the default behavior (do not run planner).",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=DEFAULT_NUM_CTX,
        help=(
            "Ollama num_ctx for inference and context-pressure handover detection "
            f"(default {DEFAULT_NUM_CTX})."
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help=(
            "Maximum generated tokens per Ollama response "
            f"(default {DEFAULT_NUM_PREDICT})."
        ),
    )
    parser.add_argument(
        "--context-threshold",
        type=float,
        default=DEFAULT_CONTEXT_THRESHOLD,
        help="Fraction of context budget that triggers a soft chat handover.",
    )
    parser.add_argument(
        "--max-context-handovers",
        type=int,
        default=DEFAULT_MAX_CONTEXT_HANDOVERS,
        help="Maximum mid-attempt context handovers per worker attempt.",
    )
    parser.add_argument(
        "--no-auto-push",
        action="store_true",
        default=not DEFAULT_AUTO_PUSH,
        help="Create subtask commits locally but do not push to the remote.",
    )
    parser.add_argument(
        "--auto-push",
        dest="no_auto_push",
        action="store_false",
        help="Push subtask commits even when the active project profile disables auto-push.",
    )
    parser.add_argument(
        "--no-rollback-on-failure",
        action="store_true",
        help="Do not restore task-changed files before retrying a failed attempt.",
    )
    parser.add_argument(
        "--hide-model-output",
        action="store_true",
        help="Hide visible assistant text emitted by the local model.",
    )
    parser.add_argument(
        "--run-summary",
        dest="run_summary",
        default="",
        metavar="PATH",
        help=(
            "Write a machine-readable run summary JSON (steps, tool calls, tokens, "
            "latency, stop reasons, verifier results) to PATH when the run ends."
        ),
    )
    args = parser.parse_args(argv)
    if not str(args.prompt or "").strip():
        args.prompt = f"Implement {args.from_id} through {args.to_id}"
    args.command = "run"
    return args


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "create-bug":
        return _parse_create_bug(args[1:])
    return _parse_run(args)
