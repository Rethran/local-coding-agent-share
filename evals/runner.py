"""Fallback eval runner that does not require Inspect AI.

    python -m evals.runner --tags smoke --epochs 1 --max-samples 1
    python -m evals.runner --tags smoke --dry-run
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evals.build_dataset import DEFAULT_CASES, default_requirements_dir, default_workspace, load_cases
from evals.isolation import prepare_isolated_run, run_orchestrator
from evals.report import REPORTS_DIR, build_report, write_report
from evals.scorers import score_isolated_run


def _dry_summary(model_tag: str, prompt_variant: str) -> dict:
    return {
        "schema": "run-summary/v1",
        "model": model_tag,
        "promptVariant": prompt_variant,
        "totals": {
            "attempts": 0,
            "steps": 0,
            "toolCalls": 0,
            "toolErrors": 0,
            "repeatedIdenticalCalls": 0,
            "promptTokens": 0,
            "completionTokens": 0,
        },
        "attempts": [],
    }


def run_case(
    case: dict,
    *,
    prompt_variant: str,
    model_tag: str,
    timeout: int,
    epoch: int,
    dry_run: bool,
) -> dict:
    meta = case["metadata"]
    sample_id = case["id"]
    isolated = prepare_isolated_run(
        sample_id=f"{sample_id}-e{epoch}",
        task_id=meta["task_id"],
        base_commit=meta["base_commit"],
        repo=default_workspace(),
        source_requirements=default_requirements_dir(),
    )
    try:
        if dry_run:
            isolated.state_path.write_text(
                json.dumps({
                    "taskId": meta["task_id"],
                    "status": "FAILED",
                    "lastFailureReason": "dry_run",
                    "subtasks": [{"id": meta["subtask_id"], "status": "FAILED"}],
                }),
                encoding="utf-8",
            )
            isolated.run_summary_path.write_text(
                json.dumps(_dry_summary(model_tag, prompt_variant)),
                encoding="utf-8",
            )
            isolated.exit_code = 1
        else:
            run_orchestrator(
                isolated,
                subtask_id=meta["subtask_id"],
                model_tag=model_tag,
                prompt_variant=prompt_variant,
                timeout=timeout,
            )
        scores = score_isolated_run(
            isolated,
            subtask_id=meta["subtask_id"],
            repo=default_workspace(),
            accepted_commit=meta["accepted_commit"],
            reference_tests=list(meta.get("reference_tests") or []),
        )
        meta_scores = scores.pop("metadata")
        excerpt = isolated.to_dict()
        packed_scores = {}
        for name, row in scores.items():
            packed = {"ok": row["ok"], "explanation": row["explanation"]}
            if row.get("checks"):
                packed["checks"] = row["checks"]
            packed_scores[name] = packed
        return {
            "sampleId": sample_id,
            "epoch": epoch,
            "input": case["input"],
            "title": meta.get("title") or sample_id,
            "scores": packed_scores,
            "wallSeconds": isolated.wall_seconds,
            "tokens": meta_scores.get("tokens") or 0,
            "steps": (excerpt.get("totals") or {}).get("steps"),
            "totals": excerpt.get("totals") or {},
            "stopReasons": excerpt.get("stopReasons") or [],
            "timedOut": isolated.timed_out,
            "stopReason": "timeout" if isolated.timed_out else f"exit={isolated.exit_code}",
            "exitCode": isolated.exit_code,
        }
    finally:
        isolated.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TDD-coder eval cases without Inspect AI.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--tags", default="smoke")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--prompt-variant", default="baseline")
    parser.add_argument("--model", default="night-runner:latest")
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases), tags=args.tags)
    if args.max_samples:
        cases = cases[: args.max_samples]
    if not cases:
        print("No cases matched.")
        return 2

    trials = []
    for epoch in range(1, args.epochs + 1):
        for case in cases:
            print(f"=== {case['id']} epoch {epoch} ===")
            trials.append(
                run_case(
                    case,
                    prompt_variant=args.prompt_variant,
                    model_tag=args.model,
                    timeout=args.timeout,
                    epoch=epoch,
                    dry_run=args.dry_run,
                )
            )

    report = build_report(
        trials,
        header={
            "sourceLog": "evals.runner",
            "model": args.model,
            "promptVariant": args.prompt_variant,
            "runStatus": "success",
            "dryRun": bool(args.dry_run),
            "plannedEpochs": args.epochs,
            "plannedSamples": len(cases),
        },
    )
    stem = Path(args.out) if args.out else REPORTS_DIR / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{args.prompt_variant}"
        + ("-dry" if args.dry_run else "")
    )
    html_path, json_path = write_report(report, stem)
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
