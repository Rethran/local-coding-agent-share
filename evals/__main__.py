"""python -m evals {build-dataset,report,run}"""

from __future__ import annotations

import sys


def _finalize(rest: list[str]) -> int:
    import argparse
    from pathlib import Path

    from evals.report import finalize_log

    parser = argparse.ArgumentParser(prog="python -m evals finalize")
    parser.add_argument("log", help="Inspect .eval log.")
    parser.add_argument("--pdf-always", action="store_true", help="Write a PDF even if the run did not finish.")
    parser.add_argument("--force", action="store_true", help="Rewrite even when the report is newer than the log.")
    args = parser.parse_args(rest)
    result = finalize_log(Path(args.log), pdf_always=args.pdf_always, force=args.force)
    if result.get("skipped"):
        print(f"up to date: {result['json']}")
    else:
        print(f"HTML: {result['html']}")
        print(f"JSON: {result['json']}")
        print(f"PDF: {result['pdf'] or 'not written (run not successful; use --pdf-always)'}")
        print(f"run status: {result.get('runStatus')} ({result.get('trials')} trials)")
    return 0


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("Usage: python -m evals {build-dataset|report|run|example|dashboard|finalize|index} [args...]")
        return 0
    command, rest = args[0], args[1:]
    if command in {"build-dataset", "build"}:
        from evals.build_dataset import main as build_main

        return build_main(rest)
    if command == "report":
        from evals.report import main as report_main

        return report_main(rest)
    if command == "run":
        from evals.runner import main as run_main

        return run_main(rest)
    if command == "example":
        from evals.example_run import main as example_main

        return example_main(rest)
    if command == "dashboard":
        from evals.dashboard import main as dashboard_main

        return dashboard_main(rest)
    if command == "finalize":
        return _finalize(rest)
    if command == "index":
        from evals.report import build_index

        print(f"Index: {build_index()}")
        return 0
    print(f"Unknown command: {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
