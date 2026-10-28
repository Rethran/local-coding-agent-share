"""CLI orchestrator for running HEAT tasks outside Continue chat.

The orchestrator keeps context small by starting a fresh Ollama conversation
per task and writing every full tool result to a handoff record. The model only
receives compact summaries of those results.
"""

from __future__ import annotations

import sys

from orchestrator.bug_ticket import run_create_bug
from orchestrator.cli import parse_args
from orchestrator.loop import run_orchestrator


def main() -> int:
    args = parse_args()
    if getattr(args, "command", "run") == "create-bug":
        return run_create_bug(args)
    return run_orchestrator(args)


if __name__ == "__main__":
    sys.exit(main())
