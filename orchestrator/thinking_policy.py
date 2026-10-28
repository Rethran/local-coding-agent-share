"""Balanced per-turn thinking-effort routing for Ollama agents."""

from __future__ import annotations

from typing import Literal, Protocol, Sequence

from orchestrator.config import READ_ONLY_TOOLS

ThinkingEffort = Literal["low", "medium", "high"]

INITIAL_WORKER_EFFORT: ThinkingEffort = "medium"
PLANNER_EFFORT: ThinkingEffort = "high"
RECOVERY_EFFORT: ThinkingEffort = "medium"
FAILURE_EFFORT: ThinkingEffort = "high"


class ToolOutcome(Protocol):
    name: str
    status: str
    raw_result: str


def effort_after_tool_results(results: Sequence[ToolOutcome]) -> ThinkingEffort:
    """Choose the effort for interpreting the latest tool-result batch."""
    if not results:
        return RECOVERY_EFFORT
    if any(result.status != "ok" for result in results):
        expected_read_refusal = all(
            result.name in READ_ONLY_TOOLS
            and any(
                marker in result.raw_result
                for marker in (
                    "contextPack.referenceSnippets already includes",
                    "Duplicate read_file",
                    "Repeated read_file",
                )
            )
            for result in results
            if result.status != "ok"
        )
        if expected_read_refusal:
            return "medium"
        return FAILURE_EFFORT
    if any(
        result.name == "run_command"
        and not result.raw_result.lstrip().startswith("Exit code: 0")
        for result in results
    ):
        return FAILURE_EFFORT
    if all(result.name in READ_ONLY_TOOLS for result in results):
        return "medium"
    return "medium"
