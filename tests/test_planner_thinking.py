"""Tests for planner thinking-effort configuration."""

from types import SimpleNamespace
from unittest.mock import patch

from orchestrator.planner import run_llm_planner
from orchestrator.spec_readiness import ReadinessResult


def test_planner_requests_high_thinking_effort() -> None:
    response = SimpleNamespace(
        message={
            "role": "assistant",
            "content": '{"enrichments": {"instructions": ["inspect"]}}',
        },
    )
    readiness = ReadinessResult(
        ready=False,
        blockers=["missing instructions"],
        planner_hints=[],
        rule_ids=["GR-1"],
    )

    with (
        patch("orchestrator.planner.ollama_chat", return_value=response) as chat,
        patch("orchestrator.planner.save_plan_sidecar", return_value="plan.json"),
    ):
        payload = run_llm_planner(
            "HEAT-009",
            "HEAT-009.2",
            {},
            readiness,
            model="heat-agent:latest",
            planner_round=1,
        )

    assert payload["enrichments"]["instructions"] == ["inspect"]
    assert chat.call_args.kwargs["think"] == "high"
