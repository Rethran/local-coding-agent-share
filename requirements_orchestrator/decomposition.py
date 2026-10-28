"""One-task-at-a-time decomposition prompts and response checks."""

from __future__ import annotations

import json
from typing import Any

from requirements_orchestrator.dependencies import build_artifact_frontier
from requirements_orchestrator.ollama_client import StructuredOllamaClient, StructuredOutputError
from requirements_orchestrator.state import GenerationState
from requirements_orchestrator.validation import decomposition_response_schema


def _accepted_summaries(state: GenerationState) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for entry in state.accepted:
        task = entry.get("task") if isinstance(entry, dict) else None
        if isinstance(task, dict):
            summaries.append(
                {
                    "id": str(task.get("id") or ""),
                    "summary": str(task.get("summary") or ""),
                }
            )
    return summaries


def build_decomposition_context(state: GenerationState) -> dict[str, Any]:
    context: dict[str, Any] = {
        "vision": state.fields,
        "nextTaskId": (
            f"{state.fields['taskPrefix']}-{len(state.accepted) + 1:03d}"
        ),
        "acceptedTasks": _accepted_summaries(state),
        "uncoveredAcceptanceSignals": state.uncovered_acceptance_ids(),
        "artifactFrontier": build_artifact_frontier(state.accepted),
    }
    if state.last_rejected is not None:
        context["rejectedProposal"] = state.last_rejected
        context["validatorErrors"] = state.validation_errors
    return context


def decompose_next_task(
    state: GenerationState,
    client: StructuredOllamaClient,
) -> dict[str, Any]:
    context = build_decomposition_context(state)
    payload = client.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "You decompose a project vision into exactly one small, independently "
                    "verifiable coding task at a time. Return status='task' with exactly the "
                    "requested next task, or status='done' only when all listed acceptance "
                    "signals are covered. Never include dependsOn; software wires dependencies. "
                    "Use explicit subtasks in TDD-sized order. Each subtask verification command "
                    "must name its target test path or a focused selector. Include an explicit "
                    "negative constraint. relatedArtifacts must name earlier artifacts when the "
                    "new work consumes them. Keep scope and instructions short. For status='task', "
                    "reason must be empty. For status='done', task must be null and covers empty."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(context, indent=2, ensure_ascii=False),
            },
        ],
        schema=decomposition_response_schema(),
    )
    status = payload.get("status")
    if status == "task":
        if not isinstance(payload.get("task"), dict):
            raise StructuredOutputError("Task response must contain one task object.")
        if payload.get("reason"):
            raise StructuredOutputError("Task response reason must be empty.")
    elif status == "done":
        if payload.get("task") is not None or payload.get("covers"):
            raise StructuredOutputError("Done response must have task=null and covers=[].")
        if not str(payload.get("reason") or "").strip():
            raise StructuredOutputError("Done response must explain why decomposition is complete.")
    else:
        raise StructuredOutputError(f"Unknown decomposition status: {status!r}")
    return payload
