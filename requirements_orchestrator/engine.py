"""Deterministic state transitions around model decomposition proposals."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from requirements_orchestrator.dependencies import wire_dependencies
from requirements_orchestrator.state import GenerationState
from requirements_orchestrator.validation import validate_task


@dataclass(frozen=True)
class TransitionResult:
    accepted: bool
    complete: bool
    errors: list[str]


def apply_decomposition_response(
    state: GenerationState,
    response: dict[str, Any],
) -> TransitionResult:
    if response.get("status") == "done":
        uncovered = state.uncovered_acceptance_ids()
        if uncovered:
            errors = [f"done rejected: uncovered acceptance signals {uncovered}"]
            state.stage = "VALIDATING"
            state.last_rejected = copy.deepcopy(response)
            state.validation_errors = errors
            return TransitionResult(False, False, errors)
        if not state.accepted:
            errors = ["done rejected: at least one task is required"]
            state.stage = "VALIDATING"
            state.last_rejected = copy.deepcopy(response)
            state.validation_errors = errors
            return TransitionResult(False, False, errors)
        state.stage = "READY_TO_PUBLISH"
        state.last_rejected = None
        state.validation_errors = []
        return TransitionResult(True, True, [])

    proposal = copy.deepcopy(response.get("task"))
    if not isinstance(proposal, dict):
        errors = ["proposal must contain a task object"]
        state.stage = "VALIDATING"
        state.last_rejected = copy.deepcopy(response)
        state.validation_errors = errors
        return TransitionResult(False, False, errors)

    covers = [str(item) for item in response.get("covers") or []]
    uncovered_before = set(state.uncovered_acceptance_ids())
    if not uncovered_before.intersection(covers):
        errors = ["covers must include at least one currently uncovered acceptance signal"]
        state.stage = "VALIDATING"
        state.last_rejected = copy.deepcopy(response)
        state.validation_errors = errors
        return TransitionResult(False, False, errors)

    wire_dependencies(proposal, state.accepted)
    acceptance_ids = {
        str(item["id"])
        for item in state.fields.get("acceptanceSignals") or []
        if isinstance(item, dict) and item.get("id")
    }
    prior_task_ids = {
        str(entry.get("task", {}).get("id"))
        for entry in state.accepted
        if isinstance(entry, dict)
    }
    result = validate_task(
        proposal,
        prefix=str(state.fields.get("taskPrefix") or ""),
        task_number=len(state.accepted) + 1,
        implementation_prefixes=[
            str(item) for item in state.fields.get("implementationPrefixes") or []
        ],
        acceptance_ids=acceptance_ids,
        covers=covers,
        prior_task_ids=prior_task_ids,
    )
    if not result.valid:
        state.stage = "VALIDATING"
        state.last_rejected = copy.deepcopy(response)
        state.validation_errors = result.errors
        return TransitionResult(False, False, result.errors)

    state.accepted.append({"task": proposal, "covers": covers})
    state.stage = "DECOMPOSING"
    state.last_rejected = None
    state.validation_errors = []
    return TransitionResult(True, False, [])
