"""Turn an unexecutable dataExample into a pass condition the developer can run.

The coordinator oracle reports that an example is not executable when its keys
are not function arguments. The investigator names the condition those keys
describe. The developer runs that condition against the target function. A
passing condition lets the already-green subtask continue instead of ending
the run.
"""

from __future__ import annotations

from dataclasses import dataclass

from orchestrator.contract_gate import bind_call_kwargs, load_target_function

# CTX-005 stored-event payload. Optional CTX-022 fields are absent on purpose.
_BASELINE_EVENT = {
    "org_id": "org-a",
    "team_id": "team-1",
    "user_id": "alice",
    "ts": "2026-10-05T12:00:00Z",
    "provider": "ollama",
    "model_id": "qwen3:8b",
    "prompt_tokens": 10,
    "completion_tokens": 4,
    "context_window": 32768,
    "context_pct": 0.01,
    "action": "none",
    "tokens_saved_est": 0,
    "token_source": "provider",
}


@dataclass(frozen=True)
class PassCondition:
    summary: str
    omitted: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateRecovery:
    passed: bool
    summary: str
    blockers: tuple[str, ...] = ()


def _unexecutable_only(blockers: list[str]) -> bool:
    if not blockers:
        return False
    return all("not executable by the coordinator oracle" in item for item in blockers)


def _omitted_keys(data_examples: object) -> tuple[str, ...] | None:
    if not isinstance(data_examples, list) or len(data_examples) != 1:
        return None
    example = data_examples[0]
    if not isinstance(example, dict) or set(example) != {"omitted"}:
        return None
    omitted = example.get("omitted")
    if not isinstance(omitted, list) or not omitted:
        return None
    if not all(isinstance(item, str) and item for item in omitted):
        return None
    return tuple(omitted)


def investigate_pass_condition(merged_subtask: dict) -> PassCondition | None:
    """Name a pass condition for an example the oracle cannot call directly."""
    omitted = _omitted_keys(merged_subtask.get("dataExamples"))
    if omitted is None:
        return None
    names = ", ".join(omitted)
    return PassCondition(
        summary=(
            f"Call the target with a payload that omits {names}. "
            "Pass when those keys are absent from the stored result."
        ),
        omitted=omitted,
    )


def develop_pass_condition(
    label: str,
    merged_subtask: dict,
    condition: PassCondition,
) -> list[str]:
    """Run the investigator's condition. Empty means the subtask may continue."""
    target = merged_subtask.get("targetArtifacts") or {}
    module_rel = str(target.get("module") or "").strip()
    function = str(target.get("function") or "").strip()
    if not module_rel or not function:
        return [f"{label} developer has no target function for the pass condition."]
    try:
        _module, fn = load_target_function(module_rel, function)
    except Exception as exc:
        return [f"{label} developer cannot load {function} from {module_rel}: {exc}"]
    payload = {
        key: value
        for key, value in _BASELINE_EVENT.items()
        if key not in condition.omitted
    }
    try:
        kwargs = bind_call_kwargs(fn, {"payload": payload, "salt": "test-salt"}, _module)
        result = fn(**kwargs)
    except Exception as exc:
        return [f"{label} developer pass condition raised {type(exc).__name__}: {exc}"]
    if not isinstance(result, dict):
        return [f"{label} developer expected a stored dict, got {type(result).__name__}."]
    blockers: list[str] = []
    for key in condition.omitted:
        if key in result:
            blockers.append(
                f"{label} stored result still contains omitted key {key!r}."
            )
    return blockers


def recover_unexecutable_duplicate(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict,
    blockers: list[str],
) -> DuplicateRecovery | None:
    """Investigate, then let the developer apply the condition.

    Returns None when the failure is not an unexecutable-example block, so
    stub failures and real oracle mismatches still stop the run.
    """
    if not _unexecutable_only(blockers):
        return None
    label = f"{task_id}.{subtask_id}"
    condition = investigate_pass_condition(merged_subtask)
    if condition is None:
        return DuplicateRecovery(
            passed=False,
            summary="Investigator found no pass condition for these dataExamples.",
            blockers=tuple(blockers),
        )
    print(f"{label}: investigator: {condition.summary}")
    developed = develop_pass_condition(label, merged_subtask, condition)
    if developed:
        print(f"{label}: developer could not satisfy the pass condition")
        return DuplicateRecovery(
            passed=False,
            summary=condition.summary,
            blockers=tuple(developed),
        )
    print(f"{label}: developer applied the pass condition")
    return DuplicateRecovery(passed=True, summary=condition.summary)
