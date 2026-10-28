"""Merge hand-authored HEAT subtask specs with planner sidecar enrichments."""

from __future__ import annotations

from typing import Any

LIST_MERGE_KEYS = (
    "legacySymbols",
    "relatedArtifacts",
    "allowedSearchTerms",
    "instructions",
    "verification",
    "scope",
    "dataExamples",
    "businessRules",
)


def merge_subtask_definition(
    spec_subtask: dict[str, Any],
    enrichments: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(spec_subtask)
    if not enrichments:
        return merged

    for key in LIST_MERGE_KEYS:
        value = enrichments.get(key)
        if isinstance(value, list) and value:
            merged[key] = value

    target = enrichments.get("targetArtifacts")
    if isinstance(target, dict) and target:
        existing = merged.get("targetArtifacts")
        if isinstance(existing, dict):
            merged["targetArtifacts"] = {**existing, **target}
        else:
            merged["targetArtifacts"] = dict(target)

    if enrichments.get("readLegacy") is not None and merged.get("readLegacy") is None:
        merged["readLegacy"] = enrichments["readLegacy"]

    exploration_policy = enrichments.get("explorationPolicy")
    if isinstance(exploration_policy, dict) and exploration_policy:
        existing_policy = merged.get("explorationPolicy")
        if isinstance(existing_policy, dict):
            merged["explorationPolicy"] = {**existing_policy, **exploration_policy}
        else:
            merged["explorationPolicy"] = dict(exploration_policy)

    for key in ("title", "summary"):
        if enrichments.get(key) and not merged.get(key):
            merged[key] = enrichments[key]

    return merged
