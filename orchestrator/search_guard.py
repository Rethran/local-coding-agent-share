"""Guardrails for orchestrator search_files calls."""

from __future__ import annotations

import json
import re
from typing import Any

from orchestrator import config
from tools import _tasks
from tools._handoffs import CONSUMED_DIR, HANDOFFS_DIR
from tools._common import safe_path

from orchestrator.config import SEARCH_FORBIDDEN_LEGACY_QUERIES, SEARCH_MAX_MATCHES
from orchestrator.read_guard import normalize_relative_path
from orchestrator.worker_context import get_worker_attempt, get_worker_task_id


def _full_subtask_definition(task: dict[str, Any], subtask_id: str | None) -> dict[str, Any] | None:
    if not subtask_id:
        return None
    for item in task.get("subtasks", []):
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return item
    return None


def load_subtask_definition(task_id: str, subtask_id: str) -> dict[str, Any] | None:
    packet = _tasks.get_task_packet(task_id)
    task = packet["task"]
    full = _full_subtask_definition(task, subtask_id)
    if full is not None:
        return full
    for item in _tasks.derive_subtasks(task_id, task):
        if item.get("id") == subtask_id:
            return item
    return None


def load_active_subtask_definition(task_id: str) -> dict[str, Any] | None:
    packet = _tasks.get_task_packet(task_id)
    normalized = _tasks.normalize_subtask_state(task_id, packet["task"], packet["state"])
    active_id = normalized.get("activeSubtaskId")
    return load_subtask_definition(task_id, active_id) if active_id else None


def is_handoff_search_path(path: str) -> bool:
    normalized = normalize_relative_path(path)
    handoff_prefixes = []
    for directory in (HANDOFFS_DIR, CONSUMED_DIR):
        try:
            handoff_prefixes.append(normalize_relative_path(str(directory.relative_to(safe_path(".")))))
        except ValueError:
            continue
    return any(
        normalized == prefix or normalized.startswith(prefix + "/")
        for prefix in handoff_prefixes
    )


def classify_search_path(path: str) -> str:
    normalized = normalize_relative_path(path)
    if is_handoff_search_path(normalized):
        return "handoff"
    if any(normalized.startswith(prefix) for prefix in config.IMPLEMENTATION_PREFIXES):
        return "implementation"
    if normalized.endswith(".xml") or normalized.endswith(".json"):
        return "fixture"
    if any(normalized.startswith(prefix) for prefix in config.LEGACY_PREFIXES) and normalized.endswith(".py"):
        return "legacy_py"
    if any(
        normalized.startswith(prefix) or normalized == prefix.rstrip("/")
        for prefix in config.LEGACY_PREFIXES
    ):
        return "legacy"
    if normalized == ".":
        return "workspace"
    return "other"


def build_allowed_search_terms(subtask: dict[str, Any], task: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            terms.append(cleaned)

    for item in subtask.get("allowedSearchTerms") or []:
        add(str(item))

    for symbol in subtask.get("legacySymbols") or []:
        if isinstance(symbol, dict):
            add(str(symbol.get("name", "")))

    target = subtask.get("targetArtifacts") or {}
    if isinstance(target, dict):
        add(str(target.get("function", "")))
        add(str(target.get("module", "")))
        add(str(target.get("test", "")))

    impl = task.get("implementationLogic") or {}
    for fixture in impl.get("testFixtures") or []:
        add(str(fixture))

    return terms


def query_matches_allowed_term(query: str, allowed_terms: list[str]) -> bool:
    query_lower = query.lower().strip()
    if not query_lower:
        return False
    for term in allowed_terms:
        term_lower = term.lower()
        if term_lower in query_lower or query_lower in term_lower:
            return True
    return False


def matches_forbidden_pattern(
    query: str,
    path: str,
    rule: dict[str, Any],
) -> bool:
    query_pattern = str(rule.get("queryPattern", ""))
    if not query_pattern:
        return False
    if not re.search(query_pattern, query, flags=re.IGNORECASE):
        return False
    path_pattern = rule.get("pathPattern")
    if path_pattern and not re.search(str(path_pattern), normalize_relative_path(path)):
        return False
    return True


def first_legacy_symbol(subtask: dict[str, Any]) -> dict[str, Any] | None:
    symbols = subtask.get("legacySymbols") or []
    for symbol in symbols:
        if isinstance(symbol, dict) and symbol.get("name"):
            return symbol
    return None


def format_refusal(
    query: str,
    path: str,
    subtask_id: str,
    reason: str,
    allowed_terms: list[str],
    legacy_symbol: dict[str, Any] | None,
) -> str:
    allowed_preview = ", ".join(allowed_terms[:12])
    if len(allowed_terms) > 12:
        allowed_preview += ", ..."
    message = (
        f"Refusing search_files query {query!r} on {path} for subtask {subtask_id}. "
        f"{reason} "
        f"Allowed terms on legacy paths: {allowed_preview or '(none)'}."
    )
    if legacy_symbol:
        name = legacy_symbol.get("name")
        default_legacy_path = (
            f"{config.LEGACY_PREFIXES[0]}entsoe_prod.py"
            if config.LEGACY_PREFIXES
            else "legacy module"
        )
        symbol_path = legacy_symbol.get("path", default_legacy_path)
        message += (
            f" Use get_subtask_context(task_id) or "
            f"extract_symbol(name={name!r}, path={symbol_path!r}) instead."
        )
    else:
        message += " Use get_subtask_context(task_id) for curated legacy context."
    return message


def validate_search_files(query: str, path: str = ".") -> str | None:
    """Return an error message when search_files must be refused, else None."""
    normalized_path = normalize_relative_path(path)
    tier = classify_search_path(normalized_path)

    if tier == "handoff":
        return (
            f"Refusing search_files under handoff storage ({path}). "
            "Use read_handoff with handoff_id instead."
        )

    attempt = get_worker_attempt()
    task_id = get_worker_task_id()
    from orchestrator.session import get_session

    session = get_session()
    key = (query.strip().lower(), normalized_path)

    if session is not None and task_id:
        prior = session.get_tool_record(task_id, "search_files", {"query": query, "path": path})
        if prior is not None and prior.get("status") == "ok":
            handoff_id = prior.get("handoffId", "")
            return (
                f"Duplicate search_files({query!r}, {path!r}) in this session "
                f"(handoff {handoff_id}). Proceed to replace_in_file or run verification."
            )
    elif attempt is not None:
        if key in attempt.seen_searches:
            return (
                f"Duplicate search_files({query!r}, {path!r}) in this attempt. "
                "Proceed to replace_in_file or run verification."
            )
        attempt.seen_searches.add(key)
    subtask = load_active_subtask_definition(task_id) if task_id else None
    task = _tasks.get_task_packet(task_id)["task"] if task_id else {}
    subtask_id = str(subtask.get("id")) if subtask else "unknown"
    allowed_terms = build_allowed_search_terms(subtask or {}, task) if subtask else []
    legacy_symbol = first_legacy_symbol(subtask) if subtask else None

    if subtask:
        for rule in subtask.get("forbiddenSearches") or []:
            if isinstance(rule, dict) and matches_forbidden_pattern(query, path, rule):
                return format_refusal(
                    query,
                    path,
                    subtask_id,
                    str(rule.get("reason", "Query matches forbiddenSearches rule.")),
                    allowed_terms,
                    legacy_symbol,
                )

    for pattern in SEARCH_FORBIDDEN_LEGACY_QUERIES:
        if tier in ("legacy_py", "legacy") and re.search(pattern, query, flags=re.IGNORECASE):
            return format_refusal(
                query,
                path,
                subtask_id,
                "Overly broad legacy Python search.",
                allowed_terms,
                legacy_symbol,
            )

    if subtask and subtask.get("legacySymbols") and tier == "legacy_py":
        return format_refusal(
            query,
            path,
            subtask_id,
            "search_files on legacy Python is blocked when legacySymbols is defined.",
            allowed_terms,
            legacy_symbol,
        )

    if tier in ("legacy_py", "legacy") and subtask and allowed_terms:
        if not query_matches_allowed_term(query, allowed_terms):
            return format_refusal(
                query,
                path,
                subtask_id,
                "Query is not in allowedSearchTerms for this subtask.",
                allowed_terms,
                legacy_symbol,
            )

    return None


def cap_search_results(result: str, path: str) -> str:
    tier = classify_search_path(normalize_relative_path(path))
    if tier not in ("legacy_py", "legacy", "workspace"):
        return result

    try:
        matches = json.loads(result)
    except json.JSONDecodeError:
        return result

    if not isinstance(matches, list) or len(matches) <= SEARCH_MAX_MATCHES:
        return result

    return (
        f"ERROR: Search returned {len(matches)} matches on {path} "
        f"(limit {SEARCH_MAX_MATCHES}). Narrow the query or use extract_symbol / "
        "get_subtask_context instead."
    )
