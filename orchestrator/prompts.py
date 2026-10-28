from __future__ import annotations

import json
from typing import Any

from tools import _tasks

from orchestrator import config
from orchestrator.continuity import load_continuity_packet
from orchestrator.log import get_logger
from orchestrator.prompt_variants import clause
from orchestrator.session import SessionStore, get_session
from orchestrator.verification_spec import slice_token

logger = get_logger(__name__)


def _legacy_paths_text() -> str:
    return ", ".join(config.LEGACY_PREFIXES) if config.LEGACY_PREFIXES else "legacy paths"


def _project_guidance_clause(subtask_id: str | None = None) -> str:
    from orchestrator.knowledge import guidance_for

    guidance = guidance_for(subtask_id).strip()
    return f"Project guidance: {guidance} " if guidance else ""


def _full_subtask_definition(task: dict[str, Any], subtask_id: str | None) -> dict[str, Any] | None:
    if not subtask_id:
        return None
    for item in task.get("subtasks", []):
        if isinstance(item, dict) and item.get("id") == subtask_id:
            return item
    return None


def _scope_task_payload(
    task: dict[str, Any],
    subtask_state: dict[str, Any],
    *,
    merged_subtask: dict[str, Any] | None = None,
    tool_case: str | None = None,
) -> dict[str, Any]:
    active_subtask = subtask_state.get("activeSubtask") or {}
    active_id = active_subtask.get("id")
    full_subtask = merged_subtask or _full_subtask_definition(task, active_id)

    scoped: dict[str, Any] = {
        "id": task.get("id"),
        "summary": task.get("summary"),
        "description": task.get("description"),
        "dependsOn": task.get("dependsOn", []),
        "constraints": task.get("constraints", []),
        "acceptanceCriteria": task.get("acceptanceCriteria", ""),
    }

    if full_subtask:
        scoped["activeSubtask"] = full_subtask
        scoped["scope"] = full_subtask.get("scope", task.get("scope", []))
        scoped["verification"] = full_subtask.get("verification", task.get("verification", []))
    else:
        scoped["scope"] = task.get("scope", [])
        scoped["verification"] = task.get("verification", [])

    implementation_logic = dict(task.get("implementationLogic", {}))
    if full_subtask is not None:
        if full_subtask.get("businessRules"):
            implementation_logic["businessRules"] = full_subtask["businessRules"]
        else:
            implementation_logic.pop("businessRules", None)
        if full_subtask.get("readLegacy") is False:
            for key in ("legacySources", "testFixtures", "edgeCases"):
                implementation_logic.pop(key, None)
            algorithm_notes = [
                note
                for note in implementation_logic.get("algorithmNotes", [])
                if "legacy" not in note.lower()
            ]
            if algorithm_notes:
                implementation_logic["algorithmNotes"] = algorithm_notes
            else:
                implementation_logic.pop("algorithmNotes", None)
    if implementation_logic:
        scoped["implementationLogic"] = implementation_logic

    if full_subtask and full_subtask.get("readLegacy") is False:
        scoped["instructions"] = list(full_subtask.get("instructions") or []) or [
            "Implement the active subtask only.",
            "This subtask is skeleton creation only: types/constants/contracts, no extra business logic.",
            f"Do not read, search, or extract_symbol from {_legacy_paths_text()} or legacy modules.",
            "Do not create temp inspection scripts.",
            "Write target files directly from the subtask scope.",
        ]
    elif full_subtask and full_subtask.get("readLegacy") is True:
        if full_subtask.get("legacySymbols"):
            scoped["legacySymbols"] = full_subtask["legacySymbols"]
        if full_subtask.get("allowedSearchTerms"):
            scoped["allowedSearchTerms"] = full_subtask["allowedSearchTerms"]
        if full_subtask.get("forbiddenSearches"):
            scoped["forbiddenSearchHints"] = full_subtask["forbiddenSearches"]
        if full_subtask.get("targetArtifacts"):
            scoped["targetArtifacts"] = full_subtask["targetArtifacts"]
        if full_subtask.get("dataExamples"):
            scoped["dataExamples"] = full_subtask["dataExamples"]
        instructions = list(full_subtask.get("instructions") or [])
        if (
            tool_case in {None, "explore"}
            and full_subtask.get("legacySymbols")
            and not any("get_subtask_context" in line for line in instructions)
            and not any("contextPack" in line for line in instructions)
        ):
            instructions.insert(
                0,
                "Call get_subtask_context(task_id) first when legacySymbols is present.",
            )
        if instructions:
            scoped["instructions"] = instructions

    return scoped


def _prompt_variables(subtask_id: str | None = None, task_id: str | None = None) -> dict[str, str]:
    return {
        "subtask_id": subtask_id or "",
        "task_id": task_id or "",
        "legacy_paths": _legacy_paths_text(),
        "project_name": config.PROJECT_NAME,
        "task_prefix": config.TASK_PREFIX,
        "implementation_prefixes": ", ".join(config.IMPLEMENTATION_PREFIXES),
    }


def _read_handoff_sentence() -> str:
    return (
        "Call read_handoff(include_result=true) only when a compact tool excerpt "
        "ends with [truncated; full output stored in handoff]; use it once per "
        "handoffId — do not use run_command to cat or re-read the same file. "
    )


def _cased_tools_text(tool_case: str, *, include_read_handoff: bool) -> str:
    prefixes = ", ".join(config.IMPLEMENTATION_PREFIXES)
    read_limit = (
        "Use read_file only for small files, XML/JSON/config assets, or "
        f"code under {prefixes}. "
    )
    if tool_case == "explore":
        text = (
            "Use search_files, list_directory, find_symbol, and extract_symbol before writing. "
            f"{read_limit}"
            "Never read handoff JSON with read_file. "
            "For Python definitions in large legacy modules, call extract_symbol "
            "(optionally with path and start_line/end_line) instead of read_file "
            "on the whole module. "
            "When legacySymbols is present on the active subtask and contextPack does not "
            "already include legacySlices or referenceSnippets, call "
            f"get_subtask_context(task_id) once; search_files on {_legacy_paths_text()} "
            "is blocked in the orchestrator worker. "
            "Obey contextPack.explorationPolicy maxReadOnlyBeforeWrite; after the budget, "
            "replace_in_file or write_file is required. "
        )
    elif tool_case == "test_write":
        text = (
            "Use list_directory and read_file to locate the test file. "
            f"{read_limit}"
            "Use write_file for a new test file and replace_in_file for an existing one. "
            "Run the exact declared verification command with run_command. "
        )
    else:
        text = (
            f"{read_limit}"
            "For an existing file use replace_in_file with a small unique old_text/new_text pair. "
            "For a new small file use write_file. "
            "Run the exact verification command with run_command. "
            "Do not search, list directories, or extract symbols; the context pack already "
            "has the needed source. "
        )
    if include_read_handoff:
        text += _read_handoff_sentence()
    return text


def system_prompt(
    subtask_id: str | None = None,
    *,
    role: str = "implementer",
    tool_case: str | None = None,
    include_read_handoff: bool = False,
) -> str:
    """Worker system prompt assembled from named clauses.

    Each clause can be overridden by the active prompt variant (see
    ``orchestrator/prompt_variants.py``). Clause names are stable identifiers
    so that eval reports can say which clause a variant changed.
    """
    variables = _prompt_variables(subtask_id)
    subtask_clause = clause(
        "system.subtask" if subtask_id else "system.subtask_unpinned",
        (
            f"Implement exactly ONE subtask ({subtask_id}). "
            if subtask_id
            else "Implement exactly one active subtask. "
        ),
        **variables,
    )
    legacy_clause = clause(
        "system.legacy",
        (
            f"When the active subtask says readLegacy is false or forbids legacy reads, "
            f"do not inspect {_legacy_paths_text()}. "
            if config.LEGACY_PREFIXES
            else "When the active subtask says readLegacy is false, do not inspect unrelated legacy modules. "
        ),
        **variables,
    )
    if role == "test_writer":
        role_clause = clause(
            "system.role.test_writer",
            "You are the test-writer for this subtask. Write or update failing tests "
            "and fixtures only. Do not implement production code. Do not mark the "
            "subtask DONE. Include the slice token from the user prompt in the test "
            "name or docstring so the declared -k selector collects it. Run the exact "
            "declared verification command. When that command fails with a real "
            "assertion or import failure (exit 1), stop; a fresh implementer "
            "session will make the tests pass. Collection errors, SyntaxError, "
            "and pytest exit 2 are not a failing test: fix the test file and rerun. "
            "If the declared command exits 0, stop after the command; the coordinator "
            "will run the independent verifier and decide whether the subtask was "
            "already satisfied. ",
            **variables,
        )
        tdd_clause = ""
        verify_clause = ""
    else:
        role_clause = clause(
            "system.role.implementer",
            "You are the implementer for this subtask. Tests already exist. Do not "
            "rewrite tests. Implement production code so the declared verification "
            "command passes. ",
            **variables,
        )
        tdd_clause = clause(
            "system.tdd",
            "Follow Clean Code, SOLID, KISS, and DRY. Use fakes, fixtures, injected "
            "clocks, or mocks for network calls, hardware, filesystem edge cases, and time. ",
            **variables,
        )
        verify_clause = clause(
            "system.verify",
            "Run the exact verification command declared by the active subtask. When "
            "that command exits successfully, stop; the coordinator runs an independent "
            "verifier and marks the subtask DONE. Do not start the next subtask; the "
            "orchestrator starts a fresh session for each subtask. "
            "Do not do extra searches or inspections after verification passes. ",
            **variables,
        )
    intro_clause = clause(
        "system.intro",
        f"You are implementing exactly one {config.PROJECT_NAME} subtask in a local repository. "
        "Use native Ollama tool calls only. Do not describe tool calls in text. "
        "Stay within the active subtask scope and constraints. ",
        **variables,
    )
    if tool_case is None:
        state_text = (
            "Call get_active_subtask early and call update_subtask_state when the "
            "subtask is started, failed, or blocked. The coordinator owns the final "
            "DONE transition after independent verification. "
        )
        tools_text = (
            "Use search/list/read before writing. "
            "Use read_file only for small files, XML/JSON/config assets, or "
            f"code under {', '.join(config.IMPLEMENTATION_PREFIXES)}. Never read handoff JSON with read_file; "
            "use read_handoff with handoff_id when you need a stored tool result. "
            "For Python definitions in large legacy modules, call extract_symbol "
            "(optionally with path and start_line/end_line) instead of read_file "
            "on the whole module. "
            "When legacySymbols is present on the active subtask and contextPack does not "
            "already include legacySlices or referenceSnippets, call "
            f"get_subtask_context(task_id) once; search_files on {_legacy_paths_text()} "
            "is blocked in the orchestrator worker. "
            "When contextPack.subtaskContext.legacySlices or contextPack.referenceSnippets "
            "are present, do not call get_subtask_context or extract_symbol for those "
            "symbols; proceed directly to replace_in_file for an existing file or "
            "write_file for a new small file. "
            "Call read_handoff(include_result=true) only when a compact tool excerpt "
            "ends with [truncated; full output stored in handoff]; use it once per "
            "handoffId — do not use run_command to cat or re-read the same file. "
            "Obey contextPack.explorationPolicy maxReadOnlyBeforeWrite; after the budget, "
            "replace_in_file or write_file is required. "
        )
        failure_text = (
            "If blocked or failing, call update_subtask_state or update_task_state "
            "with status FAILED or BLOCKED and include progress_summary, "
            "completed_scope_items, remaining_scope_items, verified_behaviors, "
            "partial_files_touched, last_failure_reason, and recommended_resume_step. "
            f"Do not work on dependent {config.TASK_PREFIX} tasks early."
        )
    else:
        state_text = (
            "Call update_subtask_state when the subtask is started, failed, or "
            "blocked. The coordinator owns the final DONE transition after "
            "independent verification. "
        )
        tools_text = _cased_tools_text(tool_case, include_read_handoff=include_read_handoff)
        failure_text = (
            "If blocked or failing, call update_subtask_state "
            "with status FAILED or BLOCKED and include progress_summary, "
            "completed_scope_items, remaining_scope_items, verified_behaviors, "
            "partial_files_touched, last_failure_reason, and recommended_resume_step. "
            f"Do not work on dependent {config.TASK_PREFIX} tasks early."
        )
    state_clause = clause("system.state", state_text, **variables)
    tools_clause = clause("system.tools", tools_text, **variables)
    next_action_clause = clause(
        "system.next_action",
        "When a compact tool result includes nextAction, call that tool next unless "
        "the result status is error. "
        "Exception: if the error is a Python syntax diagnostic (Python syntax invalid), "
        "you MUST call nextAction.tool again on the same path with a syntax-valid "
        "version; use the SyntaxError/IndentationError diagnostic in the tool excerpt; "
        "do not switch tools or reply with a plan. ",
        **variables,
    )
    verification_style_clause = clause(
        "system.verification_style",
        "Prefer bounded verification commands such as unit tests, imports, or "
        "FastAPI TestClient requests. "
        "Do not spend tool steps investigating occupied ports after behavior "
        "has already been verified; treat an occupied port as an environment "
        "constraint and use an in-process request check instead. ",
        **variables,
    )
    failure_clause = clause("system.failure", failure_text, **variables)
    return (
        f"{intro_clause}"
        f"{subtask_clause}"
        f"{role_clause}"
        f"{_project_guidance_clause(subtask_id)}"
        f"{state_clause}"
        f"{legacy_clause}"
        f"{tdd_clause}"
        f"{tools_clause}"
        f"{next_action_clause}"
        f"{verification_style_clause}"
        f"{verify_clause}"
        f"{failure_clause}"
    )


def task_user_prompt_for_subtask(
    task_id: str,
    subtask_id: str,
    *,
    context_pack: dict[str, Any] | None = None,
    merged_subtask: dict[str, Any] | None = None,
    recovery_context: dict[str, Any] | None = None,
    session: SessionStore | None = None,
    attempt_number: int = 1,
    role: str = "implementer",
    tool_case: str | None = None,
) -> str:
    packet = _tasks.get_task_packet(task_id)
    checklist = _tasks.build_verification_checklist(task_id)
    subtask_state = _tasks.normalize_subtask_state(
        task_id,
        packet["task"],
        packet["state"],
    )
    scoped_task = _scope_task_payload(
        packet["task"],
        subtask_state,
        merged_subtask=merged_subtask,
        tool_case=tool_case,
    )
    payload: dict[str, Any] = {
        "task": scoped_task,
        "state": packet["state"],
        "subtaskState": subtask_state,
        "verificationChecklist": checklist,
        "pinnedSubtaskId": subtask_id,
        "workerRole": role,
        "sliceToken": slice_token(subtask_id),
    }
    if context_pack is not None:
        payload["contextPack"] = context_pack

    continuity = load_continuity_packet(task_id, subtask_id)
    if continuity is not None:
        payload["continuityPacket"] = continuity
        logger.info(
            "%s.%s: loaded continuityPacket (attempt %s, nextStep=%r)",
            task_id,
            subtask_id,
            continuity.get("attempt"),
            (continuity.get("nextStep") or "")[:80],
        )

    active_session = session or get_session()
    if recovery_context:
        compact_recovery = dict(recovery_context)
        compact_recovery.pop("continuityPacket", None)
        payload["recoveryContext"] = compact_recovery
        logger.info(
            "%s.%s: injecting recoveryContext (retry attempt %s)",
            task_id,
            subtask_id,
            attempt_number,
        )
    elif active_session is not None and active_session.should_inject_on_first_attempt(task_id):
        logger.info(
            "%s.%s: building sessionContext (prior handoffs or in-progress state)",
            task_id,
            subtask_id,
        )
        state = packet["state"] or {}
        active_session.seed_subtask_lessons(task_id)
        if state:
            logger.info(
                "%s.%s: extracting lessons from handoff chain",
                task_id,
                subtask_id,
            )
            active_session.extract_lessons_from_handoffs(task_id, state)
        include_cached_context = True
        if context_pack is not None:
            exploration_policy = context_pack.get("explorationPolicy") or {}
            if exploration_policy.get("preflightSatisfied"):
                include_cached_context = False
        session_payload = active_session.get_session_payload_for_retry(
            task_id,
            include_cached_context=include_cached_context,
        )
        payload["sessionContext"] = session_payload
        logger.info(
            "%s.%s: sessionContext ready (%d lessons, cachedSubtaskContext=%s)",
            task_id,
            subtask_id,
            len(session_payload.get("sessionLessons") or []),
            session_payload.get("cachedSubtaskContext") is not None,
        )

    context_note = ""
    if context_pack is not None:
        exploration_policy = context_pack.get("explorationPolicy") or {}
        scout_brief = context_pack.get("scoutBrief") or {}
        if scout_brief.get("findings"):
            context_note = (
                "Parallel scouts already inspected the paths in "
                "contextPack.scoutBrief. Treat those findings as authoritative "
                "and do not repeat the same reads.\n"
            )
        if context_pack.get("domainKnowledge"):
            context_note += (
                "contextPack.domainKnowledge is authoritative for this task's "
                "language or parser rules. Follow it before inventing syntax.\n"
            )
        if exploration_policy.get("preflightSatisfied"):
            context_note += (
                "contextPack already includes legacySlices and/or referenceSnippets. "
                "Do not call get_subtask_context, extract_symbol, or read_handoff "
                "for preloaded symbols unless the compact excerpt is truncated.\n"
            )
        max_read_only = exploration_policy.get("maxReadOnlyBeforeWrite")
        if max_read_only is not None:
            context_note += (
                f"Exploration budget: at most {max_read_only} read-only tools "
                "before the first replace_in_file or write_file.\n"
            )

    resume_note = ""
    if payload.get("continuityPacket"):
        resume_note = (
            "continuityPacket is authoritative for resume. Read it first. "
            "Obey doNotRepeat. Execute nextStep. Use evidence handoff IDs "
            "only when the excerpt is insufficient.\n"
        )
    elif recovery_context or payload.get("sessionContext"):
        resume_note = (
            "If recoveryContext or sessionContext is present, preserve verified "
            "completed work, read sessionLessons first, and continue from "
            "recommendedResumeStep or next_action lessons after inspecting "
            "current files.\n"
        )

    return (
        user_prompt_opening(task_id, subtask_id, role=role)
        + clause(
            "user.tooling",
            "Use the tool set to inspect, edit, and verify the repository.\n"
            "Context pack and merged subtask spec are authoritative when present.\n",
            **_prompt_variables(subtask_id, task_id),
        )
        + f"{context_note}"
        f"{resume_note}"
        "Task payload:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}"
    )


def user_prompt_opening(task_id: str, subtask_id: str, *, role: str = "implementer") -> str:
    """First lines of the worker user prompt (role-specific, variant-overridable)."""
    variables = _prompt_variables(subtask_id, task_id)
    if role == "test_writer":
        return clause(
            "user.opening.test_writer",
            f"Write failing tests for {task_id} subtask {subtask_id} only.\n"
            "Do not implement production code. Include sliceToken in the test "
            "name or docstring. Run the exact declared verification and stop "
            "when it fails with a real test failure, not a SyntaxError or "
            "collection error.\n",
            **variables,
        )
    return clause(
        "user.opening.implementer",
        f"Implement {task_id} subtask {subtask_id} only.\n"
        "Tests already exist. Do not rewrite them. Implement production "
        "code so the declared verification passes.\n"
        "If prior state is FAILED but the files already satisfy the subtask, "
        "run the exact declared verification instead of reimplementing from scratch; "
        "the coordinator owns the DONE transition.\n",
        **variables,
    )


def task_user_prompt(
    task_id: str,
    recovery_context: dict[str, Any] | None = None,
    session: SessionStore | None = None,
) -> str:
    subtask_state = _tasks.get_active_subtask(task_id)
    subtask_id = subtask_state.get("activeSubtaskId")
    if not subtask_id:
        subtask_id = task_id
    return task_user_prompt_for_subtask(
        task_id,
        subtask_id,
        recovery_context=recovery_context,
        session=session,
    )
