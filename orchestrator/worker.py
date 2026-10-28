from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from tools import _tasks
from tools._repair import repair_output_text

from orchestrator import run_metrics, tool_compact
from orchestrator.attempts import infer_likely_cause_and_fix
from orchestrator.continuity import persist_and_sync
from orchestrator.config import (
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_MAX_CONTEXT_HANDOVERS,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    MALFORMED_TOOL_CALL_RETRY_MAX,
    READ_ONLY_TOOLS,
    SYNTAX_WRITE_RETRY_MAX,
    TEXT_ONLY_RETRY_MAX,
)
from orchestrator.context import (
    context_budget_chars,
    context_usage_ratio,
    estimate_messages_chars,
    extract_eval_count,
    extract_prompt_eval_count,
    perform_context_handover,
    should_handover_context,
)
from orchestrator.log import get_logger
from orchestrator.ollama_client import ollama_chat
from orchestrator.prompts import system_prompt, task_user_prompt_for_subtask
from orchestrator.python_write_gate import (
    SYNTAX_WRITE_RETRY_HINT,
    is_python_syntax_write_error,
)
from orchestrator.session import SessionStore, get_session
from orchestrator.spec_readiness import is_subtask_done
from orchestrator.thinking_policy import (
    INITIAL_WORKER_EFFORT,
    RECOVERY_EFFORT,
    effort_after_tool_results,
)
from orchestrator.tool_cases import select_worker_tools, session_flags_from
from orchestrator.tools_registry import TOOL_FUNCTIONS
from orchestrator.verifier import (
    is_declared_verification_failure,
    is_declared_verification_success,
)
from orchestrator.worker_context import begin_worker_attempt, clear_worker_context

logger = get_logger(__name__)


def response_to_message(response: Any) -> dict[str, Any]:
    message = response.message
    if isinstance(message, dict):
        normalized = dict(message)
    elif hasattr(message, "model_dump"):
        normalized = message.model_dump(exclude_none=True)
    else:
        normalized = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", ""),
            "tool_calls": getattr(message, "tool_calls", None),
        }
    normalized.pop("thinking", None)
    return normalized


def print_visible_model_output(task_id: str, step: int, content: str) -> None:
    cleaned = content.strip()
    if not cleaned:
        return
    print(f"\n{task_id}: visible model output at step {step}:")
    print(cleaned)
    print()


@dataclass(frozen=True)
class WorkerResult:
    ok: bool
    state: dict[str, Any]
    previous_handoff_id: str = ""
    ready_for_verification: bool = False


def run_worker_for_task(
    task_id: str,
    model: str,
    max_tool_steps: int,
    *,
    show_model_output: bool,
    attempt_number: int = 1,
    recovery_context: dict[str, Any] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    context_threshold: float = DEFAULT_CONTEXT_THRESHOLD,
    max_context_handovers: int = DEFAULT_MAX_CONTEXT_HANDOVERS,
    session: SessionStore | None = None,
) -> WorkerResult:
    normalized = _tasks.ensure_subtask_state(task_id)
    subtask_id = normalized.get("activeSubtaskId")
    if not subtask_id:
        raise ValueError(f"No active subtask for {task_id}")
    return run_worker_for_subtask(
        task_id,
        str(subtask_id),
        model,
        max_tool_steps,
        show_model_output=show_model_output,
        attempt_number=attempt_number,
        recovery_context=recovery_context,
        num_ctx=num_ctx,
        num_predict=num_predict,
        context_threshold=context_threshold,
        max_context_handovers=max_context_handovers,
        session=session,
    )


def run_worker_for_subtask(
    task_id: str,
    subtask_id: str,
    model: str,
    max_tool_steps: int,
    *,
    show_model_output: bool,
    attempt_number: int = 1,
    recovery_context: dict[str, Any] | None = None,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    context_threshold: float = DEFAULT_CONTEXT_THRESHOLD,
    max_context_handovers: int = DEFAULT_MAX_CONTEXT_HANDOVERS,
    session: SessionStore | None = None,
    context_pack: dict[str, Any] | None = None,
    merged_subtask: dict[str, Any] | None = None,
    role: str = "implementer",
) -> WorkerResult:
    print(
        f"\n=== {task_id}.{subtask_id}: starting {role} attempt {attempt_number} ==="
    )
    _tasks.update_task_state(
        task_id,
        "IN_PROGRESS",
        summary=f"Orchestrator started subtask {subtask_id} attempt {attempt_number}.",
        active_subtask_id=subtask_id,
    )
    _tasks.ensure_subtask_state(task_id)

    state = _tasks.load_task_state(task_id) or {}
    for item in state.get("subtasks") or []:
        if isinstance(item, dict) and item.get("id") == subtask_id:
            if item.get("status") == "PENDING":
                _tasks.update_subtask_state(
                    task_id,
                    subtask_id,
                    "IN_PROGRESS",
                    summary=f"Worker attempt {attempt_number} started.",
                )
            break

    active_session = session or get_session()
    if active_session is not None:
        logger.info(
            "%s.%s: seeding subtask lessons",
            task_id,
            subtask_id,
        )
        active_session.seed_subtask_lessons(task_id)

    attempt_ctx = begin_worker_attempt(
        task_id,
        role=role,
        merged_subtask=merged_subtask,
    )
    run_metrics.begin_attempt(task_id, subtask_id, role, attempt_number)
    result: WorkerResult | None = None
    try:
        result = _run_worker_loop(
            task_id=task_id,
            subtask_id=subtask_id,
            model=model,
            max_tool_steps=max_tool_steps,
            show_model_output=show_model_output,
            attempt_number=attempt_number,
            recovery_context=recovery_context,
            num_ctx=num_ctx,
            num_predict=num_predict,
            context_threshold=context_threshold,
            max_context_handovers=max_context_handovers,
            attempt_ctx=attempt_ctx,
            session=session or get_session(),
            context_pack=context_pack,
            merged_subtask=merged_subtask,
            role=role,
        )
        return result
    finally:
        if result is None:
            run_metrics.end_attempt(ok=False, stop_reason="exception")
        else:
            run_metrics.end_attempt(
                ok=result.ok,
                stop_reason="ok" if result.ok else str(
                    result.state.get("lastFailureReason") or "failed"
                ),
            )
        clear_worker_context()


def _next_action_hint(session: SessionStore | None, task_id: str) -> str:
    if session is not None:
        task_state = session.tasks.get(task_id)
        action = task_state.pending_next_action if task_state is not None else None
        if action:
            tool = str(action.get("tool") or "replace_in_file")
            hint = str(action.get("hint") or "").strip()
            return f"Call {tool}" + (f" to {hint}" if hint else "")
    return "Call replace_in_file for one small edit, or run_command for verification"


def _text_only_retry_message(
    task_id: str,
    subtask_id: str,
    step: int,
    *,
    next_action: str,
) -> str:
    return (
        f"You must call a tool at step {step} for {task_id}.{subtask_id}. "
        f"Do not reply with text or a plan. {next_action}. "
        "Use replace_in_file instead of rewriting an existing file. "
        "When verified, stop. The coordinator owns DONE."
    )


def _syntax_write_retry_message(
    *,
    task_id: str,
    subtask_id: str,
    tool_name: str,
    path: str,
    diagnostic: str,
    attempt: int,
    limit: int,
) -> str:
    target = f" on {path}" if path else ""
    return (
        f"Python syntax write refused for {task_id}.{subtask_id} "
        f"(retry {attempt}/{limit}). "
        f"You must call {tool_name}{target} again with a syntax-valid version. "
        f"{SYNTAX_WRITE_RETRY_HINT}. Do not reply with a plan or switch tools. "
        "The SyntaxError diagnostic is:\n"
        f"{diagnostic}"
    )


def _syntax_write_failures(
    tool_results: list[Any],
) -> list[Any]:
    return [
        result
        for result in tool_results
        if result.status == "error"
        and result.name in {"write_file", "replace_in_file"}
        and is_python_syntax_write_error(result.raw_result)
    ]


def _include_read_handoff(tools: list[dict[str, Any]]) -> bool:
    return any(tool["function"]["name"] == "read_handoff" for tool in tools)


def _is_malformed_tool_call_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "invalid tool call arguments" in message
        or "unexpected end of json" in message
    )


def _run_worker_loop(
    *,
    task_id: str,
    subtask_id: str,
    model: str,
    max_tool_steps: int,
    show_model_output: bool,
    attempt_number: int,
    recovery_context: dict[str, Any] | None,
    num_ctx: int,
    num_predict: int,
    context_threshold: float,
    max_context_handovers: int,
    attempt_ctx: Any,
    session: SessionStore | None = None,
    context_pack: dict[str, Any] | None = None,
    merged_subtask: dict[str, Any] | None = None,
    role: str = "implementer",
) -> WorkerResult:
    active_session = session or get_session()
    scope = f"{task_id}.{subtask_id}"

    logger.info("%s: building worker bootstrap prompt (attempt %s)", scope, attempt_number)
    case_name, selected_tools = select_worker_tools(
        role,
        context_pack,
        session_flags_from(active_session, task_id),
    )
    system_content = system_prompt(
        subtask_id,
        role=role,
        tool_case=case_name,
        include_read_handoff=_include_read_handoff(selected_tools),
    )
    logger.info(
        "%s: system prompt ready (%d chars)",
        scope,
        len(system_content),
    )

    prompt_flags: list[str] = []
    if context_pack is not None:
        prompt_flags.append("contextPack")
    if recovery_context is not None:
        prompt_flags.append("recoveryContext")
    if merged_subtask is not None:
        prompt_flags.append("mergedSubtask")
    logger.info(
        "%s: building user prompt (%s)",
        scope,
        ", ".join(prompt_flags) if prompt_flags else "spec+state only",
    )
    user_content = task_user_prompt_for_subtask(
        task_id,
        subtask_id,
        context_pack=context_pack,
        merged_subtask=merged_subtask,
        recovery_context=recovery_context,
        session=active_session,
        attempt_number=attempt_number,
        role=role,
        tool_case=case_name,
    )
    logger.info(
        "%s: user prompt ready (%d chars, ~%d tokens est)",
        scope,
        len(user_content),
        len(user_content) // 4,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    total_chars = sum(len(str(message.get("content") or "")) for message in messages)
    logger.info(
        "%s: chat bootstrap ready (%d messages, %d chars, %d tools, case=%s, num_ctx=%d)",
        scope,
        len(messages),
        total_chars,
        len(selected_tools),
        case_name,
        num_ctx,
    )
    if active_session is not None and context_pack is not None:
        exploration_policy = context_pack.get("explorationPolicy") or {}
        active_session.set_exploration_policy(
            task_id,
            preflight_satisfied=bool(exploration_policy.get("preflightSatisfied")),
            max_read_only_before_write=exploration_policy.get("maxReadOnlyBeforeWrite"),
        )
        from orchestrator.read_guard import normalize_relative_path

        preloaded_paths: set[str] = set()
        for snippet in context_pack.get("referenceSnippets") or []:
            if isinstance(snippet, dict) and snippet.get("path"):
                preloaded_paths.add(normalize_relative_path(str(snippet["path"])))
        subtask_context = context_pack.get("subtaskContext") or {}
        for legacy_slice in subtask_context.get("legacySlices") or []:
            if isinstance(legacy_slice, dict) and legacy_slice.get("path"):
                preloaded_paths.add(normalize_relative_path(str(legacy_slice["path"])))
        scout_brief = context_pack.get("scoutBrief") or {}
        for path in scout_brief.get("coveredPaths") or []:
            preloaded_paths.add(normalize_relative_path(str(path)))
        target_module = ""
        target = (merged_subtask or {}).get("targetArtifacts") or {}
        if isinstance(target, dict) and target.get("module"):
            target_module = str(target["module"])
        active_session.set_preloaded_context(
            task_id,
            preloaded_paths=preloaded_paths,
            target_module_path=target_module or None,
        )

    previous_handoff_id = ""
    handovers_used = 0
    last_prompt_eval_count: int | None = None
    stopped_without_tool_call = False
    malformed_tool_call_retries = 0
    last_step = 0
    budget_chars = context_budget_chars(num_ctx)
    thinking_effort = INITIAL_WORKER_EFFORT

    for step in range(1, max_tool_steps + 1):
        last_step = step
        usage_ratio = context_usage_ratio(
            messages,
            budget_chars,
            prompt_eval_count=last_prompt_eval_count,
            num_ctx=num_ctx,
        )
        if should_handover_context(
            usage_ratio=usage_ratio,
            threshold=context_threshold,
            handovers_used=handovers_used,
            max_handovers=max_context_handovers,
        ):
            handovers_used += 1
            run_metrics.record_handover()
            messages, previous_handoff_id = perform_context_handover(
                task_id=task_id,
                subtask_id=subtask_id,
                attempt_number=attempt_number,
                previous_handoff_id=previous_handoff_id,
                usage_ratio=usage_ratio,
                handover_number=handovers_used,
                max_handovers=max_context_handovers,
                session=active_session,
                merged_subtask=merged_subtask,
                role=role,
            )
            last_prompt_eval_count = None

        case_name, selected_tools = select_worker_tools(
            role,
            context_pack,
            session_flags_from(active_session, task_id),
        )
        include_read_handoff = _include_read_handoff(selected_tools)
        refreshed_system = system_prompt(
            subtask_id,
            role=role,
            tool_case=case_name,
            include_read_handoff=include_read_handoff,
        )
        if (
            messages
            and messages[0].get("role") == "system"
            and messages[0].get("content") != refreshed_system
        ):
            messages[0]["content"] = refreshed_system
            logger.info(
                "%s: tool case is now %s (%d tools)",
                scope,
                case_name,
                len(selected_tools),
            )

        logger.info(
            "%s: waiting for Ollama at step %s (model=%s, context_usage=%.0f%%, "
            "request_chars=%d, tools_chars=%d, case=%s, thinking=%s)",
            scope,
            step,
            model,
            usage_ratio * 100,
            estimate_messages_chars(messages),
            len(json.dumps(selected_tools, ensure_ascii=False)),
            case_name,
            thinking_effort,
        )
        chat_started = time.perf_counter()
        try:
            response = ollama_chat(
                model=model,
                messages=messages,
                tools=selected_tools,
                num_ctx=num_ctx,
                num_predict=num_predict,
                think=thinking_effort,
            )
        except Exception as exc:  # noqa: BLE001 - convert model crashes into retryable failures
            if (
                _is_malformed_tool_call_error(exc)
                and malformed_tool_call_retries < MALFORMED_TOOL_CALL_RETRY_MAX
            ):
                malformed_tool_call_retries += 1
                handovers_used += 1
                run_metrics.record_handover()
                logger.warning(
                    "%s: malformed tool call at step %s; compact recovery %s/%s",
                    scope,
                    step,
                    malformed_tool_call_retries,
                    MALFORMED_TOOL_CALL_RETRY_MAX,
                )
                messages, previous_handoff_id = perform_context_handover(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    attempt_number=attempt_number,
                    previous_handoff_id=previous_handoff_id,
                    usage_ratio=usage_ratio,
                    handover_number=handovers_used,
                    max_handovers=max(max_context_handovers, handovers_used),
                    session=active_session,
                    merged_subtask=merged_subtask,
                    role=role,
                )
                messages.append({
                    "role": "user",
                    "content": (
                        "The previous tool call was truncated. Call replace_in_file "
                        "once with a small unique old_text/new_text pair. Do not "
                        "rewrite a complete existing file and do not explain."
                    ),
                })
                last_prompt_eval_count = None
                thinking_effort = RECOVERY_EFFORT
                continue
            failure_reason = f"Ollama chat failed at step {step}: {exc}"
            _, possible_fix = infer_likely_cause_and_fix(failure_reason)
            print(f"{task_id}.{subtask_id}: {failure_reason}")
            state = _tasks.update_task_state(
                task_id,
                "FAILED",
                summary="Worker crashed while waiting for Ollama response.",
                blockers=[failure_reason],
                last_failure_reason=failure_reason,
                recommended_resume_step=possible_fix,
            )
            return WorkerResult(
                ok=False,
                state=state,
                previous_handoff_id=previous_handoff_id,
            )
        chat_elapsed = time.perf_counter() - chat_started
        last_prompt_eval_count = extract_prompt_eval_count(response)
        eval_count = extract_eval_count(response)
        run_metrics.record_chat(
            step=step,
            chat_seconds=chat_elapsed,
            prompt_eval_count=last_prompt_eval_count,
            eval_count=eval_count,
        )
        logger.info(
            "%s: Ollama responded at step %s in %.1fs (prompt_eval=%s, eval=%s)",
            scope,
            step,
            chat_elapsed,
            last_prompt_eval_count if last_prompt_eval_count is not None else "unknown",
            eval_count if eval_count is not None else "unknown",
        )
        assistant_message = response_to_message(response)
        content = assistant_message.get("content") or ""
        tool_calls = assistant_message.get("tool_calls") or []
        if not tool_calls and content.strip():
            repaired = repair_output_text(content)
            if repaired is not None and repaired.tool_name in TOOL_FUNCTIONS:
                tool_calls = [{
                    "function": {
                        "name": repaired.tool_name,
                        "arguments": repaired.arguments,
                    },
                }]
                assistant_message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                }
                content = ""
                logger.info(
                    "%s: recovered XML tool call %s at step %s",
                    scope,
                    repaired.tool_name,
                    step,
                )

        if not tool_calls:
            if show_model_output:
                print_visible_model_output(task_id, step, content)
            messages.append({
                "role": "assistant",
                "content": (
                    f"[omitted text-only response: {len(content)} chars; "
                    "a native tool call is required]"
                ),
            })
            if attempt_ctx.text_only_retries < TEXT_ONLY_RETRY_MAX:
                attempt_ctx.text_only_retries += 1
                logger.warning(
                    "%s: omitted %d-character text-only response at step %s",
                    scope,
                    len(content),
                    step,
                )
                print(
                    f"{task_id}.{subtask_id}: assistant stopped without tool call at step {step}; "
                    f"retry {attempt_ctx.text_only_retries}/{TEXT_ONLY_RETRY_MAX}."
                )
                messages.append({
                    "role": "user",
                    "content": _text_only_retry_message(
                        task_id,
                        subtask_id,
                        step,
                        next_action=_next_action_hint(active_session, task_id),
                    ),
                })
                thinking_effort = RECOVERY_EFFORT
                continue
            print(f"{task_id}.{subtask_id}: assistant stopped without tool call at step {step}.")
            stopped_without_tool_call = True
            break

        messages.append(assistant_message)
        if show_model_output:
            print_visible_model_output(task_id, step, content)
        if active_session is not None and content.strip():
            active_session.record_visible_output(task_id, content)

        tool_results = tool_compact.execute_tool_batch(tool_calls)
        for result in tool_results:
            run_metrics.record_tool(result.name, result.status, result.arguments)
        thinking_effort = effort_after_tool_results(tool_results)
        if any(result.status == "ok" for result in tool_results):
            attempt_ctx.text_only_retries = 0
        if any(
            result.status == "ok" and result.name in {"write_file", "replace_in_file"}
            for result in tool_results
        ):
            attempt_ctx.syntax_write_retries = 0
        if (
            len(tool_results) > 1
            and all(result.name in READ_ONLY_TOOLS for result in tool_results)
        ):
            print(
                f"{task_id}.{subtask_id}: tool step {step}: "
                f"parallel readonly batch ({len(tool_results)} tools)"
            )

        for result in tool_results:
            print(
                f"{task_id}.{subtask_id}: tool step {step}: "
                f"{result.name}({json.dumps(result.arguments, ensure_ascii=False)})"
            )
            previous_handoff_id = tool_compact.append_compact_tool_result(
                messages=messages,
                task_id=task_id,
                previous_handoff_id=previous_handoff_id,
                tool_name=result.name,
                arguments=result.arguments,
                status=result.status,
                raw_result=result.raw_result,
                merged_subtask=merged_subtask,
            )

            if role == "test_writer" and is_declared_verification_failure(
                tool_name=result.name,
                arguments=result.arguments,
                status=result.status,
                output=result.raw_result,
                merged_subtask=merged_subtask or {},
            ):
                logger.info(
                    "%s: declared verification is red; yielding test-writer",
                    scope,
                )
                state = _tasks.load_task_state(task_id) or {}
                return WorkerResult(
                    ok=True,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                    ready_for_verification=False,
                )

            if role == "test_writer" and is_declared_verification_success(
                tool_name=result.name,
                arguments=result.arguments,
                status=result.status,
                output=result.raw_result,
                merged_subtask=merged_subtask or {},
            ):
                logger.info(
                    "%s: declared verification is already green; yielding to verifier",
                    scope,
                )
                persist_and_sync(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    attempt=attempt_number,
                    status="in_progress",
                    merged_subtask=merged_subtask,
                    session=active_session,
                    stop_reason="already_satisfied",
                    last_handoff_id=previous_handoff_id,
                    changed_files=list(
                        (_tasks.load_task_state(task_id) or {}).get(
                            "changedFilesSinceAttemptStart"
                        )
                        or []
                    ),
                    recommended_resume_step=(
                        "Existing implementation already satisfies the declared tests; "
                        "the coordinator will mark the subtask DONE after the verifier."
                    ),
                )
                state = _tasks.load_task_state(task_id) or {}
                return WorkerResult(
                    ok=True,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                    ready_for_verification=True,
                )

            if role != "test_writer" and is_declared_verification_success(
                tool_name=result.name,
                arguments=result.arguments,
                status=result.status,
                output=result.raw_result,
                merged_subtask=merged_subtask or {},
            ):
                logger.info(
                    "%s: declared verification passed; yielding to coordinator",
                    scope,
                )
                state = _tasks.load_task_state(task_id) or {}
                return WorkerResult(
                    ok=True,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                    ready_for_verification=True,
                )

            if role != "test_writer" and is_subtask_done(task_id, subtask_id):
                state = _tasks.load_task_state(task_id) or {}
                return WorkerResult(
                    ok=True,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                )

            state = _tasks.load_task_state(task_id) or {}
            if state.get("status") in ("FAILED", "BLOCKED"):
                return WorkerResult(
                    ok=False,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                )

        syntax_failures = _syntax_write_failures(tool_results)
        if syntax_failures:
            last_failure = syntax_failures[-1]
            diagnostic = str(last_failure.raw_result)
            retries = getattr(attempt_ctx, "syntax_write_retries", 0) + 1
            attempt_ctx.syntax_write_retries = retries
            path = str(last_failure.arguments.get("path") or "").strip()
            if retries > SYNTAX_WRITE_RETRY_MAX:
                failure_reason = diagnostic
                persist_and_sync(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    attempt=attempt_number,
                    status="failed",
                    merged_subtask=merged_subtask,
                    session=active_session,
                    stop_reason="python_syntax_write",
                    blockers=[failure_reason],
                    last_handoff_id=previous_handoff_id,
                    changed_files=list(
                        (_tasks.load_task_state(task_id) or {}).get(
                            "changedFilesSinceAttemptStart"
                        )
                        or []
                    ),
                )
                state = _tasks.update_task_state(
                    task_id,
                    "FAILED",
                    summary=(
                        f"Python syntax write refused {retries} times "
                        f"for subtask {subtask_id}."
                    ),
                    blockers=[failure_reason],
                    last_failure_reason=failure_reason,
                )
                print(
                    f"{task_id}.{subtask_id}: python syntax write retries "
                    f"exhausted ({retries}/{SYNTAX_WRITE_RETRY_MAX})."
                )
                return WorkerResult(
                    ok=False,
                    state=state,
                    previous_handoff_id=previous_handoff_id,
                )
            messages.append({
                "role": "user",
                "content": _syntax_write_retry_message(
                    task_id=task_id,
                    subtask_id=subtask_id,
                    tool_name=last_failure.name,
                    path=path,
                    diagnostic=diagnostic,
                    attempt=retries,
                    limit=SYNTAX_WRITE_RETRY_MAX,
                ),
            })
            thinking_effort = RECOVERY_EFFORT
            print(
                f"{task_id}.{subtask_id}: python syntax write refused; "
                f"forcing retry {retries}/{SYNTAX_WRITE_RETRY_MAX}."
            )

    state = _tasks.load_task_state(task_id) or {}
    if role != "test_writer" and is_subtask_done(task_id, subtask_id):
        return WorkerResult(ok=True, state=state, previous_handoff_id=previous_handoff_id)

    if stopped_without_tool_call:
        failure_reason = f"Assistant stopped without tool call at step {last_step}."
        stop_reason = "no_tool_call"
        summary = f"Worker stopped before marking subtask {subtask_id} DONE."
    else:
        failure_reason = (
            f"Worker exhausted tool step budget ({max_tool_steps} steps) "
            f"without marking subtask {subtask_id} DONE."
        )
        stop_reason = "tool_step_budget"
        summary = f"Worker exhausted tool step budget for subtask {subtask_id}."

    persist_and_sync(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt=attempt_number,
        status="failed",
        merged_subtask=merged_subtask,
        session=active_session,
        stop_reason=stop_reason,
        blockers=[failure_reason],
        last_handoff_id=previous_handoff_id,
        changed_files=list(state.get("changedFilesSinceAttemptStart") or []),
    )
    state = _tasks.update_task_state(
        task_id,
        "FAILED",
        summary=summary,
        blockers=[failure_reason],
        last_failure_reason=failure_reason,
    )
    return WorkerResult(ok=False, state=state, previous_handoff_id=previous_handoff_id)
