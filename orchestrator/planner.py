"""Read-only LLM planner for enriching subtask specs when golden rules fail."""

from __future__ import annotations

import json
import re
from typing import Any

from orchestrator import tool_compact
from orchestrator import config
from orchestrator.ollama_client import ollama_chat
from orchestrator.project import PROJECT_PROFILE
from orchestrator.spec_readiness import ReadinessResult
from orchestrator.thinking_policy import PLANNER_EFFORT
from orchestrator.tools_registry import TOOLS
from orchestrator.worker_context import begin_worker_attempt, clear_worker_context

PLANNER_TOOL_NAMES = frozenset({
    "read_file",
    "search_files",
    "extract_symbol",
    "find_symbol",
    "list_directory",
})

PLANNER_TOOLS = [tool for tool in TOOLS if tool["function"]["name"] in PLANNER_TOOL_NAMES]
MAX_PLANNER_TOOL_STEPS = 20


def plan_sidecar_path(task_id: str, subtask_id: str) -> str:
    return str(config.PLANS_DIR / f"{task_id}.{subtask_id}.plan.json")


def load_plan_sidecar(task_id: str, subtask_id: str) -> dict[str, Any] | None:
    path = config.PLANS_DIR / f"{task_id}.{subtask_id}.plan.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_plan_sidecar(payload: dict[str, Any]) -> str:
    config.PLANS_DIR.mkdir(parents=True, exist_ok=True)
    task_id = str(payload["taskId"])
    subtask_id = str(payload["subtaskId"])
    path = config.PLANS_DIR / f"{task_id}.{subtask_id}.plan.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(path)


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Planner response did not contain a JSON object.")
    return json.loads(cleaned[start : end + 1])


def _planner_system_prompt() -> str:
    return (
        f"You are a read-only {PROJECT_PROFILE.name} task planner. Inspect the repository with tools, "
        "then output ONE JSON object only. Do not write code files. Do not call "
        "update_task_state or update_subtask_state. "
        "Follow the enrichments schema: legacySymbols, verification, targetArtifacts, "
        "instructions, allowedSearchTerms, scope, readLegacy, dataExamples."
    )


def _planner_user_prompt(
    task_id: str,
    subtask_id: str,
    context_pack: dict[str, Any],
    readiness: ReadinessResult,
    planner_round: int,
) -> str:
    return (
        f"Plan enrichments for {task_id} subtask {subtask_id} (round {planner_round}).\n"
        f"Failing rule ids: {readiness.rule_ids}\n"
        f"Blockers:\n- " + "\n- ".join(readiness.blockers) + "\n"
        "Return JSON with keys: taskId, subtaskId, plannerRound, enrichments, addressedRuleIds.\n"
        "Use existing enriched subtasks as the reference shape for task metadata.\n"
        f"Context pack:\n{json.dumps(context_pack, indent=2, ensure_ascii=False)}"
    )


def run_llm_planner(
    task_id: str,
    subtask_id: str,
    context_pack: dict[str, Any],
    readiness: ReadinessResult,
    *,
    model: str,
    planner_round: int,
    num_ctx: int | None = None,
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _planner_system_prompt()},
        {
            "role": "user",
            "content": _planner_user_prompt(
                task_id,
                subtask_id,
                context_pack,
                readiness,
                planner_round,
            ),
        },
    ]

    attempt_ctx = begin_worker_attempt(task_id)
    previous_handoff_id = ""
    final_content = ""
    resolved_num_ctx = num_ctx if num_ctx is not None else config.DEFAULT_NUM_CTX
    try:
        for step in range(1, MAX_PLANNER_TOOL_STEPS + 1):
            response = ollama_chat(
                model=model,
                messages=messages,
                tools=PLANNER_TOOLS,
                num_ctx=resolved_num_ctx,
                think=PLANNER_EFFORT,
            )
            message = response.message
            if hasattr(message, "model_dump"):
                assistant_message = message.model_dump(exclude_none=True)
            elif isinstance(message, dict):
                assistant_message = message
            else:
                assistant_message = {
                    "role": getattr(message, "role", "assistant"),
                    "content": getattr(message, "content", ""),
                    "tool_calls": getattr(message, "tool_calls", None),
                }
            messages.append(assistant_message)
            final_content = assistant_message.get("content") or final_content

            tool_calls = assistant_message.get("tool_calls") or []
            if not tool_calls:
                break

            for tool_call in tool_calls:
                name, arguments = tool_compact.get_tool_name_and_args(tool_call)
                if name not in PLANNER_TOOL_NAMES:
                    status, raw_result = "error", f"ERROR: Planner tool not allowed: {name}"
                else:
                    status, raw_result = tool_compact.execute_tool(name, arguments)
                previous_handoff_id = tool_compact.append_compact_tool_result(
                    messages=messages,
                    task_id=task_id,
                    previous_handoff_id=previous_handoff_id,
                    tool_name=name,
                    arguments=arguments,
                    status=status,
                    raw_result=raw_result,
                )
    finally:
        clear_worker_context()

    if not final_content.strip():
        raise ValueError("Planner produced no JSON content.")

    parsed = _extract_json_object(final_content)
    payload = {
        "taskId": task_id,
        "subtaskId": subtask_id,
        "plannerRound": planner_round,
        "enrichments": parsed.get("enrichments") or parsed,
        "addressedRuleIds": parsed.get("addressedRuleIds") or readiness.rule_ids,
    }
    save_plan_sidecar(payload)
    return payload
