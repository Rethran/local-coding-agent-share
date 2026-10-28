"""MCP tools for HEAT task graph orchestration in Continue chat."""

import json

from tools._common import mcp
from tools import _handoffs, _tasks


def _json_response(payload: dict) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _error(message: str) -> str:
    return _json_response({"error": message})


def _with_handoff(
    *,
    task_id: str,
    tool_name: str,
    input_data: dict,
    result: dict,
    summary: str,
    next_action: dict | None = None,
    previous_handoff_id: str = "",
    chain_id: str = "",
    prompt_action: str = "implement",
) -> dict:
    handoff = _handoffs.record_handoff(
        task_id=task_id,
        tool=tool_name,
        input_data=input_data,
        result=result,
        summary=summary,
        next_action=next_action,
        previous_handoff_id=previous_handoff_id,
        chain_id=chain_id or task_id,
    )
    compact = handoff["compact"]
    prompt = _tasks.build_handoff_prompt(
        task_id=task_id,
        handoff_id=compact["handoffId"],
        handoff_path=compact["handoffPath"],
        action=prompt_action,
    )
    handoff = _handoffs.set_copy_paste_prompt(handoff, prompt)
    return {
        **result,
        "handoff": handoff["compact"],
    }


@mcp.tool()
def list_tasks(from_id: str = "", to_id: str = "") -> str:
    """
    List HEAT tasks with computed ready/done/blocked status.

    Scans local-coding-agent/requirements/HEAT-*.json and merges runtime state files.
    Use from_id/to_id to limit range (e.g. HEAT-001 through HEAT-005, or HEAT-003.4
    through HEAT-004). Subtask suffixes pin/stop inside the parent task; the numeric
    range still uses parent task ids.

    Returns JSON with tasks[], ready[], done[], blocked[].
    """
    try:
        graph = _tasks.build_task_graph(
            from_id=from_id or None,
            to_id=to_id or None,
        )
        task_id = graph["ready"][0] if graph["ready"] else (graph["tasks"][0]["id"] if graph["tasks"] else "HEAT-000")
        next_action = graph.get("nextAction")
        result = _with_handoff(
            task_id=task_id,
            tool_name="list_tasks",
            input_data={"from_id": from_id, "to_id": to_id},
            result=graph,
            summary=f"Task graph loaded. Ready: {graph['ready']}. Blocked: {graph['blocked']}.",
            next_action=next_action,
            prompt_action="read",
        )
        return _json_response(result)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def get_task(task_id: str) -> str:
    """
    Load a single HEAT task definition plus current state.

    Returns one task packet only — use this as the worker context for the active task.
    """
    try:
        packet = _tasks.get_task_packet(task_id)
        result = _with_handoff(
            task_id=task_id,
            tool_name="get_task",
            input_data={"task_id": task_id},
            result=packet,
            summary=f"{task_id} task packet loaded.",
            next_action={
                "tool": "local_agent_update_task_state",
                "arguments": {
                    "task_id": task_id,
                    "status": "IN_PROGRESS",
                },
            },
        )
        return _json_response(result)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def update_task_state(
    task_id: str,
    status: str,
    summary: str = "",
    files_touched: str = "",
    blockers: str = "",
    verification_result: str = "",
) -> str:
    """
    Create or update HEAT-XXX.state.json for runtime orchestration.

    status: PENDING | READY | IN_PROGRESS | VERIFYING | DONE | FAILED | BLOCKED
    files_touched: comma-separated paths (optional)
    blockers: comma-separated blocker descriptions (optional)
    verification_result: short pass/fail note when marking DONE or FAILED (optional)
    """
    try:
        files_list = [p.strip() for p in files_touched.split(",") if p.strip()] or None
        blockers_list = [b.strip() for b in blockers.split(",") if b.strip()] or None
        state = _tasks.update_task_state(
            task_id,
            status,
            summary=summary or None,
            files_touched=files_list,
            blockers=blockers_list,
            verification_result=verification_result or None,
        )
        return _json_response(state)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def get_verification_checklist(task_id: str) -> str:
    """
    Return agent-driven verification checklist for a task.

    Includes acceptanceCriteria and verification steps from task JSON.
    """
    try:
        checklist = _tasks.build_verification_checklist(task_id)
        commands = [
            item for item in checklist.get("verification") or []
            if isinstance(item, str) and item.strip()
        ]
        result = _with_handoff(
            task_id=task_id,
            tool_name="get_verification_checklist",
            input_data={"task_id": task_id},
            result=checklist,
            summary=f"{task_id} verification checklist loaded.",
            next_action={
                "tool": "local_agent_run_command",
                "arguments": {"command": commands[0]},
            } if commands else None,
        )
        return _json_response(result)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def get_resume_brief(task_id: str) -> str:
    """
    Compact handoff payload when resuming a task in a new Continue chat.

    Use after context budget is hit or when user says 'Resume HEAT-XXX'.
    """
    try:
        brief = _tasks.build_resume_brief(task_id)
        result = _with_handoff(
            task_id=task_id,
            tool_name="get_resume_brief",
            input_data={"task_id": task_id},
            result=brief,
            summary=f"{task_id} resume brief loaded.",
        )
        return _json_response(result)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def prepare_next_handoff(from_id: str = "HEAT-001", to_id: str = "HEAT-005") -> str:
    """
    Pick active work or the first READY task and write a compact handoff.

    Use this as the preferred one-call entry point for fragile local models.
    """
    try:
        selection = _tasks.select_next_task_id(from_id or None, to_id or None)
        task_id = selection["taskId"]
        if task_id is None:
            return _json_response({
                "error": "No active or ready task found",
                "graph": selection["graph"],
            })

        packet = _tasks.get_task_packet(task_id)
        result = {
            "selectedTaskId": task_id,
            "selectionReason": selection["reason"],
            "graph": selection["graph"],
            "task": packet["task"],
            "state": packet["state"],
        }
        next_action = {
            "tool": "local_agent_read_handoff",
            "arguments": {},
        }
        result = _with_handoff(
            task_id=task_id,
            tool_name="prepare_next_handoff",
            input_data={"from_id": from_id, "to_id": to_id},
            result=result,
            summary=f"Prepared handoff for {task_id} ({selection['reason']}).",
            next_action=next_action,
        )
        result["oneCallInstruction"] = "Stop after this tool result and copy copyPastePrompt into a fresh chat if needed."
        return _json_response(result)
    except ValueError as exc:
        return _error(str(exc))


@mcp.tool()
def read_handoff(handoff_id: str = "", handoff_path: str = "", include_result: bool = False) -> str:
    """
    Read a stored handoff by id or path.

    Use as the first call in a fresh chat that continues a handoff chain.
    """
    try:
        record = _handoffs.read_handoff_record(handoff_id, handoff_path)
        response = {
            "handoffId": record["handoffId"],
            "previousHandoffId": record.get("previousHandoffId"),
            "stepIndex": record["stepIndex"],
            "chainId": record["chainId"],
            "taskId": record["taskId"],
            "tool": record["tool"],
            "summary": record.get("summary", ""),
            "nextAction": record.get("nextAction"),
            "copyPastePrompt": record.get("copyPastePrompt", ""),
        }
        if include_result:
            response["result"] = record.get("result")
        return _json_response(response)
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc))


@mcp.tool()
def record_handoff(
    task_id: str,
    tool: str,
    input_json: str = "",
    result_json: str = "",
    summary: str = "",
    next_action_json: str = "",
    previous_handoff_id: str = "",
    chain_id: str = "",
) -> str:
    """
    Manually store a tool result as the next handoff in a chain.

    JSON arguments are strings so local models can pass them through MCP reliably.
    """
    try:
        input_data = _handoffs.parse_json_object(input_json, "input_json")
        result_data = _handoffs.parse_json_value(result_json)
        next_action = _handoffs.parse_json_object(next_action_json, "next_action_json") if next_action_json else None
        handoff = _handoffs.record_handoff(
            task_id=task_id,
            tool=tool,
            input_data=input_data,
            result=result_data,
            summary=summary,
            next_action=next_action,
            previous_handoff_id=previous_handoff_id,
            chain_id=chain_id,
        )
        compact = handoff["compact"]
        prompt = _tasks.build_handoff_prompt(
            task_id=task_id,
            handoff_id=compact["handoffId"],
            handoff_path=compact["handoffPath"],
        )
        handoff = _handoffs.set_copy_paste_prompt(handoff, prompt)
        return _json_response(handoff["compact"])
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc))


@mcp.tool()
def delete_handoff(handoff_id: str = "", handoff_path: str = "", mode: str = "consume") -> str:
    """
    Consume or delete one handoff after it has been used successfully.

    mode: consume (default) or delete.
    """
    try:
        return _json_response(_handoffs.delete_or_consume_handoff(
            handoff_id=handoff_id,
            handoff_path=handoff_path,
            mode=mode,
        ))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc))


@mcp.tool()
def delete_handoff_chain(chain_id: str, mode: str = "consume") -> str:
    """
    Consume or delete all active handoffs in a chain after the task is DONE.

    mode: consume (default) or delete.
    """
    try:
        return _json_response(_handoffs.delete_or_consume_chain(chain_id, mode))
    except (ValueError, json.JSONDecodeError) as exc:
        return _error(str(exc))
