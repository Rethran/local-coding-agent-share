"""Middleware that repairs malformed <tool_call> inputs and retries automatically."""

from __future__ import annotations

import json

import mcp.types as mt
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools.base import ToolResult

from tools._repair import (
    RepairMemory,
    RepairResult,
    looks_like_xml_tool_call,
    repair_arguments,
    repair_output_text,
)

MAX_REPAIR_RETRIES = 2

# Errors that retry/repair cannot fix — fail fast with the original message.
NON_RECOVERABLE_MARKERS = (
    "File does not exist:",
    "Directory does not exist:",
    "Not a file:",
    "Not a directory:",
    "Path outside workspace:",
)


def _is_non_recoverable_error(exc: Exception) -> bool:
    message = str(exc)
    return any(marker in message for marker in NON_RECOVERABLE_MARKERS)


def _reraise_tool_error(exc: Exception) -> None:
    """Surface the real tool error without XML-repair noise."""
    if isinstance(exc, ToolError):
        raise exc
    raise ToolError(str(exc)) from exc


def _extract_text(result: ToolResult) -> str:
    parts = [c.text for c in result.content if hasattr(c, "text") and c.text]
    return "\n".join(parts)


def _prepend_repair_notice(result: ToolResult, notes: list[str]) -> ToolResult:
    if not notes:
        return result
    notice = "[Auto-repair] " + " | ".join(notes)
    if result.content and hasattr(result.content[0], "text"):
        first = result.content[0]
        patched = first.model_copy(update={"text": f"{notice}\n\n{first.text}"})
        return result.model_copy(update={"content": [patched, *result.content[1:]]})
    return result


class RepairMiddleware(Middleware):
    """Repairs XML-style tool calls in arguments and retries on recoverable failures."""

    def __init__(self, memory: RepairMemory | None = None):
        self.memory = memory or RepairMemory.load()

    async def on_call_tool(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        original_tool = context.message.name
        original_args = dict(context.message.arguments or {})

        repair = repair_arguments(original_tool, original_args, self.memory)
        last_error: Exception | None = None

        for attempt in range(MAX_REPAIR_RETRIES + 1):
            try:
                result = await self._dispatch(context, call_next, repair)
                text = _extract_text(result)

                output_repair = repair_output_text(text, self.memory)
                if output_repair is not None and attempt < MAX_REPAIR_RETRIES:
                    repair = output_repair
                    continue

                if looks_like_xml_tool_call(text) and attempt < MAX_REPAIR_RETRIES:
                    repair = RepairResult(
                        tool_name=original_tool,
                        arguments=original_args,
                        was_repaired=True,
                        repair_notes=["Output looked like XML tool-call text; retrying"],
                    )
                    continue

                if repair.was_repaired:
                    self.memory.record_success(repair, original_tool, original_args)
                return _prepend_repair_notice(result, repair.repair_notes)

            except (ToolError, ValueError, TypeError, json.JSONDecodeError) as exc:
                if _is_non_recoverable_error(exc):
                    _reraise_tool_error(exc)

                last_error = exc
                if attempt >= MAX_REPAIR_RETRIES:
                    break

                new_repair = repair_arguments(original_tool, original_args, self.memory)
                if not new_repair.was_repaired:
                    _reraise_tool_error(exc)

                repair = new_repair
                repair.repair_notes.append(f"Retry after error: {exc}")

        if _is_non_recoverable_error(last_error) if last_error else False:
            _reraise_tool_error(last_error)

        raise ToolError(
            _format_failure_message(original_tool, original_args, last_error, repair)
        )

    async def _dispatch(
        self,
        context: MiddlewareContext[mt.CallToolRequestParams],
        call_next: CallNext[mt.CallToolRequestParams, ToolResult],
        repair: RepairResult,
    ) -> ToolResult:
        if repair.redirect_tool and context.fastmcp_context is not None:
            return await context.fastmcp_context.fastmcp.call_tool(
                repair.tool_name,
                repair.arguments,
                run_middleware=False,
            )

        message = mt.CallToolRequestParams(
            name=repair.tool_name,
            arguments=repair.arguments,
        )
        return await call_next(context.copy(message=message))


def _format_failure_message(
    tool_name: str,
    arguments: dict,
    error: Exception | None,
    repair: RepairResult,
) -> str:
    if error is not None and not repair.was_repaired:
        return str(error)

    lines = [f"Tool '{tool_name}' failed after {MAX_REPAIR_RETRIES} repair retries."]
    if repair.was_repaired:
        lines.append(
            "Arguments were auto-repaired from <tool_call> XML but execution still failed."
        )
    if repair.repair_notes:
        lines.append("Repair attempts: " + "; ".join(repair.repair_notes))
    if error is not None:
        lines.append(f"Last error: {error}")
    if arguments:
        preview = json.dumps(arguments, ensure_ascii=False)[:300]
        lines.append(f"Received arguments preview: {preview}")
    return "\n".join(lines)
