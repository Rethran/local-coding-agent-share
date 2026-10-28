from __future__ import annotations

from typing import Callable

from tools.create_folder import create_folder
from tools.extract_symbol import extract_symbol
from tools.find_symbol import find_symbol
from tools.list_directory import list_directory
from tools.replace_in_file import replace_in_file
from tools.write_file import write_file

from orchestrator.tool_adapters import (
    delete_handoff_chain,
    get_active_subtask_tool,
    get_subtask_context_tool,
    get_verification_checklist,
    read_handoff,
    record_handoff_tool,
    safe_read_file,
    safe_run_command,
    safe_search_files,
    update_subtask_state_tool,
    update_task_state_tool,
)

TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "read_file": safe_read_file,
    "write_file": write_file,
    "replace_in_file": replace_in_file,
    "create_folder": create_folder,
    "run_command": safe_run_command,
    "search_files": safe_search_files,
    "find_symbol": find_symbol,
    "extract_symbol": extract_symbol,
    "list_directory": list_directory,
    "update_task_state": update_task_state_tool,
    "get_verification_checklist": get_verification_checklist,
    "get_active_subtask": get_active_subtask_tool,
    "get_subtask_context": get_subtask_context_tool,
    "update_subtask_state": update_subtask_state_tool,
    "record_handoff": record_handoff_tool,
    "read_handoff": read_handoff,
    "delete_handoff_chain": delete_handoff_chain,
}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a UTF-8 text file by workspace-relative path. Use only "
                "for small files, non-Python assets (XML/JSON/config), or code "
                "under app/ and tests/. Never use on handoff JSON — call "
                "read_handoff instead. For one function or class in a large "
                "legacy .py module, use extract_symbol instead of read_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative path to a small UTF-8 text "
                            "file. Examples: 'app/domain/prices.py', "
                            "'old_entsoes/entsoe_data.xml'. Do not pass "
                            "paths under local-coding-agent/handoffs/."
                        ),
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write a complete UTF-8 file by workspace-relative path. Use only "
                "for new or small files. For focused edits to existing files, use "
                "replace_in_file with a small unique old_text/new_text pair."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Atomically replace exactly one matching text block in an existing "
                "UTF-8 file. Keep old_text and new_text small and include enough "
                "surrounding context to make old_text unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to an existing file.",
                    },
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_folder",
            "description": "Create a folder by workspace-relative path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a non-destructive test or inspection command in the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search text files for a case-insensitive substring. "
                "path may be a file or a directory. Orchestrator worker enforces "
                "subtask allowedSearchTerms on legacy paths and blocks "
                "old_entsoes/*.py when legacySymbols is defined; prefer "
                "get_subtask_context or extract_symbol for legacy Python."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_symbol",
            "description": "Find Python class/function definitions by exact symbol name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_symbol",
            "description": (
                "Extract one Python function/class source body by exact name. "
                "Prefer over read_file for large legacy modules. Optional path "
                "narrows to a file or directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List immediate entries in a workspace-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_task_state",
            "description": "Update HEAT task runtime state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                    "files_touched": {"type": "string"},
                    "blockers": {"type": "string"},
                    "verification_result": {"type": "string"},
                    "last_failure_reason": {"type": "string"},
                    "last_recovery_action": {"type": "string"},
                    "changed_files_since_attempt_start": {"type": "string"},
                    "progress_summary": {"type": "string"},
                    "completed_scope_items": {
                        "type": "string",
                        "description": "Pipe-separated scope items completed before failure.",
                    },
                    "remaining_scope_items": {
                        "type": "string",
                        "description": "Pipe-separated scope items still remaining.",
                    },
                    "verified_behaviors": {
                        "type": "string",
                        "description": "Pipe-separated behaviors verified before failure.",
                    },
                    "partial_files_touched": {
                        "type": "string",
                        "description": "Comma-separated files partially changed before failure.",
                    },
                    "recommended_resume_step": {"type": "string"},
                },
                "required": ["task_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_verification_checklist",
            "description": "Get acceptance criteria, verification steps, and suggested commands for one HEAT task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_subtask_context",
            "description": (
                "Return curated implementation context for the active subtask: "
                "legacy source slices, fixture snippet, target artifacts, and "
                "next steps. Prefer over search_files when legacySymbols is present."
            ),
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_subtask",
            "description": "Get active subtask and subtask progress for one HEAT task.",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_subtask_state",
            "description": "Update progress for one HEAT subtask without marking the whole task done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "subtask_id": {"type": "string"},
                    "status": {"type": "string"},
                    "summary": {"type": "string"},
                },
                "required": ["task_id", "subtask_id", "status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_handoff",
            "description": "Read a compact handoff record by id or path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "handoff_id": {"type": "string"},
                    "handoff_path": {"type": "string"},
                    "include_result": {"type": "boolean", "default": False},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_handoff",
            "description": "Manually store a compact handoff record in the active task chain.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "input_json": {"type": "string"},
                    "result_json": {"type": "string"},
                    "summary": {"type": "string"},
                    "previous_handoff_id": {"type": "string"},
                    "chain_id": {"type": "string"},
                },
                "required": ["task_id", "tool", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_handoff_chain",
            "description": "Consume or delete all active handoffs for a task chain after the task is done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chain_id": {"type": "string"},
                    "mode": {"type": "string", "default": "consume"},
                },
                "required": ["chain_id"],
            },
        },
    },
]
