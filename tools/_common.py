import json
import sys
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext

from orchestrator.project import PROJECT_PROFILE

ARCH_MCP_DIR = Path(__file__).resolve().parent.parent.parent / "arch-mcp"
sys.path.insert(0, str(ARCH_MCP_DIR))

WORKSPACE_ROOT = PROJECT_PROFILE.workspace_root

JSON_VALIDATED_TOOLS = {
    "search_files",
    "find_symbol",
    "extract_symbol",
    "analyze_project",
    "extract_structure",
    "map_summarize_modules",
    "reduce_architecture_review",
    "analyze_architecture",
    "list_tasks",
    "get_task",
    "update_task_state",
    "get_verification_checklist",
    "get_resume_brief",
    "prepare_next_handoff",
    "read_handoff",
    "record_handoff",
    "delete_handoff",
    "delete_handoff_chain",
}

IGNORED_DIRS = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
})


class ValidationMiddleware(Middleware):
    async def on_call_tool(self, context: MiddlewareContext, call_next):
        result = await call_next(context)
        tool_name = context.message.name

        text_parts = [c.text for c in result.content if hasattr(c, "text")]
        combined = "\n".join(text_parts)

        if tool_name in JSON_VALIDATED_TOOLS:
            try:
                json.loads(combined)
            except json.JSONDecodeError:
                raise ToolError(
                    f"Validointivirhe: Työkalu '{tool_name}' palautti "
                    f"viallista JSON-rakennetta. Korjaa tuloste."
                )

        return result


mcp = FastMCP("local-agent")

from tools._repair_middleware import RepairMiddleware

mcp.add_middleware(RepairMiddleware())
mcp.add_middleware(ValidationMiddleware())


def ensure_directory_exists(directory: Path, original_path: str) -> None:
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {original_path}")
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {original_path}")


def ensure_file_exists(file_path: Path, original_path: str) -> None:
    if not file_path.exists():
        suggestion = suggest_paths_by_filename(original_path)
        message = f"File does not exist: {original_path}"
        if suggestion:
            message += f". Did you mean: {suggestion}?"
        raise ValueError(message)
    if not file_path.is_file():
        raise ValueError(f"Not a file: {original_path}")


def suggest_paths_by_filename(relative_path: str, limit: int = 3) -> str | None:
    """Find workspace files that share the same basename as a missing path."""
    basename = Path(relative_path).name
    if not basename:
        return None

    matches = [
        str(path.relative_to(WORKSPACE_ROOT))
        for path in WORKSPACE_ROOT.rglob(basename)
        if path.is_file() and not is_within_ignored_directory(path)
    ]

    if not matches:
        return None

    return ", ".join(sorted(matches)[:limit])


def safe_path(path: str) -> Path:
    candidate = (WORKSPACE_ROOT / path).resolve()

    if candidate != WORKSPACE_ROOT and WORKSPACE_ROOT not in candidate.parents:
        raise ValueError(f"Path outside workspace: {path}")

    return candidate


def is_within_ignored_directory(file: Path) -> bool:
    return any(part in IGNORED_DIRS for part in file.parts)


def find_matching_lines(file: Path, query: str) -> list[dict]:
    try:
        content = file.read_text(encoding="utf-8")
    except (UnicodeDecodeError, PermissionError, OSError):
        return []

    query_lower = query.lower()
    matches = []
    for line_number, line in enumerate(content.splitlines(), start=1):
        if query_lower in line.lower():
            matches.append({
                "file": str(file.relative_to(WORKSPACE_ROOT)),
                "line": line_number,
                "text": line.strip(),
            })
    return matches
