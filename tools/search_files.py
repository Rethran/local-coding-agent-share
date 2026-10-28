import json
from typing import Annotated

from pydantic import Field

from tools._common import (
    find_matching_lines,
    is_within_ignored_directory,
    mcp,
    safe_path,
)


@mcp.tool()
def search_files(
    query: Annotated[
        str,
        Field(
            description=(
                "Text to search for, matched case-insensitively as a plain "
                "substring — not a regular expression. Matches anywhere "
                "within a line. Example: 'ValidationMiddleware'."
            )
        ),
    ],
    path: Annotated[
        str,
        Field(
            description=(
                "File or directory to search within, given as a path relative "
                "to the project workspace root (never an absolute path). "
                "When path is a file, only that file is searched. When path "
                "is a directory, searches recursively through all "
                "subdirectories. Use '.' to search the entire workspace. "
                "Common noise directories (.git, .venv, node_modules, "
                "__pycache__, etc.) are always skipped automatically. "
                "Example: 'old_entsoes/entsoe_prod.py' or 'local-coding-agent'."
            )
        ),
    ] = ".",
) -> str:
    """Search for a text string across readable text files in the workspace.

    ``path`` may be a single file or a directory. Directory searches are
    recursive.

    Returns:
        A JSON array of matches, each with 'file' (relative path), 'line'
        (1-indexed line number), and 'text' (the matching line, trimmed).
        An empty array means no matches were found, not an error.

    Raises:
        ValueError: if the path does not exist, is outside the workspace,
            or is neither a file nor a directory.
    """
    root = safe_path(path)
    if not root.exists():
        raise ValueError(f"Path does not exist: {path}")

    if root.is_file():
        if is_within_ignored_directory(root):
            return json.dumps([], indent=2)
        return json.dumps(find_matching_lines(root, query), indent=2)

    if not root.is_dir():
        raise ValueError(f"Not a file or directory: {path}")

    results = [
        match
        for file in root.rglob("*")
        if file.is_file() and not is_within_ignored_directory(file)
        for match in find_matching_lines(file, query)
    ]

    return json.dumps(results, indent=2)
