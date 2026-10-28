import json
from typing import Annotated

from pydantic import Field

from tools._common import mcp
from tools._symbols import scan_for_extract_matches


@mcp.tool()
def extract_symbol(
    name: Annotated[
        str,
        Field(
            description=(
                "Exact Python symbol name to extract — matched case-sensitively "
                "against class, function, and async function definitions. "
                "Example: 'haehinnat'."
            )
        ),
    ],
    path: Annotated[
        str,
        Field(
            description=(
                "Optional workspace-relative file or directory that narrows "
                "the scan. Use a file path to extract from one module, a "
                "directory to scan that subtree, or '.' for the whole "
                "workspace. Example: 'old_entsoes/entsoe_prod.py'."
            )
        ),
    ] = ".",
    start_line: Annotated[
        int | None,
        Field(
            description=(
                "Optional 1-indexed start line for an absolute file slice. "
                "When used with end_line, returns only those lines from path."
            )
        ),
    ] = None,
    end_line: Annotated[
        int | None,
        Field(
            description=(
                "Optional 1-indexed end line for an absolute file slice. "
                "When used with start_line, returns only those lines from path."
            )
        ),
    ] = None,
) -> str:
    """Extract a Python function or class source body by exact symbol name.

    Prefer this over read_file when inspecting one definition in a large
    legacy module. Returns the definition (including decorators) with line
    numbers, without loading the rest of the file into context.

    When start_line and end_line are provided with a file path, returns that
    absolute line range instead of the full symbol body.

    Prefer find_symbol when:
        - You only need location metadata (file + line range), not source.

    Prefer search_files when:
        - You only know part of a name or want case-insensitive text search.

    Returns:
        A JSON array of matches. Each object contains:
            - ``file``: path relative to the workspace root
            - ``name``: matched symbol name
            - ``type``: ``FunctionDef``, ``AsyncFunctionDef``, or ``ClassDef``
            - ``start_line`` / ``end_line``: inclusive 1-indexed source range
            - ``truncated``: true when ``source`` was capped for size
            - ``source``: the definition body (decorators included)

        An empty array ``[]`` means no definition was found — not an error.
        Oversized bodies are truncated around 10000 characters with a marker.

    Raises:
        ValueError: if ``path`` does not exist or is outside the workspace.
    """
    if (start_line is None) ^ (end_line is None):
        raise ValueError("start_line and end_line must both be provided or both omitted.")
    matches = scan_for_extract_matches(
        name,
        path,
        start_line=start_line,
        end_line=end_line,
    )
    return json.dumps(matches, indent=2)
