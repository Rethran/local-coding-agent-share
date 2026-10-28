import json
from typing import Annotated

from pydantic import Field

from tools._common import mcp
from tools._symbols import scan_for_location_matches


@mcp.tool()
def find_symbol(
    name: Annotated[
        str,
        Field(
            description=(
                "Exact Python symbol name to locate — matched case-sensitively "
                "against class, function, and async function definitions. "
                "This is not a substring or regex search; 'middleware' will not "
                "match 'ValidationMiddleware'. Use search_files when you only "
                "know part of a name or want text inside a file. "
                "Example: 'RepairMiddleware'."
            )
        ),
    ],
) -> str:
    """Locate Python class and function definitions by exact symbol name.

    Scans every ``.py`` file under the project workspace recursively.
    Use this when you know the definition name and need its file path and
    line range — for example before extract_symbol, or to disambiguate
    duplicate names across modules.

    Prefer extract_symbol when:
        - You need the function or class source body without reading the
          whole file.

    Prefer search_files when:
        - You only know part of a name or want case-insensitive text search.
        - You need matches inside strings, comments, or non-definition lines.

    Prefer extract_structure when:
        - You need a full module map (all symbols, imports, LOC) in one pass.

    Returns:
        A JSON array of matches. Each object contains:
            - ``file``: path relative to the workspace root
            - ``line``: 1-indexed line of the ``def`` or ``class`` statement
            - ``end_line``: 1-indexed last line of the definition body
            - ``type``: ``FunctionDef``, ``AsyncFunctionDef``, or ``ClassDef``
            - ``name``: the matched symbol name (same as the ``name`` argument)

        An empty array ``[]`` means no definition was found — not an error.
        Nested definitions (methods on a class, inner functions) are included
        when their name equals ``name``.

    Raises:
        Does not raise for missing symbols or unreadable files; those files
        are skipped silently and produce fewer or zero matches.
    """
    matches = scan_for_location_matches(name)
    return json.dumps(matches, indent=2)
