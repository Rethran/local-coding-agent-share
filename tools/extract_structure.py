import json
from pathlib import Path
from typing import Annotated

from pydantic import Field

from tools._common import ensure_directory_exists, mcp
from tools._arch import write_structure_file
from structure_extractor import build_structure_map


@mcp.tool()
def extract_structure(
    repo_path: Annotated[
        str,
        Field(
            description=(
                "Path to the repository root to analyze. ..."
            )
        ),
    ],
) -> str:
    """Extract the codebase structure (classes, functions, and imports)
    without using an LLM.

    ALWAYS use this tool first before any map/reduce phase.

    Returns:
        A JSON object with 'structure_path' (path to the generated
        .structure.json file) and 'module_count' (number of modules found).

    Raises:
        ValueError: if the path does not exist or does not point to a directory.
    """
    root = Path(repo_path).resolve()
    ensure_directory_exists(root, repo_path)

    structure = build_structure_map(root)
    out_path = write_structure_file(root, structure)

    return json.dumps({
        "structure_path": str(out_path),
        "module_count": len(structure["modules"]),
    })
