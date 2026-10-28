from typing import Annotated

from pydantic import Field

from tools._common import WORKSPACE_ROOT, ensure_directory_exists, mcp, safe_path


@mcp.tool()
def list_directory(
    path: Annotated[
        str,
        Field(
            description=(
                "Directory to list, given as a path relative to the project "
                "workspace root (never an absolute path). Use '.' to list "
                "the workspace root itself. Lists immediate contents only "
                "— does not recurse into subdirectories. "
                "Example: 'local-coding-agent'."
            )
        ),
    ] = ".",
) -> str:
    """List the immediate files and subdirectories of a directory inside
    the project workspace.

    Returns:
        One relative path per line, sorted alphabetically. Directories and
        files are not distinguished in the output — check for a file
        extension or call this tool again on the entry to tell them apart.

    Raises:
        ValueError: if the path does not exist, is outside the workspace,
            or does not point to a directory.
    """
    directory = safe_path(path)
    ensure_directory_exists(directory, path)

    entries = sorted(directory.iterdir())
    return "\n".join(str(entry.relative_to(WORKSPACE_ROOT)) for entry in entries)
