from typing import Annotated

from pydantic import Field

from tools._common import ensure_file_exists, mcp, safe_path


@mcp.tool()
def read_file(
    path: Annotated[
        str,
        Field(
            description=(
                "File to read, given as a path relative to the project "
                "workspace root (never an absolute path). Must be a UTF-8 "
                "text file — binary files will raise an error. Returns the "
                "entire file with no truncation, so prefer search_files "
                "first on large files to locate the relevant section. "
                "Example: 'local-coding-agent/mcp_server.py'."
            )
        ),
    ],
) -> str:
    """Read the full contents of a UTF-8 text file inside the project
    workspace.

    Returns:
        The raw file contents as a single string, encoding preserved.

    Raises:
        ValueError: if the path does not exist, is outside the workspace,
            or does not point to a regular file.
        UnicodeDecodeError: if the file's bytes are not valid UTF-8.
    """
    file_path = safe_path(path)
    ensure_file_exists(file_path, path)

    return file_path.read_text(encoding="utf-8")
