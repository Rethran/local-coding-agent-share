from tools._common import WORKSPACE_ROOT, mcp, safe_path


@mcp.tool()
def create_folder(path: str) -> str:
    """Create a directory inside the project workspace."""
    folder = safe_path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return f"Created folder: {folder.relative_to(WORKSPACE_ROOT)}"
