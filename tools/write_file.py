from tools._common import WORKSPACE_ROOT, mcp, safe_path

from orchestrator.python_write_gate import refuse_invalid_python_write


@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write a UTF-8 text file inside the project workspace."""
    refusal = refuse_invalid_python_write(path, content)
    if refusal:
        raise ValueError(refusal)
    file = safe_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding="utf-8")
    return f"Wrote file: {file.relative_to(WORKSPACE_ROOT)}"
