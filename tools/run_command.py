import subprocess

from tools._common import WORKSPACE_ROOT, mcp


@mcp.tool()
def run_command(command: str) -> str:
    """Run a command in the project workspace."""
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
    )

    output = result.stdout

    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"

    return f"Exit code: {result.returncode}\n{output}"
