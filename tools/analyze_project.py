"""Project structure analysis tool.

This module exposes the ``analyze_project`` MCP tool, which walks the
workspace and reports its layout. It is intended as a first, cheap
orientation step for agents: before reading individual files, an agent
can call this tool to learn which directories and files exist and which
of them are likely to be the most important (build configs, manifests,
documentation, container definitions).

The tool returns a JSON document with four keys:

* ``root``            -- absolute path of the workspace root.
* ``directories``     -- relative paths of directories (capped at 500).
* ``files``           -- relative paths of files (capped at 1000).
* ``important_files`` -- the subset of files whose names match a known
                         set of "important" filenames.

Paths are reported relative to the workspace root so that the output is
stable regardless of where the workspace lives on disk.
"""

import json
from pathlib import Path

from tools._common import WORKSPACE_ROOT, is_within_ignored_directory, mcp

# Filenames that typically describe a project's build, packaging,
# documentation, or deployment setup. These are surfaced separately so
# that agents can prioritise them when orienting themselves.
IMPORTANT_FILENAMES = frozenset({
    "pyproject.toml",
    "package.json",
    "requirements.txt",
    "README.md",
    "Dockerfile",
    "docker-compose.yml",
})

# Upper bounds on the number of entries returned, so that very large
# workspaces do not produce an unwieldy response.
MAX_DIRECTORIES = 500
MAX_FILES = 1000


def _is_important_file(path: Path) -> bool:
    """Return True if the file's name is one of the important filenames."""
    return path.name in IMPORTANT_FILENAMES


def _collect_workspace_entries() -> tuple[list[str], list[str]]:
    """Walk the workspace and split entries into directories and files.

    Entries that live inside an ignored directory (see
    ``tools._common.IGNORED_DIRS``) are skipped. Both lists contain
    paths relative to the workspace root.
    """
    directories: list[str] = []
    files: list[str] = []

    for path in WORKSPACE_ROOT.rglob("*"):
        if is_within_ignored_directory(path):
            continue

        relative = str(path.relative_to(WORKSPACE_ROOT))
        if path.is_dir():
            directories.append(relative)
        else:
            files.append(relative)

    return directories, files


def _build_result(directories: list[str], files: list[str]) -> dict:
    """Assemble the JSON-serialisable result document."""
    important_files = [
        f for f in files if _is_important_file(Path(f))
    ]

    return {
        "root": str(WORKSPACE_ROOT),
        "directories": directories[:MAX_DIRECTORIES],
        "files": files[:MAX_FILES],
        "important_files": important_files,
    }


@mcp.tool()
def analyze_project() -> str:
    """Analyze the project structure and identify important files.

    Walks the workspace root, skipping ignored directories (VCS, virtual
    environments, caches, dependency folders), and returns a JSON document
    describing the layout: the root path, the list of directories, the
    list of files, and the subset of files considered important (build
    configs, manifests, documentation, container definitions).

    Returns:
        A JSON string with keys ``root``, ``directories``, ``files``,
        and ``important_files``.
    """
    directories, files = _collect_workspace_entries()
    result = _build_result(directories, files)
    return json.dumps(result, indent=2)
