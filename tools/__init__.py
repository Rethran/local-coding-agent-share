import importlib
import os
from pathlib import Path


def load_all_tools(tools_dir: Path | None = None) -> list[str]:
    """Import every tool module in the tools directory.

    Skips private modules (``_*.py``) and ``__init__.py``.
    Returns the list of loaded module names.
    """
    directory = tools_dir or Path(
        os.environ.get(
            "MCP_TOOLS_DIR",
            Path(__file__).resolve().parent,
        )
    )

    loaded = []
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_") or path.name == "__init__.py":
            continue

        module_name = f"tools.{path.stem}"
        importlib.import_module(module_name)
        loaded.append(module_name)

    return loaded
