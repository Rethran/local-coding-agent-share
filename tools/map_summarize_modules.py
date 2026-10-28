import json
from pathlib import Path

from tools._common import mcp
from tools._arch import cache, hash_file
from ollama_client import summarize_module


@mcp.tool()
def map_summarize_modules(structure_path: str, changed_only: bool = True) -> str:
    """
   Summarizes each module one at a time using the local Qwen3 model.
   If changed_only=True, uses a hash cache to process only files that have changed since the previous run.
   Returns a list of module summaries.
    """
    structure = json.loads(Path(structure_path).read_text())
    summaries = []

    for module in structure["modules"]:
        content_hash = hash_file(module["path"])

        if changed_only and cache.has(module["path"], content_hash):
            summaries.append(cache.get(module["path"]))
            continue

        summary = summarize_module(module)
        cache.set(module["path"], content_hash, summary)
        summaries.append(summary)

    return json.dumps(summaries, indent=2)
