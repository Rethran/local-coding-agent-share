import json
from pathlib import Path

import requests

from tools._common import mcp
from tools._arch import cache, hash_file
from ollama_client import reduce_architecture, summarize_module
from structure_extractor import build_structure_map


@mcp.tool()
def analyze_architecture(
    repo_path: str,
    focus_question: str = "Tarkista kerrosarkkitehtuurin rikkeet ja syklit",
    changed_only: bool = True,
) -> str:
    """
    Runs the complete architecture analysis in a single call:
    extract_structure -> map_summarize_modules -> reduce_architecture_review.

    Use this tool when you want a single architecture assessment from a large codebase without manually orchestrating the three tool calls. Suitable for large repositories because raw source code is never sent to the model all at once.

    Returns the architecture assessment plus map-phase statistics, including the number of cached modules and newly processed modules.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        return json.dumps({"error": f"Polkua ei löydy: {repo_path}"})

    structure = build_structure_map(root)
    structure_path = root / ".structure.json"
    structure_path.write_text(json.dumps(structure, indent=2))

    if not structure["modules"]:
        return json.dumps({"error": "Ei .py-tiedostoja löytynyt annetusta polusta"})

    summaries = []
    cache_hits, fresh_runs, errors = 0, 0, []

    for module in structure["modules"]:
        content_hash = hash_file(module["path"])

        if changed_only and cache.has(module["path"], content_hash):
            summaries.append(cache.get(module["path"]))
            cache_hits += 1
            continue

        try:
            summary = summarize_module(module)
            cache.set(module["path"], content_hash, summary)
            summaries.append(summary)
            fresh_runs += 1
        except (json.JSONDecodeError, KeyError, requests.RequestException) as e:
            errors.append({"module": module["path"], "error": str(e)})

    if not summaries:
        return json.dumps({
            "error": "Map-vaihe ei tuottanut yhtään tiivistystä",
            "details": errors,
        })

    review = reduce_architecture(summaries, focus_question)

    return json.dumps({
        "architecture_review": review,
        "stats": {
            "total_modules": len(structure["modules"]),
            "cache_hits": cache_hits,
            "fresh_analysis": fresh_runs,
            "map_errors": len(errors),
        },
        "map_errors_detail": errors if errors else None,
    }, indent=2, ensure_ascii=False)
