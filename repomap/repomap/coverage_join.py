"""
Joins a coverage.py JSON report (`coverage json`) onto the symbol table
by line-range intersection. No new tooling needed beyond coverage.py,
which the ai-test-orchestrator pipeline already produces.
"""
from __future__ import annotations

import json
from pathlib import Path

from repomap.symbols import Symbol


def attach_coverage(symbols: list[Symbol], coverage_json_path: Path) -> dict[str, float | None]:
    """
    Returns {symbol_id: coverage_percent} where coverage_percent is
    (executed lines in range / total tracked lines in range) * 100,
    or None if the file isn't in the report or no lines in range are tracked.
    """
    coverage_json_path = Path(coverage_json_path)
    results: dict[str, float | None] = {}

    try:
        report = json.loads(coverage_json_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {sym.symbol_id: None for sym in symbols}

    files = report.get("files", {})

    for sym in symbols:
        file_data = files.get(str(sym.file))
        if file_data is None:
            results[sym.symbol_id] = None
            continue

        executed = set(file_data.get("executed_lines", []))
        missing = set(file_data.get("missing_lines", []))
        line_range = range(sym.start_line, sym.end_line + 1)

        executed_in_range = sum(1 for ln in line_range if ln in executed)
        missing_in_range = sum(1 for ln in line_range if ln in missing)
        total_tracked = executed_in_range + missing_in_range

        if total_tracked == 0:
            results[sym.symbol_id] = None
        else:
            results[sym.symbol_id] = round(100.0 * executed_in_range / total_tracked, 4)

    return results
