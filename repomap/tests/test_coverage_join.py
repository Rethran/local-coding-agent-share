"""
Coverage join: attach a coverage percentage to each symbol by
intersecting its line range with coverage.py's JSON report
(executed_lines / missing_lines per file).
"""
import json
from pathlib import Path

from repomap.symbols import Symbol
from repomap.coverage_join import attach_coverage


def _write_coverage_json(path: Path, file_key: str, executed: list[int], missing: list[int]):
    data = {
        "files": {
            file_key: {
                "executed_lines": executed,
                "missing_lines": missing,
            }
        }
    }
    path.write_text(json.dumps(data))


def test_fully_covered_symbol_gets_100_percent(tmp_path):
    cov_file = tmp_path / "coverage.json"
    src_file = tmp_path / "mod.py"
    _write_coverage_json(cov_file, str(src_file), executed=[1, 2, 3], missing=[])

    sym = Symbol(name="f", file=src_file, signature="def f()", start_line=1, end_line=3)

    results = attach_coverage([sym], cov_file)

    assert results[sym.symbol_id] == 100.0


def test_partially_covered_symbol_gets_correct_percentage(tmp_path):
    cov_file = tmp_path / "coverage.json"
    src_file = tmp_path / "mod.py"
    _write_coverage_json(cov_file, str(src_file), executed=[1, 2], missing=[3, 4])

    sym = Symbol(name="f", file=src_file, signature="def f()", start_line=1, end_line=4)

    results = attach_coverage([sym], cov_file)

    assert results[sym.symbol_id] == 50.0


def test_symbol_with_no_coverage_data_gets_none(tmp_path):
    cov_file = tmp_path / "coverage.json"
    src_file = tmp_path / "mod.py"
    other_file = tmp_path / "other.py"
    _write_coverage_json(cov_file, str(other_file), executed=[1], missing=[])

    sym = Symbol(name="f", file=src_file, signature="def f()", start_line=1, end_line=2)

    results = attach_coverage([sym], cov_file)

    assert results[sym.symbol_id] is None


def test_missing_coverage_file_returns_none_for_all_symbols(tmp_path):
    src_file = tmp_path / "mod.py"
    sym = Symbol(name="f", file=src_file, signature="def f()", start_line=1, end_line=2)

    results = attach_coverage([sym], tmp_path / "does_not_exist.json")

    assert results[sym.symbol_id] is None


def test_zero_coverable_lines_gets_none_not_division_error(tmp_path):
    cov_file = tmp_path / "coverage.json"
    src_file = tmp_path / "mod.py"
    _write_coverage_json(cov_file, str(src_file), executed=[], missing=[])

    # a one-line stub, e.g. `def f(): ...` with nothing coverage.py tracks in range
    sym = Symbol(name="f", file=src_file, signature="def f()", start_line=10, end_line=10)

    results = attach_coverage([sym], cov_file)

    assert results[sym.symbol_id] is None
