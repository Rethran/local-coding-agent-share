"""Run the arithmetic example for real: tests fail, then an implementation, then tests again.

    python -m evals example

No model and no GPU. Each try is a fresh copy of evals/examples/laskin.
The first pytest must fail. The second pytest is the result that lands in the report.
summa is implemented correctly three times, erotus fails once, tulo never passes
but is still marked done.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from evals.report import REPORTS_DIR, build_index, build_report, write_report

EXAMPLE_DIR = Path(__file__).resolve().parent / "examples" / "laskin"
TASKS = ("summa", "erotus", "tulo")
INPUTS = {
    "summa": "Laske 2 + 3. Vastaa pelkällä luvulla.",
    "erotus": "Laske 5 - 2. Vastaa pelkällä luvulla.",
    "tulo": "Laske 2 * 4. Vastaa pelkällä luvulla.",
}
# Third erotus is wrong on purpose. tulo is always wrong and still marked done.
BODIES = {
    "summa": ("return a + b", "return a + b", "return a + b"),
    "erotus": ("return a - b", "return a - b", "return a + b"),
    "tulo": ("return 0", "return 0", "return 0"),
}


def _pytest(work: Path, task: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable, "-m", "pytest",
            "-c", "pytest.ini",
            "--rootdir", str(work),
            "tests/test_laskin.py",
            "-k", f"test_{task}",
            "-q", "--tb=short",
        ],
        cwd=work,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _write_impl(work: Path, task: str, body: str) -> None:
    parts = []
    for name in TASKS:
        impl = body if name == task else "raise NotImplementedError"
        parts.append(f"def {name}(a, b):\n    {impl}\n")
    (work / "laskin.py").write_text("\n".join(parts), encoding="utf-8")


def run_trial(task: str, epoch: int, body: str) -> dict[str, Any]:
    """One isolated try. Returns the trial plus whether the first pytest failed."""
    work = Path(tempfile.mkdtemp(prefix=f"esimerkki-{task}-{epoch}-"))
    started = time.perf_counter()
    try:
        shutil.copytree(EXAMPLE_DIR, work, dirs_exist_ok=True)
        red = _pytest(work, task)
        if red.returncode == 0:
            raise RuntimeError(f"{task} kerta {epoch}: testit menivät läpi ennen toteutusta")
        _write_impl(work, task, body)
        green = _pytest(work, task)
    finally:
        wall = time.perf_counter() - started
        shutil.rmtree(work, ignore_errors=True)
    passed = green.returncode == 0
    tail = ((green.stdout or "") + (green.stderr or "")).strip()
    tail = tail[-500:] if tail else ("läpi" if passed else "kaatui")
    return {
        "sampleId": task,
        "epoch": epoch,
        "title": task,
        "input": INPUTS[task],
        "wallSeconds": round(wall, 3),
        "redFailed": True,
        "scores": {
            "subtask_done": {"ok": True, "explanation": "Toteutus merkittiin valmiiksi."},
            "reference_tests_pass": {"ok": passed, "explanation": tail},
            "process_budget": {"ok": True, "explanation": "Pieni esimerkki pysyi rajassa."},
        },
    }


def run_example(dest_stem: Path | None = None) -> tuple[dict[str, Any], Path]:
    trials = []
    for task in TASKS:
        for epoch, body in enumerate(BODIES[task], start=1):
            trial = run_trial(task, epoch, body)
            ok = trial["scores"]["reference_tests_pass"]["ok"]
            loppu = "Testit menivät läpi." if ok else "Testit kaatuivat yhä."
            print(f"{task.capitalize()}, kerta {epoch}. Testit kaatuivat ensin. Toteutus kirjoitettiin. {loppu}")
            trials.append(trial)
    stem = dest_stem or (REPORTS_DIR / "esimerkki-summa")
    report = build_report(
        trials,
        header={
            "sourceLog": str(EXAMPLE_DIR),
            "model": "esimerkki",
            "promptVariant": "summa",
            "runStatus": "success",
            "dryRun": False,
            "plannedEpochs": 3,
            "plannedSamples": 3,
        },
    )
    html_path, _json_path = write_report(report, stem)
    if stem.resolve().parent == REPORTS_DIR.resolve():
        build_index(REPORTS_DIR)
    return report, html_path


def main(argv: list[str] | None = None) -> int:
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Aja summa-esimerkki oikeilla testeillä ja kirjoita raportti.")
    parser.add_argument("--out", default="", help="Raportin tiedostonimi ilman päätettä.")
    args = parser.parse_args(argv)
    stem = Path(args.out) if args.out else None
    _report, html_path = run_example(stem)
    print(f"HTML: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
