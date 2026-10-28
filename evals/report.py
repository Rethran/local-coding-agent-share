"""Turn eval logs into a human-readable HTML + machine-readable JSON report."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from evals.criteria import CRITERIA, PROCESS_BUDGET

AGENT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
CORRECT_VALUES = {"C", "c", True, 1, 1.0, "correct", "CORRECT"}

env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html"]),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def bernoulli_stderr(pass_rate: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return math.sqrt(pass_rate * (1 - pass_rate) / n)


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _is_correct(value: Any) -> bool:
    return value in CORRECT_VALUES


def trials_from_canonical(data: dict[str, Any]) -> list[dict[str, Any]]:
    return list(data.get("trials") or [])


def trials_from_inspect_log(log: Any) -> list[dict[str, Any]]:
    trials: list[dict[str, Any]] = []
    for sample in log.samples or []:
        if not (sample.scores or {}) and getattr(sample, "error", None):
            # Interrupted by operator or crashed before scoring: not a trial.
            continue
        scores: dict[str, dict[str, Any]] = {}
        for name, score in (sample.scores or {}).items():
            short = name.rsplit("/", 1)[-1]
            value = getattr(score, "value", None)
            scores[short] = {
                "ok": _is_correct(value),
                "explanation": getattr(score, "explanation", None) or "",
                "value": value,
            }
        meta = sample.metadata or {}
        result = meta.get("result") or {}
        totals = result.get("totals") if isinstance(result.get("totals"), dict) else {}
        if not totals:
            scores_meta = next(iter((sample.scores or {}).values()), None)
            extra = getattr(scores_meta, "metadata", None) or {}
            totals = extra.get("totals") if isinstance(extra.get("totals"), dict) else {}
        for name, score in (sample.scores or {}).items():
            short = name.rsplit("/", 1)[-1]
            extra = getattr(score, "metadata", None) or {}
            if isinstance(extra, dict) and extra.get("checks") and short in scores:
                scores[short]["checks"] = extra["checks"]
        steps = totals.get("steps")
        trials.append({
            "sampleId": str(sample.id),
            "epoch": int(sample.epoch),
            "input": sample.input if isinstance(sample.input, str) else str(sample.input),
            "title": (meta.get("title") or str(sample.id)),
            "scores": scores,
            "wallSeconds": float(result.get("wallSeconds") or sample.total_time or 0),
            "tokens": int(totals.get("promptTokens") or 0) + int(totals.get("completionTokens") or 0),
            "steps": steps,
            "totals": totals,
            "stopReasons": list(result.get("stopReasons") or []),
            "timedOut": bool(result.get("timedOut")),
            "stopReason": (
                "timeout" if result.get("timedOut")
                else (result.get("stdoutTail") or "")[-120:]
            ),
            "exitCode": result.get("exitCode"),
        })
    return trials


def header_from_inspect_log(log: Any, source: str) -> dict[str, Any]:
    """Model, prompt variant, run status and planned size, read from a log object."""
    meta = log.eval.metadata or {}
    header: dict[str, Any] = {
        "sourceLog": source,
        "model": meta.get("model_tag") or getattr(log.eval, "model", None),
        "promptVariant": meta.get("prompt_variant"),
        "runStatus": getattr(log, "status", None),
        "dryRun": False,
        "plannedEpochs": None,
        "plannedSamples": None,
    }
    config = getattr(log.eval, "config", None)
    header["plannedEpochs"] = getattr(config, "epochs", None) if config else None
    dataset = getattr(log.eval, "dataset", None)
    header["plannedSamples"] = getattr(dataset, "samples", None) if dataset else None
    if log.samples:
        sample_meta = log.samples[0].metadata or {}
        header["model"] = header["model"] or sample_meta.get("modelTag")
        header["promptVariant"] = header["promptVariant"] or sample_meta.get("promptVariant")
        header["dryRun"] = bool(sample_meta.get("dryRun"))
    return header


def load_trials(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".eval", ".evalz"} or path.name.endswith(".eval"):
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path))
        return trials_from_inspect_log(log), header_from_inspect_log(log, str(path))
    header: dict[str, Any] = {
        "sourceLog": str(path),
        "model": None,
        "promptVariant": None,
    }
    data = json.loads(path.read_text(encoding="utf-8"))
    header["model"] = data.get("model")
    header["promptVariant"] = data.get("promptVariant")
    header["runStatus"] = data.get("runStatus")
    header["dryRun"] = bool(data.get("dryRun"))
    if data.get("trials"):
        return trials_from_canonical(data), header
    raise ValueError(f"Unsupported report source: {path}")


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    scorer_names = sorted({name for trial in trials for name in trial.get("scores") or {}})
    scorers: dict[str, dict[str, Any]] = {}
    for name in scorer_names:
        values = [
            1.0 if (trial.get("scores") or {}).get(name, {}).get("ok") else 0.0
            for trial in trials
            if name in (trial.get("scores") or {})
        ]
        n = len(values)
        rate = (sum(values) / n) if n else 0.0
        scorers[name] = {
            "passRate": round(rate, 4),
            "stderr": round(bernoulli_stderr(rate, n), 4),
            "n": n,
            "passed": int(sum(values)),
        }
    walls = [float(trial.get("wallSeconds") or 0) for trial in trials]
    tokens = [int(trial.get("tokens") or 0) for trial in trials]
    return {
        "scorers": scorers,
        "p95WallSeconds": round(p95(walls) or 0.0, 3),
        "meanWallSeconds": round(statistics.mean(walls), 3) if walls else 0.0,
        "totalTokens": sum(tokens),
        "nTrials": len(trials),
        "nSamples": len({trial["sampleId"] for trial in trials}),
    }


def failures_first(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trial in trials:
        for name, score in (trial.get("scores") or {}).items():
            if score.get("ok"):
                continue
            rows.append({
                "sampleId": trial["sampleId"],
                "epoch": trial.get("epoch"),
                "title": trial.get("title"),
                "input": trial.get("input"),
                "criterion": name,
                "explanation": score.get("explanation") or "",
                "stopReason": trial.get("stopReason") or "",
            })
    return rows


def flaky_cases(trials: list[dict[str, Any]], scorer: str = "subtask_done") -> list[dict[str, Any]]:
    grouped: dict[str, list[bool]] = {}
    for trial in trials:
        score = (trial.get("scores") or {}).get(scorer)
        if score is None:
            continue
        grouped.setdefault(str(trial["sampleId"]), []).append(bool(score.get("ok")))
    rows: list[dict[str, Any]] = []
    for sample_id, outcomes in sorted(grouped.items()):
        if len(outcomes) < 2:
            continue
        passed = sum(1 for ok in outcomes if ok)
        if 0 < passed < len(outcomes):
            rows.append({
                "sampleId": sample_id,
                "scorer": scorer,
                "passed": passed,
                "epochs": len(outcomes),
            })
    return rows


FAILURE_ORDER = ("timeout", "loop", "tool_errors", "step_budget", "reference_tests", "other")
FAILURE_LABELS = {
    "timeout": "aikakatkaisu",
    "loop": "silmukka",
    "tool_errors": "työkaluvirheet",
    "step_budget": "askelbudjetti",
    "reference_tests": "testit",
    "other": "muu",
}

SCORE_LABELS_FI = {
    "subtask_done": "Valmis-merkintä",
    "reference_tests_pass": "Testit",
    "process_budget": "Prosessiraja",
}

CRITERIA_FI = {
    "subtask_done": {
        "otsikko": "Valmis-merkintä",
        "mita": "Agentti merkitsi tehtävän valmiiksi ja tarkistus hyväksyi sen.",
        "miksi": "Sujuvakaan keskustelu ei riitä, jos tehtävää ei merkitä valmiiksi.",
        "miten": "Luetaan tehtävän tila ajon jälkeen. Merkintä menee läpi vain, jos tila on valmis.",
    },
    "reference_tests_pass": {
        "otsikko": "Testit",
        "mita": "Aiemmin hyväksytyn version testit menevät läpi agentin kirjoittamalla koodilla.",
        "miksi": "Agentin omat testit voivat olla liian heikot. Nämä ovat aiemmin hyväksyttyjä testejä, eivät erillisiä salaisia testejä.",
        "miten": "Ajon jälkeen ajetaan hyväksytyn version testitiedostot.",
    },
    "process_budget": {
        "otsikko": "Prosessiraja",
        "mita": "Agentti valmistui ilman silmukkaa, askelbudjetin loppumista tai työkaluvirheiden tulvaa.",
        "miksi": "Paikallinen malli epäonnistuu usein siinä, miten työ tehdään, ei vain siinä, onko vastaus oikein.",
        "miten": (
            "Luetaan ajon yhteenveto. Tulos hylätään, jos yrityksiä on yli "
            f"{PROCESS_BUDGET['max_attempts']}, askeleita yli {PROCESS_BUDGET['max_steps']}, "
            f"sama työkalukutsu toistuu yli {PROCESS_BUDGET['max_repeat_of_one_call']} kertaa, "
            f"tai työkaluvirheiden osuus on yli {PROCESS_BUDGET['max_tool_error_rate']:.0%}."
        ),
    },
}


def _ok(trial: dict[str, Any], name: str) -> bool | None:
    score = (trial.get("scores") or {}).get(name)
    if not isinstance(score, dict) or "ok" not in score:
        return None
    return bool(score.get("ok"))


def false_done_rate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Share of DONE claims whose reference tests still fail."""
    claimed = [trial for trial in trials if _ok(trial, "subtask_done") is True]
    false = [trial for trial in claimed if _ok(trial, "reference_tests_pass") is False]
    n = len(claimed)
    rate = (len(false) / n) if n else None
    return {
        "rate": round(rate, 4) if rate is not None else None,
        "stderr": round(bernoulli_stderr(rate or 0.0, n), 4) if rate is not None else None,
        "falseDone": len(false),
        "claimedDone": n,
    }


def consistency_buckets(
    trials: list[dict[str, Any]],
    scorer: str = "reference_tests_pass",
) -> dict[str, Any]:
    """Group cases by how many epochs the reference tests passed."""
    grouped: dict[str, dict[str, Any]] = {}
    for trial in trials:
        ok = _ok(trial, scorer)
        if ok is None:
            continue
        row = grouped.setdefault(
            str(trial["sampleId"]),
            {"title": trial.get("title") or trial["sampleId"], "passed": 0, "epochs": 0},
        )
        row["epochs"] += 1
        if ok:
            row["passed"] += 1
    reliable: list[dict[str, Any]] = []
    fragile: list[dict[str, Any]] = []
    cannot: list[dict[str, Any]] = []
    for sample_id, row in sorted(grouped.items()):
        item = {
            "sampleId": sample_id,
            "title": row["title"],
            "passed": row["passed"],
            "epochs": row["epochs"],
        }
        if row["epochs"] and row["passed"] == row["epochs"]:
            reliable.append(item)
        elif row["passed"] == 0:
            cannot.append(item)
        else:
            fragile.append(item)
    return {
        "scorer": scorer,
        "reliable": reliable,
        "fragile": fragile,
        "cannot": cannot,
    }


def cost_per_success(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """GPU-minutes divided by cases that passed reference tests on every epoch."""
    seconds = sum(float(trial.get("wallSeconds") or 0) for trial in trials)
    full = len(consistency_buckets(trials)["reliable"])
    minutes = seconds / 60.0
    return {
        "gpuMinutes": round(minutes, 2),
        "fullSuccesses": full,
        "minutesPerSuccess": round(minutes / full, 2) if full else None,
    }


def _spread(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"median": None, "worst": None, "n": 0}
    return {
        "median": round(float(statistics.median(values)), 3),
        "worst": round(max(values), 3),
        "n": len(values),
    }


def time_and_steps(trials: list[dict[str, Any]]) -> dict[str, Any]:
    walls = [float(trial.get("wallSeconds") or 0) for trial in trials]
    steps: list[float] = []
    for trial in trials:
        totals = trial.get("totals") if isinstance(trial.get("totals"), dict) else {}
        raw = totals.get("steps", trial.get("steps"))
        if raw is None:
            continue
        steps.append(float(raw))
    return {"wallSeconds": _spread(walls), "steps": _spread(steps)}


def _checks(trial: dict[str, Any]) -> dict[str, Any]:
    budget = (trial.get("scores") or {}).get("process_budget") or {}
    checks = budget.get("checks")
    return checks if isinstance(checks, dict) else {}


def _blob(trial: dict[str, Any]) -> str:
    parts = [str(trial.get("stopReason") or "")]
    parts.extend(str(item) for item in trial.get("stopReasons") or [])
    for score in (trial.get("scores") or {}).values():
        if isinstance(score, dict):
            parts.append(str(score.get("explanation") or ""))
    return " ".join(parts).lower()


def primary_failure(trial: dict[str, Any]) -> str | None:
    """One label per failed trial, in a fixed priority so the counts add up."""
    tests_ok = _ok(trial, "reference_tests_pass")
    done_ok = _ok(trial, "subtask_done")
    if tests_ok is not False and done_ok is not False:
        return None
    blob = _blob(trial)
    if trial.get("timedOut") or "timeout" in blob or "timed out" in blob:
        return "timeout"
    checks = _checks(trial)
    repeat = checks.get("maxRepeatOfOneCall") or {}
    if isinstance(repeat, dict) and repeat.get("ok") is False:
        return "loop"
    tools = checks.get("toolErrorRate") or {}
    if isinstance(tools, dict) and tools.get("ok") is False:
        return "tool_errors"
    steps = checks.get("steps") or {}
    attempts = checks.get("attempts") or {}
    if (isinstance(steps, dict) and steps.get("ok") is False) or (
        isinstance(attempts, dict) and attempts.get("ok") is False
    ):
        return "step_budget"
    if "tool step budget" in blob or "step budget" in blob or "tool_step_budget" in blob:
        return "step_budget"
    if tests_ok is False:
        return "reference_tests"
    return "other"


def failure_types(trials: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in FAILURE_ORDER}
    for trial in trials:
        label = primary_failure(trial)
        if label:
            counts[label] += 1
    return {
        "counts": counts,
        "labels": FAILURE_LABELS,
        "order": list(FAILURE_ORDER),
        "failedTrials": sum(counts.values()),
    }


def rate_comparison(
    current: float | None,
    current_n: int,
    previous: float | None,
    previous_n: int,
) -> dict[str, Any]:
    if current is None or previous is None:
        return {"delta": None, "combinedStderr": None, "noise": None}
    delta = current - previous
    combined = math.sqrt(
        bernoulli_stderr(current, max(current_n, 0)) ** 2
        + bernoulli_stderr(previous, max(previous_n, 0)) ** 2
    )
    signal = abs(delta) > combined if combined > 0 else abs(delta) > 0
    return {
        "delta": round(delta, 4),
        "combinedStderr": round(combined, 4),
        "noise": not signal,
    }


def _reference_tests_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        1.0 if _ok(trial, "reference_tests_pass") else 0.0
        for trial in trials
        if _ok(trial, "reference_tests_pass") is not None
    ]
    n = len(values)
    rate = (sum(values) / n) if n else None
    return {
        "passRate": round(rate, 4) if rate is not None else None,
        "stderr": round(bernoulli_stderr(rate or 0.0, n), 4) if rate is not None else None,
        "n": n,
    }


def build_review(
    trials: list[dict[str, Any]],
    previous: dict[str, Any] | None = None,
) -> dict[str, Any]:
    false_done = false_done_rate(trials)
    buckets = consistency_buckets(trials)
    reference = _reference_tests_summary(trials)
    n_samples = len({str(trial["sampleId"]) for trial in trials})
    reliable_share = (len(buckets["reliable"]) / n_samples) if n_samples else None
    prev = previous or {}
    prev_false = prev.get("falseDone") or {}
    prev_ref = prev.get("referenceTests") or {}
    prev_n = int(prev.get("nSamples") or 0)
    return {
        "falseDone": false_done,
        "consistency": buckets,
        "cost": cost_per_success(trials),
        "timing": time_and_steps(trials),
        "failureTypes": failure_types(trials),
        "referenceTests": reference,
        "reliableShare": round(reliable_share, 4) if reliable_share is not None else None,
        "nSamples": n_samples,
        "deltas": {
            "referenceTestsPassRate": rate_comparison(
                reference.get("passRate"),
                int(reference.get("n") or 0),
                prev_ref.get("passRate"),
                int(prev_ref.get("n") or 0),
            ),
            "falseDoneRate": rate_comparison(
                false_done.get("rate"),
                int(false_done.get("claimedDone") or 0),
                prev_false.get("rate"),
                int(prev_false.get("claimedDone") or 0),
            ),
            "reliableShare": rate_comparison(
                reliable_share,
                n_samples,
                prev.get("reliableShare"),
                prev_n,
            ),
        },
    }


def review_of(report: dict[str, Any]) -> dict[str, Any]:
    if isinstance(report.get("review"), dict) and report["review"].get("consistency"):
        return report["review"]
    trials = report.get("trials") or []
    if trials:
        return build_review(trials)
    return {}


def load_previous_reports(reports_dir: Path, exclude: Path | None = None) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    if not reports_dir.exists():
        return reports
    for path in sorted(reports_dir.glob("*.json")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("schema") != "eval-report/v1":
            continue
        data["_path"] = str(path)
        reports.append(data)
    reports.sort(key=lambda row: row.get("createdAt") or "")
    return reports


def _fi_count(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 0.05:
        return str(int(round(number)))
    return f"{number:.1f}".replace(".", ",")


def _fi_minutes(value: float) -> str:
    if abs(value - round(value)) < 0.05:
        n = int(round(value))
        return "1 minuutti" if n == 1 else f"{n} minuuttia"
    return f"{value:.1f}".replace(".", ",") + " minuuttia"


def _vei(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 45:
        if seconds < 0.5:
            return "alle sekunnin"
        n = int(round(seconds))
        return "1 sekunnin" if n == 1 else f"{n} sekuntia"
    minutes = seconds / 60.0
    if abs(minutes - round(minutes)) < 0.05:
        n = int(round(minutes))
        return "1 minuutin" if n == 1 else f"{n} minuuttia"
    return f"{minutes:.1f}".replace(".", ",") + " minuuttia"


def _task_names(rows: list[dict[str, Any]], *, ratio: bool = False) -> str:
    parts = []
    for row in rows:
        name = row.get("title") or row.get("sampleId")
        if ratio:
            parts.append(f"{name} ({row['passed']}/{row['epochs']})")
        else:
            parts.append(str(name))
    return ", ".join(parts)


def _kertaa(n: int, total: int) -> str:
    times = "yhden kerran" if int(n) == 1 else f"{int(n)} kertaa"
    return f"{times} {int(total)} kerrasta"


def _alkuun(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def fi_kesto(seconds: float | None) -> str:
    """Nominative duration for a table cell."""
    if seconds is None:
        return "—"
    seconds = float(seconds)
    if seconds < 45:
        if seconds < 0.5:
            return "alle sekunti"
        n = int(round(seconds))
        return "1 sekunti" if n == 1 else f"{n} sekuntia"
    return _fi_minutes(seconds / 60.0)


def fi_aika(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        when = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    local = when.astimezone(ZoneInfo("Europe/Helsinki"))
    return f"{local.day}.{local.month}.{local.year} klo {local.hour:02d}.{local.minute:02d}"


env.filters["fi_kesto"] = fi_kesto
env.filters["fi_aika"] = fi_aika


def _n_sana(n: int, yksi: str, monta: str) -> str:
    return f"1 {yksi}" if n == 1 else f"{n} {monta}"


def classification_line(review: dict[str, Any]) -> str:
    buckets = review.get("consistency") or {}
    cost = review.get("cost") or {}
    reliable = len(buckets.get("reliable") or [])
    fragile = len(buckets.get("fragile") or [])
    cannot = len(buckets.get("cannot") or [])
    minutes = cost.get("minutesPerSuccess")
    if minutes is None:
        cost_bit = "ei täyttä onnistumista"
    else:
        shown = int(round(minutes)) if abs(minutes - round(minutes)) < 0.05 else f"{minutes:.1f}".replace(".", ",")
        cost_bit = f"{shown} min/onnistuminen"
    return (
        f"{_n_sana(reliable, 'luotettava', 'luotettavaa')}, "
        f"{_n_sana(fragile, 'hauras', 'haurasta')}, "
        f"{_n_sana(cannot, 'ei osaa', 'ei osaa')} · {cost_bit}"
    )


def _same_log(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return Path(str(left)).name == Path(str(right)).name


def _sample_ids(report: dict[str, Any]) -> set[str]:
    found = {str(trial.get("sampleId")) for trial in report.get("trials") or [] if trial.get("sampleId")}
    buckets = (report.get("review") or {}).get("consistency") or {}
    for key in ("reliable", "fragile", "cannot"):
        for row in buckets.get(key) or []:
            if row.get("sampleId"):
                found.add(str(row["sampleId"]))
    return found


def brief_baseline(
    source_log: str | None,
    reports: list[dict[str, Any]] | None,
    sample_ids: set[str] | None = None,
) -> dict[str, Any] | None:
    """Previous real run of the same tasks.

    Dry-runs, a rewrite of this same log, and a report about different tasks
    (the arithmetic example next to a graph run) do not count.
    """
    for item in reversed(reports or []):
        if item.get("dryRun") or is_dry_run_report(item):
            continue
        if _same_log(item.get("sourceLog"), source_log):
            continue
        if not (item.get("trials") or item.get("review")):
            continue
        other = _sample_ids(item)
        if sample_ids and other and sample_ids.isdisjoint(other):
            continue
        return item
    return None


def business_brief(review: dict[str, Any], baseline_review: dict[str, Any] | None = None) -> dict[str, str]:
    """Finnish answers for the reviewer and the maker. No metric ids."""
    buckets = review.get("consistency") or {}
    reliable = list(buckets.get("reliable") or [])
    fragile = list(buckets.get("fragile") or [])
    cannot = list(buckets.get("cannot") or [])
    every = reliable + fragile + cannot
    full_three = bool(every) and all(int(row.get("epochs") or 0) == 3 for row in every)
    if not every:
        assign = "Ei yhtään arvioitua tehtävää."
    else:
        parts: list[str] = []
        if reliable and full_three:
            if len(reliable) == 1:
                name = reliable[0].get("title") or reliable[0].get("sampleId")
                parts.append(f"{name} onnistui kaikilla kolmella kerralla")
            else:
                parts.append(f"Nämä onnistuivat kaikilla kolmella kerralla: {_task_names(reliable)}")
        elif reliable:
            for row in reliable:
                name = row.get("title") or row.get("sampleId")
                parts.append(
                    f"{name} onnistui joka kerralla ({_kertaa(int(row['passed']), int(row['epochs']))})"
                )
        else:
            parts.append("Yksikään tehtävä ei onnistunut joka kerralla")
        if fragile:
            for row in fragile:
                name = row.get("title") or row.get("sampleId")
                parts.append(
                    f"{name} onnistui vain joskus, {_kertaa(int(row['passed']), int(row['epochs']))}"
                )
        else:
            parts.append("Yksikään tehtävä ei jäänyt vain osittain onnistuneeksi")
        if cannot:
            for row in cannot:
                name = row.get("title") or row.get("sampleId")
                parts.append(f"{name} ei onnistunut kertaakaan")
        else:
            parts.append("Yksikään tehtävä ei jäänyt kokonaan tekemättä")
        assign = ". ".join(_alkuun(part) for part in parts) + "."
        if not full_three:
            assign += (
                " Jokaista tehtävää ei kokeiltu kolmea kertaa,"
                " joten onnistuminen joka kerralla ei vielä ole kolme kertaa kolmesta."
            )
    cost = review.get("cost") or {}
    minutes = cost.get("minutesPerSuccess")
    total = float(cost.get("gpuMinutes") or 0)
    full = int(cost.get("fullSuccesses") or 0)
    if minutes is None:
        cost_sentence = (
            "Yksikään tehtävä ei onnistunut joka kerralla, joten aikaa ei jaeta onnistumista kohti. "
            f"Koko ajo vei {_fi_minutes(total)}."
        )
    else:
        lead = "Yksi tehtävä onnistui joka kerralla." if full == 1 else f"{full} tehtävää onnistui joka kerralla."
        cost_sentence = (
            f"{lead} Yhtä sellaista tehtävää kohti kului {_fi_minutes(float(minutes))}, "
            f"kun epäonnistuneetkin ajot lasketaan mukaan. Koko ajo vei {_fi_minutes(total)}."
        )
    if not baseline_review:
        trend = "Tätä ajoa ei voi verrata edelliseen, joten sivusta ei voi päätellä, paraniko tulos."
    else:
        compared = rate_comparison(
            review.get("reliableShare"),
            int(review.get("nSamples") or 0),
            baseline_review.get("reliableShare"),
            int(baseline_review.get("nSamples") or 0),
        )
        if compared.get("delta") is None or compared.get("noise"):
            trend = "Ero edelliseen ajoon on niin pieni, että se voi johtua sattumasta. Älä pidä sitä muutoksena."
        else:
            trend = "Ero edelliseen ajoon on suurempi kuin sattuma. Tulos on oikeasti muuttunut."
    false_done = review.get("falseDone") or {}
    claimed = int(false_done.get("claimedDone") or 0)
    wrong = int(false_done.get("falseDone") or 0)
    if false_done.get("rate") is None:
        trust = "Yksikään ajo ei merkinnyt tehtävää valmiiksi."
    elif wrong == 0:
        trust = f"Valmis-merkintä piti paikkansa joka kerralla. Merkintöjä oli {claimed}."
    else:
        trust = (
            f"Agentti merkitsi tehtävän valmiiksi {_kertaa(wrong, claimed)}, vaikka testit eivät menneet läpi. "
            "Niissä merkintää ei voi uskoa ilman ihmisen tarkistusta."
        )
    timing = review.get("timing") or {}
    wall = timing.get("wallSeconds") or {}
    steps = timing.get("steps") or {}
    if wall.get("median") is None:
        time_sentence = "Aikaa ei ole kirjattu."
    else:
        time_sentence = f"Tyypillinen ajo vei {_vei(wall['median'])}, hitain {_vei(wall['worst'])}."
    if not steps.get("n"):
        time_sentence += " Askeleita ei ole kirjattu."
    else:
        time_sentence += (
            f" Tyypillisessä ajossa oli {_fi_count(steps['median'])} askelta,"
            f" enimmillään {_fi_count(steps['worst'])}."
        )
    counts = (review.get("failureTypes") or {}).get("counts") or {}
    process_fail = sum(int(counts.get(name) or 0) for name in ("timeout", "loop", "tool_errors", "step_budget"))
    test_fail = int(counts.get("reference_tests") or 0)
    other_fail = int(counts.get("other") or 0)
    if process_fail == 0 and test_fail == 0 and other_fail == 0:
        fix = "Yksikään ajo ei epäonnistunut."
    elif process_fail and not test_fail:
        fix = (
            "Ajo katkesi aikakatkaisuun, silmukkaan, työkaluvirheeseen tai askelten loppumiseen. "
            "Korjaus kuuluu työkaluille tai silmukkavahdille."
        )
    elif test_fail and not process_fail:
        fix = "Testit eivät menneet läpi. Korjaus kuuluu mallille tai sille ohjeelle, jonka malli sai."
    else:
        fix = (
            "Mukana on sekä vääriä vastauksia että katkenneita ajoja. "
            "Korjaus kuuluu mallille, ohjeelle ja työkaluille."
        )
    return {
        "assign": assign,
        "cost": cost_sentence,
        "trend": trend,
        "trust": trust,
        "time": time_sentence,
        "fix": fix,
        "headline": classification_line(review),
    }


def attach_delta(summary: dict[str, Any], previous: dict[str, Any] | None) -> None:
    prev_scorers = ((previous or {}).get("summary") or {}).get("scorers") or {}
    for name, row in summary["scorers"].items():
        prior = prev_scorers.get(name)
        row["delta"] = (
            round(row["passRate"] - prior["passRate"], 4) if prior else None
        )


def build_report(
    trials: list[dict[str, Any]],
    *,
    header: dict[str, Any],
    previous: dict[str, Any] | None = None,
    trend_reports: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = summarize(trials)
    attach_delta(summary, previous)
    previous_review = review_of(previous) if previous else None
    review = build_review(trials, previous_review)
    baseline = brief_baseline(
        header.get("sourceLog"),
        trend_reports,
        {str(trial.get("sampleId")) for trial in trials if trial.get("sampleId")},
    )
    baseline_review = review_of(baseline) if baseline else None
    if not baseline_review:
        baseline_review = None
    sentence_review = build_review(trials, baseline_review) if baseline_review else review
    created_at = _utc_now()
    trend = []
    for item in (trend_reports or [])[-8:]:
        prior = review_of(item)
        trend.append({
            "createdAt": item.get("createdAt"),
            "promptVariant": item.get("promptVariant"),
            "model": item.get("model"),
            "scorers": {
                name: row.get("passRate")
                for name, row in ((item.get("summary") or {}).get("scorers") or {}).items()
            },
            "referenceTestsPassRate": (prior.get("referenceTests") or {}).get("passRate"),
            "falseDoneRate": (prior.get("falseDone") or {}).get("rate"),
            "reliableCases": len((prior.get("consistency") or {}).get("reliable") or []),
        })
    trend.append({
        "createdAt": created_at,
        "promptVariant": header.get("promptVariant"),
        "model": header.get("model"),
        "scorers": {name: row.get("passRate") for name, row in summary["scorers"].items()},
        "referenceTestsPassRate": (review.get("referenceTests") or {}).get("passRate"),
        "falseDoneRate": (review.get("falseDone") or {}).get("rate"),
        "reliableCases": len((review.get("consistency") or {}).get("reliable") or []),
    })
    planned_epochs = header.get("plannedEpochs")
    planned_samples = header.get("plannedSamples")
    planned_trials = (
        int(planned_epochs) * int(planned_samples)
        if planned_epochs and planned_samples
        else None
    )
    return {
        "schema": "eval-report/v1",
        "createdAt": created_at,
        "sourceLog": header.get("sourceLog"),
        "model": header.get("model"),
        "promptVariant": header.get("promptVariant"),
        "runStatus": header.get("runStatus"),
        "dryRun": bool(header.get("dryRun")),
        "plannedTrials": planned_trials,
        "criteria": CRITERIA,
        "summary": summary,
        "review": review,
        "brief": business_brief(sentence_review, baseline_review),
        "failures": failures_first(trials),
        "flaky": flaky_cases(trials),
        "trend": trend,
        "trials": trials,
    }


def render_html(report: dict[str, Any]) -> str:
    template = env.get_template("report.html.j2")
    return template.render(report=report, score_labels=SCORE_LABELS_FI, criteria_fi=CRITERIA_FI)


def write_report(report: dict[str, Any], dest_stem: Path) -> tuple[Path, Path]:
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    json_path = dest_stem.with_suffix(".json")
    html_path = dest_stem.with_suffix(".html")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")
    return html_path, json_path


def find_browser() -> str | None:
    names = ("msedge", "chrome", "google-chrome", "chromium")
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    candidates = [
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe"),
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def write_pdf(html_path: Path) -> Path | None:
    """Print the HTML report to PDF with Edge or Chrome. Returns None if neither exists."""
    browser = find_browser()
    if not browser:
        return None
    pdf_path = html_path.with_suffix(".pdf")
    if pdf_path.exists():
        pdf_path.unlink()
    uri = html_path.resolve().as_uri()
    commands = [
        [browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf_path}", uri],
        [browser, "--headless", "--disable-gpu", f"--print-to-pdf={pdf_path}", uri],
    ]
    for command in commands:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path
    return None


def report_stem_for_log(log_path: Path, reports_dir: Path = REPORTS_DIR) -> Path:
    """Deterministic output stem so the hook and the dashboard write the same files."""
    name = log_path.name
    for suffix in (".eval", ".evalz", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return reports_dir / name


def is_dry_run_report(report: dict[str, Any]) -> bool:
    """Explicit flag first; otherwise the 0% / 0% / 100% signature of a run without a model."""
    if report.get("dryRun"):
        return True
    scorers = (report.get("summary") or {}).get("scorers") or {}
    done = scorers.get("subtask_done") or {}
    tests = scorers.get("reference_tests_pass") or {}
    budget = scorers.get("process_budget") or {}
    n = int(done.get("n") or 0)
    return (
        n > 0
        and done.get("passRate") == 0
        and tests.get("passRate") == 0
        and budget.get("passRate") == 1
        and int((report.get("summary") or {}).get("totalTokens") or 0) == 0
    )


def find_report_for_log(log_path: Path, reports_dir: Path = REPORTS_DIR) -> Path | None:
    """JSON report already written from this log, by stem or by ``sourceLog``."""
    log_path = Path(log_path)
    stem_json = report_stem_for_log(log_path, reports_dir).with_suffix(".json")
    if stem_json.exists():
        return stem_json
    try:
        wanted = log_path.resolve()
    except OSError:
        wanted = log_path
    for row in load_previous_reports(reports_dir):
        source = row.get("sourceLog")
        if not source:
            continue
        candidate = Path(source)
        if not candidate.is_absolute():
            candidate = AGENT_ROOT / candidate
        try:
            same = candidate.resolve() == wanted
        except OSError:
            same = False
        if same or candidate.name == log_path.name:
            return Path(row["_path"])
    return None


def _report_is_current(json_path: Path, log_path: Path) -> bool:
    if not json_path.exists():
        return False
    try:
        return json_path.stat().st_mtime >= log_path.stat().st_mtime
    except OSError:
        return False


def finalize_log(
    log_path: Path,
    *,
    reports_dir: Path = REPORTS_DIR,
    log: Any = None,
    pdf_always: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Write HTML + JSON (+ PDF on success) for one finished log and rebuild the index.

    Idempotent: when the JSON is newer than the log and ``force`` is false,
    nothing is rewritten. Both the Inspect hook and the dashboard poller
    call this, so they must converge on the same files.
    """
    log_path = Path(log_path)
    stem = report_stem_for_log(log_path, reports_dir)
    json_path = stem.with_suffix(".json")
    html_path = stem.with_suffix(".html")
    pdf_path = stem.with_suffix(".pdf")
    if not force and _report_is_current(json_path, log_path):
        return {
            "stem": str(stem),
            "html": str(html_path),
            "json": str(json_path),
            "pdf": str(pdf_path) if pdf_path.exists() else None,
            "skipped": True,
        }
    if log is None:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(log_path))
    trials = trials_from_inspect_log(log)
    header = header_from_inspect_log(log, str(log_path))
    previous_list = [
        row for row in load_previous_reports(reports_dir)
        if Path(row.get("_path") or "").resolve() != json_path.resolve()
    ]
    report = build_report(
        trials,
        header=header,
        previous=previous_list[-1] if previous_list else None,
        trend_reports=previous_list,
    )
    html_path, json_path = write_report(report, stem)
    pdf_written: Path | None = None
    if header.get("runStatus") == "success" or pdf_always:
        pdf_written = write_pdf(html_path)
    build_index(reports_dir)
    return {
        "stem": str(stem),
        "html": str(html_path),
        "json": str(json_path),
        "pdf": str(pdf_written) if pdf_written else None,
        "runStatus": header.get("runStatus"),
        "trials": len(trials),
        "skipped": False,
    }


def index_rows(reports_dir: Path = REPORTS_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in load_previous_reports(reports_dir):
        json_path = Path(report["_path"])
        stem = json_path.with_suffix("")
        html_path = stem.with_suffix(".html")
        pdf_path = stem.with_suffix(".pdf")
        summary = report.get("summary") or {}
        scorers = summary.get("scorers") or {}
        review = report.get("review") or {}
        tests = scorers.get("reference_tests_pass") or {}
        rows.append({
            "name": stem.name,
            "createdAt": report.get("createdAt"),
            "model": report.get("model"),
            "promptVariant": report.get("promptVariant"),
            "runStatus": report.get("runStatus") or "unknown",
            "dryRun": is_dry_run_report(report),
            "nTrials": summary.get("nTrials"),
            "nSamples": summary.get("nSamples"),
            "plannedTrials": report.get("plannedTrials"),
            "referenceTestsPassRate": tests.get("passRate"),
            "referenceTestsStderr": tests.get("stderr"),
            "falseDoneRate": (review.get("falseDone") or {}).get("rate"),
            "reliableCases": len((review.get("consistency") or {}).get("reliable") or []),
            "minutesPerSuccess": (review.get("cost") or {}).get("minutesPerSuccess"),
            "gpuMinutes": (review.get("cost") or {}).get("gpuMinutes"),
            "headline": (report.get("brief") or {}).get("headline") or classification_line(review),
            "html": html_path.name if html_path.exists() else None,
            "pdf": pdf_path.name if pdf_path.exists() else None,
            "json": json_path.name,
        })
    rows.sort(key=lambda row: row.get("createdAt") or "", reverse=True)
    return rows


def build_index(reports_dir: Path = REPORTS_DIR) -> Path:
    """Write evals/reports/index.html listing every eval-report/v1 JSON."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = index_rows(reports_dir)
    template = env.get_template("index.html.j2")
    html = template.render(rows=rows, generatedAt=_utc_now())
    out = reports_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def compare_reports(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_scorers = (left.get("summary") or {}).get("scorers") or {}
    right_scorers = (right.get("summary") or {}).get("scorers") or {}
    names = sorted(set(left_scorers) | set(right_scorers))
    rows: list[dict[str, Any]] = []
    for name in names:
        a = left_scorers.get(name) or {"passRate": 0, "stderr": 0, "n": 0}
        b = right_scorers.get(name) or {"passRate": 0, "stderr": 0, "n": 0}
        delta = b["passRate"] - a["passRate"]
        combined = math.sqrt((a.get("stderr") or 0) ** 2 + (b.get("stderr") or 0) ** 2)
        rows.append({
            "scorer": name,
            "left": a["passRate"],
            "right": b["passRate"],
            "delta": round(delta, 4),
            "combinedStderr": round(combined, 4),
            "signal": abs(delta) > combined if combined > 0 else abs(delta) > 0,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render eval logs to HTML + JSON.")
    parser.add_argument("log", nargs="?", help="Inspect .eval log or canonical trials JSON.")
    parser.add_argument("--out", default="", help="Output stem (writes .html and .json).")
    parser.add_argument("--pdf", action="store_true", help="Also write a PDF via Edge or Chrome headless.")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT.json", "RIGHT.json"))
    args = parser.parse_args(argv)

    if args.compare:
        left = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        right = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        rows = compare_reports(left, right)
        print(json.dumps(rows, indent=2))
        for row in rows:
            flag = "SIGNAL" if row["signal"] else "noise"
            print(
                f"{row['scorer']}: {row['left']:.2%} -> {row['right']:.2%} "
                f"delta={row['delta']:+.2%} stderr={row['combinedStderr']:.2%} ({flag})"
            )
        return 0

    if not args.log:
        parser.error("log path is required unless --compare is used")
    source = Path(args.log)
    trials, header = load_trials(source)
    reports_dir = REPORTS_DIR
    previous_list = load_previous_reports(reports_dir)
    previous = previous_list[-1] if previous_list else None
    report = build_report(
        trials,
        header=header,
        previous=previous,
        trend_reports=previous_list,
    )
    stem = Path(args.out) if args.out else reports_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{report.get('promptVariant') or 'run'}"
    )
    html_path, json_path = write_report(report, stem)
    print(f"HTML: {html_path}")
    print(f"JSON: {json_path}")
    if args.pdf:
        pdf_path = write_pdf(html_path)
        if pdf_path:
            print(f"PDF: {pdf_path}")
        else:
            print("PDF: Edge/Chrome not found. Open the HTML and print to PDF.")
    if stem.resolve().parent == reports_dir.resolve():
        print(f"Index: {build_index(reports_dir)}")
    if report.get("runStatus") and report["runStatus"] != "success":
        print(f"run status: {report['runStatus']} (partial log, numbers cover finished trials only)")
    review = report.get("review") or {}
    false_done = review.get("falseDone") or {}
    if false_done.get("rate") is not None:
        print(
            f"false-done: {false_done['rate']:.2%} "
            f"({false_done['falseDone']}/{false_done['claimedDone']} DONE claims)"
        )
    cost = review.get("cost") or {}
    if cost.get("minutesPerSuccess") is not None:
        print(
            f"gpu minutes per full success: {cost['minutesPerSuccess']} "
            f"({cost['gpuMinutes']} min / {cost['fullSuccesses']} cases)"
        )
    elif cost:
        print(f"gpu minutes: {cost.get('gpuMinutes')} (no case passed every epoch)")
    for name, row in report["summary"]["scorers"].items():
        delta = row.get("delta")
        delta_txt = f" delta={delta:+.2%}" if delta is not None else ""
        print(f"{name}: {row['passRate']:.2%} ± {row['stderr']:.2%} (n={row['n']}){delta_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
