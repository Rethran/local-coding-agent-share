"""Local dashboard: progress of eval runs read from the .eval logs.

    python -m evals dashboard --port 8765

One poller thread re-reads ``evals/logs/*.eval`` every few seconds. The
numbers are the same functions the final report uses, run over a partial
log, so they wobble until the last epoch lands. Logs that have finished but
have no report yet are finalized here as a fallback for runs started
without the hook. Binds 127.0.0.1 only.

``inspect view`` already browses transcripts. This page adds what it does
not show: done/total against the planned trial count, ETA, the reviewer
numbers so far, and links to the HTML/PDF/JSON reports.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from evals.hooks import LIVE_DIR, read_heartbeat
from evals.report import (
    REPORTS_DIR,
    build_review,
    business_brief,
    env,
    finalize_log,
    find_report_for_log,
    header_from_inspect_log,
    index_rows,
    report_stem_for_log,
    summarize,
    trials_from_inspect_log,
)

LOGS_DIR = Path(__file__).resolve().parent / "logs"
TERMINAL = {"success", "error", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def run_progress(
    trials: list[dict[str, Any]],
    header: dict[str, Any],
    heartbeat: dict[str, Any] | None = None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Pure function: progress numbers for one run from its finished trials."""
    now = now or datetime.now(timezone.utc)
    planned_epochs = header.get("plannedEpochs")
    planned_samples = header.get("plannedSamples")
    total = (
        int(planned_epochs) * int(planned_samples)
        if planned_epochs and planned_samples
        else None
    )
    done = len(trials)
    heartbeat = heartbeat or {}
    if total is None and heartbeat.get("total"):
        total = int(heartbeat["total"])
    walls = [float(trial.get("wallSeconds") or 0) for trial in trials]
    median_wall = float(statistics.median(walls)) if walls else None
    remaining = (total - done) if total is not None else None
    eta_seconds = (
        remaining * median_wall
        if remaining is not None and remaining > 0 and median_wall is not None
        else (0.0 if remaining == 0 else None)
    )
    current = heartbeat.get("current") or None
    current_elapsed = None
    if current:
        started = _parse_iso(current.get("startedAt"))
        if started:
            current_elapsed = max(0.0, (now - started).total_seconds())
    status = header.get("runStatus") or "started"
    percent = (done / total * 100.0) if total else None
    recent = []
    for trial in sorted(trials, key=lambda row: (row.get("epoch") or 0, row.get("sampleId")))[-5:]:
        recent.append({
            "sampleId": trial.get("sampleId"),
            "epoch": trial.get("epoch"),
            "wallSeconds": round(float(trial.get("wallSeconds") or 0), 1),
            "scores": {
                name: bool(score.get("ok"))
                for name, score in (trial.get("scores") or {}).items()
            },
        })
    review = build_review(trials)
    summary = summarize(trials) if trials else {"scorers": {}, "nTrials": 0, "nSamples": 0}
    return {
        "status": status,
        "done": done,
        "total": total,
        "percent": round(percent, 1) if percent is not None else None,
        "medianWallSeconds": round(median_wall, 1) if median_wall is not None else None,
        "etaSeconds": round(eta_seconds, 0) if eta_seconds is not None else None,
        "current": {
            "sampleId": current.get("sampleId"),
            "epoch": current.get("epoch"),
            "elapsedSeconds": round(current_elapsed, 0) if current_elapsed is not None else None,
        } if current else None,
        "heartbeatAt": heartbeat.get("updatedAt"),
        "recent": recent,
        "scorers": {
            name: {"passRate": row.get("passRate"), "n": row.get("n")}
            for name, row in (summary.get("scorers") or {}).items()
        },
        "consistency": {
            "reliable": len((review.get("consistency") or {}).get("reliable") or []),
            "fragile": len((review.get("consistency") or {}).get("fragile") or []),
            "cannot": len((review.get("consistency") or {}).get("cannot") or []),
        },
        "falseDone": review.get("falseDone") or {},
        "failureTypes": (review.get("failureTypes") or {}).get("counts") or {},
        "failureLabels": (review.get("failureTypes") or {}).get("labels") or {},
        "cost": review.get("cost") or {},
        "brief": business_brief(review),
    }


class RunCache:
    """Reads each log fully only when its mtime changes."""

    def __init__(
        self,
        logs_dir: Path = LOGS_DIR,
        reports_dir: Path = REPORTS_DIR,
        live_dir: Path = LIVE_DIR,
        *,
        finalize: bool = True,
    ) -> None:
        self.logs_dir = logs_dir
        self.reports_dir = reports_dir
        self.live_dir = live_dir
        self.finalize = finalize
        self._entries: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.last_poll: str | None = None
        self.last_error: str | None = None

    def _heartbeats(self) -> dict[str, dict[str, Any]]:
        beats: dict[str, dict[str, Any]] = {}
        if not self.live_dir.exists():
            return beats
        for path in self.live_dir.glob("*.json"):
            data = read_heartbeat(path)
            if data.get("evalId"):
                beats[str(data["evalId"])] = data
        return beats

    def _read(self, path: Path) -> dict[str, Any]:
        from inspect_ai.log import read_eval_log

        log = read_eval_log(str(path))
        trials = trials_from_inspect_log(log)
        header = header_from_inspect_log(log, str(path))
        return {
            "path": str(path),
            "name": path.name,
            "evalId": getattr(log.eval, "eval_id", None),
            "runId": getattr(log.eval, "run_id", None),
            "createdAt": getattr(log.eval, "created", None),
            "startedAt": getattr(log.stats, "started_at", None) if getattr(log, "stats", None) else None,
            "completedAt": getattr(log.stats, "completed_at", None) if getattr(log, "stats", None) else None,
            "header": header,
            "trials": trials,
            "mtime": path.stat().st_mtime,
            "log": log,
        }

    def poll(self) -> None:
        try:
            beats = self._heartbeats()
            seen: set[str] = set()
            for path in sorted(self.logs_dir.glob("*.eval")):
                key = str(path)
                seen.add(key)
                mtime = path.stat().st_mtime
                entry = self._entries.get(key)
                if entry is None or entry["mtime"] != mtime:
                    try:
                        entry = self._read(path)
                    except Exception as exc:  # noqa: BLE001
                        entry = {
                            "path": key, "name": path.name, "evalId": None, "runId": None,
                            "createdAt": None, "startedAt": None, "completedAt": None,
                            "header": {"runStatus": "unreadable", "sourceLog": key},
                            "trials": [], "mtime": mtime, "log": None, "error": str(exc),
                        }
                    with self._lock:
                        self._entries[key] = entry
                status = (entry["header"] or {}).get("runStatus")
                if (
                    self.finalize
                    and status in TERMINAL
                    and entry.get("log") is not None
                    and find_report_for_log(path, self.reports_dir) is None
                ):
                    try:
                        finalize_log(path, reports_dir=self.reports_dir, log=entry["log"])
                    except Exception as exc:  # noqa: BLE001
                        entry["finalizeError"] = str(exc)
                entry["heartbeat"] = beats.get(str(entry.get("evalId") or ""), {})
            with self._lock:
                for key in list(self._entries):
                    if key not in seen:
                        del self._entries[key]
                self.last_poll = _utc_now()
                self.last_error = None
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self.last_error = str(exc)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            entries = list(self._entries.values())
            last_poll = self.last_poll
            last_error = self.last_error
        runs = []
        for entry in entries:
            header = entry.get("header") or {}
            progress = run_progress(entry.get("trials") or [], header, entry.get("heartbeat") or {})
            found = find_report_for_log(Path(entry["path"]), self.reports_dir)
            stem = found.with_suffix("") if found else report_stem_for_log(Path(entry["path"]), self.reports_dir)
            runs.append({
                "name": entry["name"],
                "path": entry["path"],
                "evalId": entry.get("evalId"),
                "model": header.get("model"),
                "promptVariant": header.get("promptVariant"),
                "dryRun": bool(header.get("dryRun")),
                "createdAt": entry.get("createdAt"),
                "startedAt": entry.get("startedAt"),
                "completedAt": entry.get("completedAt"),
                "error": entry.get("error"),
                "finalizeError": entry.get("finalizeError"),
                "progress": progress,
                "report": {
                    "html": stem.with_suffix(".html").name if stem.with_suffix(".html").exists() else None,
                    "pdf": stem.with_suffix(".pdf").name if stem.with_suffix(".pdf").exists() else None,
                    "json": stem.with_suffix(".json").name if stem.with_suffix(".json").exists() else None,
                },
            })
        runs.sort(key=lambda row: row.get("createdAt") or "", reverse=True)
        return {
            "generatedAt": _utc_now(),
            "lastPoll": last_poll,
            "lastError": last_error,
            "logsDir": str(self.logs_dir),
            "reportsDir": str(self.reports_dir),
            "runs": runs,
            "reports": index_rows(self.reports_dir),
        }


def _poll_loop(cache: RunCache, interval: float, stop: threading.Event) -> None:
    while not stop.is_set():
        cache.poll()
        stop.wait(interval)


def make_handler(cache: RunCache) -> type[BaseHTTPRequestHandler]:
    reports_dir = cache.reports_dir.resolve()
    content_types = {
        ".html": "text/html; charset=utf-8",
        ".json": "application/json; charset=utf-8",
        ".pdf": "application/pdf",
        ".css": "text/css; charset=utf-8",
    }

    class Handler(BaseHTTPRequestHandler):
        server_version = "evals-dashboard/1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                html = env.get_template("dashboard.html.j2").render(snapshot=cache.snapshot())
                self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/runs":
                body = json.dumps(cache.snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            if path.startswith("/reports/"):
                name = unquote(path[len("/reports/"):])
                target = (reports_dir / name).resolve()
                if target.parent != reports_dir or not target.is_file():
                    self._send(404, b"not found", "text/plain; charset=utf-8")
                    return
                content_type = content_types.get(target.suffix.lower(), "application/octet-stream")
                self._send(200, target.read_bytes(), content_type)
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(
    *,
    port: int = 8765,
    interval: float = 10.0,
    finalize: bool = True,
    logs_dir: Path = LOGS_DIR,
    reports_dir: Path = REPORTS_DIR,
    live_dir: Path = LIVE_DIR,
    block: bool = True,
) -> tuple[ThreadingHTTPServer, threading.Event]:
    cache = RunCache(logs_dir, reports_dir, live_dir, finalize=finalize)
    cache.poll()
    stop = threading.Event()
    poller = threading.Thread(target=_poll_loop, args=(cache, interval, stop), daemon=True)
    poller.start()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(cache))
    server.daemon_threads = True
    if block:
        print(f"dashboard: http://127.0.0.1:{server.server_address[1]}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            server.server_close()
    else:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, stop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the eval progress dashboard on localhost.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between log polls.")
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Do not write reports for finished logs that have none.",
    )
    args = parser.parse_args(argv)
    serve(port=args.port, interval=args.interval, finalize=not args.no_finalize)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
