"""Inspect hooks: heartbeat for the dashboard, report + PDF when the task ends.

Imported by ``evals/tdd_coder_task.py`` so the hook is registered whenever
that task file is loaded. A report failure must never fail the eval, so
every callback swallows its own exceptions and prints one line instead.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspect_ai.hooks import Hooks, SampleEnd, SampleStart, TaskEnd, TaskStart, hooks

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LIVE_DIR = LOGS_DIR / ".live"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def heartbeat_path(eval_id: str, live_dir: Path = LIVE_DIR) -> Path:
    return live_dir / f"{eval_id}.json"


def read_heartbeat(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_heartbeat(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _planned_total(spec: Any) -> int | None:
    try:
        epochs = int(spec.config.epochs or 1)
        samples = int(spec.dataset.samples or 0)
    except (AttributeError, TypeError, ValueError):
        return None
    return epochs * samples if samples else None


class ReportHooks(Hooks):
    """Writes evals/logs/.live/<eval_id>.json and finalizes the report on task end."""

    def __init__(self, live_dir: Path = LIVE_DIR) -> None:
        self.live_dir = live_dir
        self._done: dict[str, int] = {}

    def _path(self, eval_id: str) -> Path:
        return heartbeat_path(eval_id, self.live_dir)

    async def on_task_start(self, data: TaskStart) -> None:
        try:
            self._done[data.eval_id] = 0
            meta = getattr(data.spec, "metadata", None) or {}
            write_heartbeat(self._path(data.eval_id), {
                "evalId": data.eval_id,
                "runId": data.run_id,
                "logLocation": getattr(data.spec, "location", None) or None,
                "modelTag": meta.get("model_tag"),
                "promptVariant": meta.get("prompt_variant"),
                "startedAt": _now(),
                "updatedAt": _now(),
                "done": 0,
                "total": _planned_total(data.spec),
                "current": None,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"[evals.hooks] task_start heartbeat failed: {exc}", file=sys.stderr)

    async def on_sample_start(self, data: SampleStart) -> None:
        try:
            path = self._path(data.eval_id)
            payload = read_heartbeat(path)
            payload.update({
                "evalId": data.eval_id,
                "updatedAt": _now(),
                "done": self._done.get(data.eval_id, payload.get("done") or 0),
                "current": {
                    "sampleId": str(data.summary.id),
                    "epoch": int(data.summary.epoch),
                    "startedAt": _now(),
                },
            })
            payload.setdefault("total", None)
            write_heartbeat(path, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[evals.hooks] sample_start heartbeat failed: {exc}", file=sys.stderr)

    async def on_sample_end(self, data: SampleEnd) -> None:
        try:
            self._done[data.eval_id] = self._done.get(data.eval_id, 0) + 1
            path = self._path(data.eval_id)
            payload = read_heartbeat(path)
            payload.update({
                "evalId": data.eval_id,
                "updatedAt": _now(),
                "done": self._done[data.eval_id],
                "current": None,
                "last": {
                    "sampleId": str(data.sample.id),
                    "epoch": int(data.sample.epoch),
                    "error": bool(getattr(data.sample, "error", None)),
                },
            })
            write_heartbeat(path, payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[evals.hooks] sample_end heartbeat failed: {exc}", file=sys.stderr)

    async def on_task_end(self, data: TaskEnd) -> None:
        try:
            location = getattr(data.log, "location", None)
            if location:
                from evals.report import finalize_log

                result = finalize_log(Path(location), log=data.log)
                if not result.get("skipped"):
                    print(f"[evals.hooks] report: {result['html']}")
                    if result.get("pdf"):
                        print(f"[evals.hooks] pdf: {result['pdf']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[evals.hooks] finalize failed: {exc}", file=sys.stderr)
        finally:
            self._done.pop(data.eval_id, None)
            try:
                self._path(data.eval_id).unlink(missing_ok=True)
            except OSError:
                pass


@hooks(name="tdd_coder_report", description="Dashboard heartbeat and HTML/PDF report on task end.")
def tdd_coder_report_hooks() -> type[ReportHooks]:
    return ReportHooks
