"""Persistent state for an isolated requirements-generation run."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_STAGES = {
    "ELICITING",
    "DECOMPOSING",
    "VALIDATING",
    "READY_TO_PUBLISH",
    "PUBLISHED",
}


@dataclass
class GenerationState:
    run_id: str = ""
    stage: str = "ELICITING"
    fields: dict[str, Any] = field(default_factory=dict)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    last_rejected: dict[str, Any] | None = None
    validation_errors: list[str] = field(default_factory=list)
    published_profile: str | None = None
    schema_version: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationState":
        state = cls(
            run_id=str(payload.get("run_id") or ""),
            stage=str(payload.get("stage") or "ELICITING"),
            fields=dict(payload.get("fields") or {}),
            accepted=list(payload.get("accepted") or []),
            last_rejected=payload.get("last_rejected"),
            validation_errors=[str(item) for item in payload.get("validation_errors") or []],
            published_profile=payload.get("published_profile"),
            schema_version=int(payload.get("schema_version") or 1),
        )
        if state.stage not in VALID_STAGES:
            raise ValueError(f"Unknown generation stage: {state.stage}")
        return state

    def uncovered_acceptance_ids(self) -> list[str]:
        signals = self.fields.get("acceptanceSignals") or []
        expected = [str(item["id"]) for item in signals if isinstance(item, dict) and item.get("id")]
        covered = {
            str(coverage_id)
            for entry in self.accepted
            for coverage_id in entry.get("covers") or []
        }
        return [signal_id for signal_id in expected if signal_id not in covered]


class StateStore:
    """Write state and draft tasks atomically inside one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "state.json"
        self.drafts_dir = self.run_dir / "drafts"

    def load(self) -> GenerationState:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"State must be a JSON object: {self.path}")
        return GenerationState.from_dict(payload)

    def save(self, state: GenerationState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.drafts_dir.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(
            json.dumps(asdict(state), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, self.path)
        self._sync_drafts(state)

    def _sync_drafts(self, state: GenerationState) -> None:
        expected: set[str] = set()
        for entry in state.accepted:
            task = entry.get("task")
            if not isinstance(task, dict) or not task.get("id"):
                continue
            name = f"{task['id']}.json"
            expected.add(name)
            path = self.drafts_dir / name
            temp = path.with_suffix(".json.tmp")
            temp.write_text(
                json.dumps(task, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(temp, path)
        for path in self.drafts_dir.glob("*.json"):
            if path.name not in expected:
                path.unlink()
