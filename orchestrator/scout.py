"""Parallel deterministic scouts for compact, task-agnostic source discovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from tools._common import safe_path

from orchestrator.read_guard import normalize_relative_path

MAX_SCOUTS = 4
MAX_SCOUT_EXCERPT_CHARS = 1_500


@dataclass(frozen=True)
class ScoutRequest:
    path: str
    focus: str


@dataclass(frozen=True)
class ScoutFinding:
    path: str
    focus: str
    status: str
    summary: str
    excerpt: str = ""


def build_scout_plan(merged_subtask: dict[str, Any]) -> list[ScoutRequest]:
    """Build a generic file-oriented scout plan from any subtask definition."""
    candidates: list[ScoutRequest] = []
    target = merged_subtask.get("targetArtifacts") or {}
    if isinstance(target, dict):
        for key in ("module", "test"):
            if target.get(key):
                candidates.append(
                    ScoutRequest(
                        path=normalize_relative_path(str(target[key])),
                        focus=f"Current {key} artifact",
                    )
                )
    for artifact in merged_subtask.get("relatedArtifacts") or []:
        if isinstance(artifact, dict) and artifact.get("path"):
            candidates.append(
                ScoutRequest(
                    path=normalize_relative_path(str(artifact["path"])),
                    focus=str(artifact.get("purpose") or "Related artifact"),
                )
            )

    deduplicated: list[ScoutRequest] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.path and candidate.path not in seen:
            seen.add(candidate.path)
            deduplicated.append(candidate)
    return deduplicated[:MAX_SCOUTS]


def _existing_sources(context_pack: dict[str, Any]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for snippet in context_pack.get("referenceSnippets") or []:
        if isinstance(snippet, dict) and snippet.get("path"):
            path = normalize_relative_path(str(snippet["path"]))
            sources[path] = str(snippet.get("source") or "")
    snapshot = context_pack.get("targetFileSnapshot") or {}
    if isinstance(snapshot, dict) and snapshot.get("path"):
        path = normalize_relative_path(str(snapshot["path"]))
        sources.setdefault(path, str(snapshot.get("excerpt") or ""))
    return sources


def _run_scout(request: ScoutRequest, cached_sources: dict[str, str]) -> ScoutFinding:
    source = cached_sources.get(request.path)
    if source is None:
        file_path = safe_path(request.path)
        if not file_path.exists() or not file_path.is_file():
            return ScoutFinding(
                path=request.path,
                focus=request.focus,
                status="missing",
                summary=f"{request.path} does not exist yet.",
            )
        try:
            source = file_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return ScoutFinding(
                path=request.path,
                focus=request.focus,
                status="error",
                summary=f"Could not read {request.path}: {exc}",
            )

    lines = source.splitlines()
    excerpt = source[:MAX_SCOUT_EXCERPT_CHARS]
    if len(source) > MAX_SCOUT_EXCERPT_CHARS:
        excerpt += "\n[scout excerpt truncated]"
    return ScoutFinding(
        path=request.path,
        focus=request.focus,
        status="ok",
        summary=f"{len(lines)} lines and {len(source)} characters available.",
        excerpt=excerpt,
    )


def run_parallel_scouts(
    merged_subtask: dict[str, Any],
    context_pack: dict[str, Any],
) -> dict[str, Any]:
    """Run independent source scouts concurrently without parallel LLM inference."""
    plan = build_scout_plan(merged_subtask)
    cached_sources = _existing_sources(context_pack)
    if not plan:
        return {"role": "scout", "parallel": False, "findings": [], "coveredPaths": []}

    max_workers = min(len(plan), MAX_SCOUTS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        findings = list(
            executor.map(lambda request: _run_scout(request, cached_sources), plan)
        )

    return {
        "role": "scout",
        "parallel": len(plan) > 1,
        "findings": [asdict(finding) for finding in findings],
        "coveredPaths": [
            finding.path for finding in findings if finding.status == "ok"
        ],
    }


def attach_scout_brief(
    context_pack: dict[str, Any],
    merged_subtask: dict[str, Any],
) -> dict[str, Any]:
    """Return a context-pack copy enriched with generic scout findings."""
    enriched = dict(context_pack)
    enriched["scoutBrief"] = run_parallel_scouts(merged_subtask, context_pack)
    return enriched
