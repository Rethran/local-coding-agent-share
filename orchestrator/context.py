from __future__ import annotations

import json
from typing import Any

from tools import _handoffs, _tasks

from orchestrator.config import DEFAULT_CHARS_PER_TOKEN
from orchestrator.continuity import latest_packet_path, load_continuity_packet, persist_and_sync
from orchestrator.prompts import system_prompt
from orchestrator.session import SessionStore, get_session


def estimate_messages_chars(messages: list[dict[str, Any]]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, default=str))


def context_budget_chars(num_ctx: int, chars_per_token: int = DEFAULT_CHARS_PER_TOKEN) -> int:
    return max(int(num_ctx) * max(int(chars_per_token), 1), 1)


def context_usage_ratio(
    messages: list[dict[str, Any]],
    context_budget_chars_value: int,
    *,
    prompt_eval_count: int | None = None,
    num_ctx: int | None = None,
) -> float:
    if prompt_eval_count is not None and num_ctx and num_ctx > 0:
        return float(prompt_eval_count) / float(num_ctx)
    budget = max(int(context_budget_chars_value), 1)
    return estimate_messages_chars(messages) / budget


def extract_prompt_eval_count(response: Any) -> int | None:
    if response is None:
        return None
    if isinstance(response, dict):
        value = response.get("prompt_eval_count")
    else:
        value = getattr(response, "prompt_eval_count", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_eval_count(response: Any) -> int | None:
    if response is None:
        return None
    if isinstance(response, dict):
        value = response.get("eval_count")
    else:
        value = getattr(response, "eval_count", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def should_handover_context(
    *,
    usage_ratio: float,
    threshold: float,
    handovers_used: int,
    max_handovers: int,
) -> bool:
    if max_handovers <= 0:
        return False
    if handovers_used >= max_handovers:
        return False
    return usage_ratio >= threshold


def build_context_continuation_prompt(
    task_id: str,
    subtask_id: str,
    *,
    previous_handoff_id: str,
    usage_ratio: float,
    handover_number: int,
    max_handovers: int,
    session: SessionStore | None = None,
) -> str:
    continuity = load_continuity_packet(task_id, subtask_id)
    active_session = session or get_session()
    session_block = ""
    if active_session is not None:
        payload = active_session.get_session_payload_for_retry(task_id)
        session_block = (
            "\nSession lessons (supplementary; continuityPacket is authoritative):\n"
            f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n"
        )

    continuity_block = ""
    if continuity:
        continuity_block = (
            "\nContinuity packet (read first; obey doNotRepeat; execute nextStep):\n"
            f"{json.dumps(continuity, indent=2, ensure_ascii=False)}\n"
        )
    else:
        prompt = _tasks.build_handoff_prompt(
            task_id=task_id,
            handoff_id=previous_handoff_id or f"{task_id}-0000",
            handoff_path="",
            action="context_continue",
        )
        continuity_block = f"\n{prompt}\n"

    return (
        f"Context was reset for {task_id}.{subtask_id} at {usage_ratio:.0%} usage "
        f"(handover {handover_number}/{max_handovers}). "
        f"Continue without redoing verified work. Last handoff: {previous_handoff_id or 'none'}."
        f"{continuity_block}"
        f"{session_block}"
    )


def perform_context_handover(
    *,
    task_id: str,
    subtask_id: str,
    attempt_number: int,
    previous_handoff_id: str,
    usage_ratio: float,
    handover_number: int,
    max_handovers: int,
    session: SessionStore | None = None,
    merged_subtask: dict[str, Any] | None = None,
    role: str = "implementer",
) -> tuple[list[dict[str, Any]], str]:
    active_session = session or get_session()
    if active_session is not None:
        state = _tasks.load_task_state(task_id) or {}
        active_session.snapshot_before_handover(task_id, state)

    blocker = (
        f"Context handover {handover_number}/{max_handovers} at {usage_ratio:.0%} usage."
    )
    packet = persist_and_sync(
        task_id=task_id,
        subtask_id=subtask_id,
        attempt=attempt_number,
        status="handover",
        merged_subtask=merged_subtask,
        session=active_session,
        stop_reason="context_handover",
        blockers=[blocker],
        last_handoff_id=previous_handoff_id,
        recommended_resume_step=(
            f"Continue {subtask_id} from continuityPacket nextStep without re-reading "
            "artifacts listed in doNotRepeat."
        ),
    )

    handoff = _handoffs.record_handoff(
        task_id=task_id,
        tool="context_handover",
        input_data={
            "usageRatio": round(usage_ratio, 4),
            "handoverNumber": handover_number,
            "maxHandovers": max_handovers,
            "previousHandoffId": previous_handoff_id or None,
            "subtaskId": subtask_id,
        },
        result={
            "status": "ok",
            "output": json.dumps(
                {
                    "continuityPacketPath": str(
                        latest_packet_path(task_id, subtask_id).as_posix()
                    ),
                    "nextStep": packet.get("nextStep"),
                },
                ensure_ascii=False,
            ),
        },
        summary=(
            f"Context handover {handover_number}/{max_handovers} "
            f"at {usage_ratio:.0%} usage."
        ),
        previous_handoff_id=previous_handoff_id,
        chain_id=task_id,
        next_action={
            "tool": "replace_in_file",
            "hint": packet.get("nextStep"),
        },
    )
    new_handoff_id = handoff["compact"]["handoffId"]
    continuation = build_context_continuation_prompt(
        task_id,
        subtask_id,
        previous_handoff_id=previous_handoff_id or new_handoff_id,
        usage_ratio=usage_ratio,
        handover_number=handover_number,
        max_handovers=max_handovers,
        session=active_session,
    )
    messages = [
        {"role": "system", "content": system_prompt(subtask_id, role=role)},
        {"role": "user", "content": continuation},
    ]
    print(
        f"{task_id}.{subtask_id}: context handover {handover_number}/{max_handovers} "
        f"at {usage_ratio:.0%} (handoff {new_handoff_id})"
    )
    return messages, new_handoff_id
