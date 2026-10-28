"""Thin wrapper around ollama.chat with inference options and keep-alive."""

from __future__ import annotations

import time
from typing import Any

from ollama import chat

from orchestrator.config import DEFAULT_NUM_PREDICT
from orchestrator.log import get_logger
from orchestrator.thinking_policy import ThinkingEffort

logger = get_logger(__name__)

DEFAULT_KEEP_ALIVE = "30m"


def ollama_chat(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    num_ctx: int,
    num_predict: int = DEFAULT_NUM_PREDICT,
    keep_alive: str = DEFAULT_KEEP_ALIVE,
    think: bool | ThinkingEffort | None = None,
) -> Any:
    return chat(
        model=model,
        messages=messages,
        tools=tools,
        think=think,
        options={
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0.2,
        },
        keep_alive=keep_alive,
    )


def warmup_ollama_model(*, model: str, num_ctx: int, keep_alive: str = DEFAULT_KEEP_ALIVE) -> float:
    """Load the model into memory before the first worker step. Returns elapsed seconds."""
    started = time.perf_counter()
    ollama_chat(
        model=model,
        messages=[{"role": "user", "content": "ok"}],
        num_ctx=num_ctx,
        num_predict=8,
        keep_alive=keep_alive,
        think=False,
    )
    elapsed = time.perf_counter() - started
    logger.info("Ollama warmup for %s finished in %.1fs (num_ctx=%d)", model, elapsed, num_ctx)
    return elapsed
