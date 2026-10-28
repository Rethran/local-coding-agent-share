"""Tests for orchestrator Ollama client wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

AGENT_ROOT = Path(__file__).resolve().parent.parent
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from orchestrator.config import DEFAULT_NUM_PREDICT  # noqa: E402
from orchestrator.ollama_client import DEFAULT_KEEP_ALIVE, ollama_chat, warmup_ollama_model  # noqa: E402


def test_ollama_chat_passes_options_and_keep_alive() -> None:
    mock_response = MagicMock()
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    with patch("orchestrator.ollama_client.chat", return_value=mock_response) as mock_chat:
        result = ollama_chat(
            model="heat-agent:latest",
            messages=messages,
            tools=tools,
            num_ctx=24576,
        )

    assert result is mock_response
    mock_chat.assert_called_once_with(
        model="heat-agent:latest",
        messages=messages,
        tools=tools,
        think=None,
        options={
            "num_ctx": 24576,
            "num_predict": DEFAULT_NUM_PREDICT,
            "temperature": 0.2,
        },
        keep_alive=DEFAULT_KEEP_ALIVE,
    )


def test_warmup_ollama_model_calls_chat_with_minimal_prompt() -> None:
    with patch("orchestrator.ollama_client.ollama_chat") as mock_chat:
        elapsed = warmup_ollama_model(model="heat-agent:latest", num_ctx=16384)

    assert elapsed >= 0
    mock_chat.assert_called_once_with(
        model="heat-agent:latest",
        messages=[{"role": "user", "content": "ok"}],
        num_ctx=16384,
        num_predict=8,
        keep_alive=DEFAULT_KEEP_ALIVE,
        think=False,
    )


def test_ollama_chat_forwards_thinking_effort() -> None:
    with patch("orchestrator.ollama_client.chat") as mock_chat:
        ollama_chat(
            model="heat-agent:latest",
            messages=[{"role": "user", "content": "inspect"}],
            num_ctx=24576,
            think="medium",
        )

    assert mock_chat.call_args.kwargs["think"] == "medium"
