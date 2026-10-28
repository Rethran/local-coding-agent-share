"""Strict structured-output client for the requirements model."""

from __future__ import annotations

import json
from typing import Any, Callable

from jsonschema import Draft202012Validator
from ollama import chat


class StructuredOutputError(RuntimeError):
    """The model endpoint did not honor the requested JSON contract."""


class StructuredOllamaClient:
    def __init__(
        self,
        *,
        model: str = "qwen3:8b",
        num_ctx: int = 16384,
        num_predict: int = 4096,
        chat_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self._chat = chat_fn or chat

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        num_predict: int | None = None,
    ) -> dict[str, Any]:
        try:
            response = self._chat(
                model=self.model,
                messages=messages,
                format=schema,
                think=False,
                options={
                    "num_ctx": self.num_ctx,
                    "num_predict": num_predict or self.num_predict,
                    "temperature": 0,
                },
                keep_alive="30m",
            )
        except (TypeError, ValueError) as exc:
            raise StructuredOutputError(
                "Installed Ollama client/server does not accept JSON Schema structured output."
            ) from exc

        message = getattr(response, "message", None)
        if isinstance(response, dict):
            message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise StructuredOutputError("Ollama returned no textual structured response.")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise StructuredOutputError("Ollama returned malformed JSON despite schema mode.") from exc
        if not isinstance(payload, dict):
            raise StructuredOutputError("Ollama structured response must be a JSON object.")

        errors = sorted(
            Draft202012Validator(schema).iter_errors(payload),
            key=lambda item: list(item.absolute_path),
        )
        if errors:
            details = "; ".join(error.message for error in errors[:5])
            raise StructuredOutputError(f"Ollama response violated its JSON Schema: {details}")
        return payload
