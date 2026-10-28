"""Deterministic field elicitation with model-phrased questions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from requirements_orchestrator.ollama_client import StructuredOllamaClient, StructuredOutputError
from requirements_orchestrator.state import GenerationState

FIELD_SPECS: tuple[tuple[str, str], ...] = (
    ("projectName", "a short human-readable project name"),
    ("taskPrefix", "a 2-12 character uppercase task ID prefix, such as SHOP"),
    ("workspaceRoot", "the absolute path of the target implementation workspace"),
    ("goal", "the concrete product outcome to build"),
    ("constraints", "hard constraints, separated by semicolons"),
    ("nonGoals", "explicitly excluded behavior, separated by semicolons"),
    ("acceptanceSignals", "observable acceptance outcomes, separated by semicolons"),
    ("existingDependencies", "existing libraries, services, or systems, separated by semicolons"),
    ("technicalContext", "language, framework, architecture, and relevant repository context"),
    ("implementationPrefixes", "allowed relative implementation folders, separated by semicolons"),
    ("verificationConventions", "the test runner and focused-test command convention"),
)
LIST_FIELDS = {
    "constraints",
    "nonGoals",
    "existingDependencies",
    "implementationPrefixes",
}
QUESTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["question"],
    "properties": {
        "question": {"type": "string", "minLength": 2, "maxLength": 300},
    },
}


def first_missing_field(state: GenerationState) -> tuple[str, str] | None:
    for name, description in FIELD_SPECS:
        value = state.fields.get(name)
        if value is None or value == "" or value == []:
            return name, description
    return None


def elicit_next_question(
    state: GenerationState,
    client: StructuredOllamaClient,
) -> tuple[str, str] | None:
    missing = first_missing_field(state)
    if missing is None:
        return None
    field_name, description = missing
    payload = client.generate(
        messages=[
            {
                "role": "system",
                "content": (
                    "Phrase exactly one short terminal question for the specified field. "
                    "Do not ask about any other field. Return only the schema object."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Field: {field_name}\nMeaning: {description}\n"
                    f"Already filled field names: {sorted(state.fields)}"
                ),
            },
        ],
        schema=QUESTION_SCHEMA,
        num_predict=96,
    )
    question = str(payload["question"]).strip()
    if "\n" in question or len(re.findall(r"[?!]", question)) > 1:
        raise StructuredOutputError("Elicitation model must return one short question sentence.")
    return field_name, question


def _split_items(answer: str) -> list[str]:
    return [item.strip() for item in answer.split(";") if item.strip()]


def normalize_answer(field_name: str, answer: str) -> Any:
    text = answer.strip()
    if not text:
        raise ValueError("Answer cannot be empty.")
    if field_name == "taskPrefix":
        prefix = re.sub(r"[^A-Za-z0-9_-]", "", text).upper()
        if not re.fullmatch(r"[A-Z][A-Z0-9_-]{1,11}", prefix):
            raise ValueError("Task prefix must be 2-12 uppercase letters, digits, '_' or '-'.")
        return prefix
    if field_name == "workspaceRoot":
        path = Path(text).expanduser()
        if not path.is_absolute():
            raise ValueError("Workspace root must be an absolute path.")
        return str(path.resolve())
    if field_name == "acceptanceSignals":
        items = _split_items(text)
        if not items:
            raise ValueError("Provide at least one acceptance signal.")
        return [
            {"id": f"ACCEPT-{index:03d}", "text": item}
            for index, item in enumerate(items, start=1)
        ]
    if field_name in LIST_FIELDS:
        items = _split_items(text)
        if not items:
            raise ValueError("Provide at least one semicolon-separated item.")
        if field_name == "implementationPrefixes":
            return [item.replace("\\", "/").strip("/") + "/" for item in items]
        return items
    return text
