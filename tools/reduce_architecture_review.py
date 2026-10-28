import json

from tools._common import mcp
from ollama_client import reduce_architecture


@mcp.tool()
def reduce_architecture_review(
    summaries_json: str,
    focus_question: str = "Tarkista kerrosarkkitehtuurin rikkeet ja syklit",
) -> str:
    """
    Combines the map-phase summaries and produces an overall architecture assessment.
    The context is small at this stage (summaries rather than raw source code), making it well suited for a local 27B model.
    """
    summaries = json.loads(summaries_json)
    result = reduce_architecture(summaries, focus_question)
    return result
