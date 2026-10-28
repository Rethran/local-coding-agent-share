import json
import sys
from pathlib import Path

# Allow importing repair helpers when Continue runs this hook from repo cwd.
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from tools._repair import parse_xml_tool_call, looks_like_xml_tool_call
except ImportError:
    parse_xml_tool_call = None
    looks_like_xml_tool_call = lambda text: "<tool_call>" in text  # noqa: E731


def main():
    try:
        input_data = sys.stdin.read()
        if not input_data.strip():
            sys.exit(0)

        hook_payload = json.loads(input_data)
        tool_name = hook_payload.get("tool", "")
        tool_output = hook_payload.get("output", "")
    except Exception:
        sys.exit(0)

    if tool_name == "run_command":
        if len(tool_output) > 12000:
            truncated_output = (
                tool_output[:12000]
                + "\n\n[MUKAUTETTU VALIDOINTI: Tuloste katkaistu liian suuren koon vuoksi token-säästön takia.]"
            )
            sys.stdout.write(json.dumps({"output": truncated_output}))
            sys.exit(0)

    prohibited_keywords = ["AWS_SECRET_ACCESS_KEY", "DATABASE_PASSWORD", "PRIVATE_KEY"]
    for keyword in prohibited_keywords:
        if keyword in tool_output:
            sys.stderr.write(
                f"Validointivirhe: Työkalun '{tool_name}' tuloste evätty turvallisuuspolitiikan vuoksi (Havaittu herkkää dataa)."
            )
            sys.exit(2)

    # =========================================================================
    # SÄÄNTÖ 3: XML tool-call text in output (model echoed <tool_call> as text)
    # =========================================================================
    if looks_like_xml_tool_call(tool_output):
        hint = _build_xml_retry_hint(tool_output)
        sys.stderr.write(hint)
        sys.exit(2)

    json_tools = [
        "search_files",
        "find_symbol",
        "analyze_project",
        "extract_structure",
        "map_summarize_modules",
        "reduce_architecture_review",
        "analyze_architecture",
        "list_tasks",
        "get_task",
        "update_task_state",
        "get_verification_checklist",
        "get_resume_brief",
        "prepare_next_handoff",
        "read_handoff",
        "record_handoff",
        "delete_handoff",
        "delete_handoff_chain",
    ]
    if tool_name in json_tools:
        try:
            json.loads(tool_output)
        except json.JSONDecodeError:
            sys.stderr.write(
                f"Validointivirhe: Työkalu '{tool_name}' palautti viallista JSON-rakennetta. Korjaa tuloste."
            )
            sys.exit(2)

    sys.exit(0)


def _build_xml_retry_hint(output: str) -> str:
    if parse_xml_tool_call is not None:
        parsed = parse_xml_tool_call(output)
        if parsed is not None:
            return (
                "Validointivirhe: Tuloste sisälsi <tool_call>-XML:ää eikä oikeaa työkalutulosta. "
                f"Käytä MCP-työkalua {parsed.tool_name!r} suoralla JSON-kutsulla, "
                f"argumenteilla: {json.dumps(parsed.arguments, ensure_ascii=False)}. "
                "Älä kirjoita <tool_call>-taggeja tekstinä."
            )

    return (
        "Validointivirhe: Tuloste alkaa tai sisältää <tool_call>-XML:ää. "
        "Kutsu työkalua uudelleen MCP-rajapinnan kautta oikealla työkalunimellä ja JSON-parametreilla. "
        "Maksimi 2 uudelleenyritystä."
    )


if __name__ == "__main__":
    main()
