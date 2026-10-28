from orchestrator.continuity import infer_tool_next_action
from orchestrator.prompts import system_prompt
from orchestrator.python_write_gate import (
    SYNTAX_WRITE_RETRY_HINT,
    is_python_syntax_write_error,
    refuse_invalid_python_write,
)
from orchestrator.tool_compact import execute_tool

INDENTED_COUNT_ITEMS = (
    "    def count_items(items):\n"
    "        n = len(items)\n"
    "        connect()\n"
    "        return n\n"
)

COLUMN_ZERO_COUNT_ITEMS = (
    "def count_items(items):\n"
    "    n = len(items)\n"
    "    connect()\n"
    "    return n\n"
)


def test_refuse_invalid_python_write_rejects_missing_def() -> None:
    refusal = refuse_invalid_python_write(
        "tests/test_python_extractor.py",
        "test_extract_python_edges_simple_library_py():\n    assert True\n",
    )
    assert refusal is not None
    assert "Python syntax invalid" in refusal
    assert 'File "tests/test_python_extractor.py", line 1' in refusal
    assert "test_extract_python_edges_simple_library_py():" in refusal
    assert "^" in refusal
    assert "SyntaxError: invalid syntax" in refusal


def test_refuse_invalid_python_write_rejects_indented_def_with_caret() -> None:
    refusal = refuse_invalid_python_write(
        "tests/fixtures/count_items.py",
        INDENTED_COUNT_ITEMS,
    )
    assert refusal is not None
    assert 'File "tests/fixtures/count_items.py", line 1' in refusal
    assert "    def count_items(items):" in refusal
    assert "^" in refusal
    assert "IndentationError: unexpected indent" in refusal
    assert is_python_syntax_write_error(refusal)


def test_refuse_invalid_python_write_allows_column_zero_def_with_connect() -> None:
    assert (
        refuse_invalid_python_write(
            "tests/fixtures/count_items.py",
            COLUMN_ZERO_COUNT_ITEMS,
        )
        is None
    )


def test_refuse_invalid_python_write_allows_valid_module() -> None:
    assert (
        refuse_invalid_python_write(
            "src/living_graph/python_extractor.py",
            "def extract_python_edges(path):\n    return []\n",
        )
        is None
    )


def test_refuse_invalid_python_write_ignores_non_python() -> None:
    assert (
        refuse_invalid_python_write("tests/fixtures/simple_test.robot", "*** Test Cases ***\n")
        is None
    )


def test_refuse_invalid_python_write_reports_unclosed_paren() -> None:
    refusal = refuse_invalid_python_write("src/oops.py", "def oops(\n")
    assert refusal is not None
    assert "def oops(" in refusal
    assert "^" in refusal
    assert "SyntaxError:" in refusal
    assert "unexpected indent" not in refusal


def test_refuse_invalid_python_write_reports_expected_indented_block() -> None:
    refusal = refuse_invalid_python_write("src/foo.py", "def foo():\npass\n")
    assert refusal is not None
    assert "pass" in refusal
    assert "expected an indented block" in refusal
    assert "must start at column 0" not in refusal


def test_execute_tool_write_file_rejects_invalid_python() -> None:
    status, result = execute_tool(
        "write_file",
        {"path": "__invalid_syntax_gate__.py", "content": "def oops(\n"},
    )
    assert status == "error"
    assert "Python syntax invalid" in result
    assert "def oops(" in result
    assert "File \"__invalid_syntax_gate__.py\"" in result


def test_infer_tool_next_action_retries_same_write_on_syntax_error() -> None:
    diagnostic = refuse_invalid_python_write(
        "tests/fixtures/count_items.py",
        INDENTED_COUNT_ITEMS,
    )
    assert diagnostic is not None
    action = infer_tool_next_action(
        "write_file",
        {"path": "tests/fixtures/count_items.py", "content": INDENTED_COUNT_ITEMS},
        "error",
        f"ERROR executing write_file: {diagnostic}",
    )
    assert action == {
        "tool": "write_file",
        "hint": SYNTAX_WRITE_RETRY_HINT,
        "path": "tests/fixtures/count_items.py",
    }


def test_infer_tool_next_action_retries_replace_in_file_on_syntax_error() -> None:
    diagnostic = refuse_invalid_python_write("src/foo.py", "def foo():\npass\n")
    assert diagnostic is not None
    action = infer_tool_next_action(
        "replace_in_file",
        {"path": "src/foo.py", "old_text": "x", "new_text": "def foo():\npass\n"},
        "error",
        diagnostic,
    )
    assert action is not None
    assert action["tool"] == "replace_in_file"
    assert "choose another approach" not in action["hint"]
    assert action["hint"] == SYNTAX_WRITE_RETRY_HINT


def test_system_prompt_requires_retry_on_python_syntax_error() -> None:
    prompt = system_prompt("GRAPH-007.1")
    assert "unless the result status is error" in prompt
    assert "Python syntax invalid" in prompt
    assert "MUST call nextAction.tool" in prompt
    assert "do not switch tools" in prompt
