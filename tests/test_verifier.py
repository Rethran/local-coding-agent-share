"""Tests for independent deterministic subtask verification."""

from orchestrator.verification_spec import slice_token
from orchestrator.verifier import (
    is_collection_or_syntax_error,
    is_declared_verification_failure,
    is_declared_verification_success,
    run_verifier,
)


def test_declared_verification_requires_exact_successful_command() -> None:
    subtask = {"verification": ["pytest tests/ -k toggle_api -v"]}

    assert is_declared_verification_success(
        tool_name="run_command",
        arguments={"command": "pytest tests/ -k toggle_api -v"},
        status="ok",
        output="Exit code: 0\npassed",
        merged_subtask=subtask,
    )
    assert not is_declared_verification_success(
        tool_name="run_command",
        arguments={"command": "pytest tests/test_schedule_api.py -v"},
        status="ok",
        output="Exit code: 0\npassed",
        merged_subtask=subtask,
    )


def test_syntax_error_pytest_output_is_collection_error() -> None:
    output = (
        "Exit code: 2\n"
        "tests/test_python_extractor.py:5: SyntaxError: invalid syntax\n"
    )
    assert is_collection_or_syntax_error(output) is True


def test_assertion_failure_is_not_collection_error() -> None:
    output = "Exit code: 1\nAssertionError: expected one edge\n"
    assert is_collection_or_syntax_error(output) is False


def test_collection_error_is_not_test_writer_red() -> None:
    subtask = {"verification": ["python -m pytest tests/test_python_extractor.py -q"]}
    output = (
        "Exit code: 2\n"
        "collected 0 items / 1 error\n"
        "SyntaxError: invalid syntax\n"
    )
    assert is_declared_verification_failure(
        tool_name="run_command",
        arguments={"command": "python -m pytest tests/test_python_extractor.py -q"},
        status="ok",
        output=output,
        merged_subtask=subtask,
    ) is False


def test_assertion_failure_is_test_writer_red() -> None:
    subtask = {"verification": ["python -m pytest tests/test_python_extractor.py -q"]}
    assert is_declared_verification_failure(
        tool_name="run_command",
        arguments={"command": "python -m pytest tests/test_python_extractor.py -q"},
        status="ok",
        output="Exit code: 1\nAssertionError: expected one edge\n",
        merged_subtask=subtask,
    ) is True


def test_verifier_passes_all_declared_commands_sequentially() -> None:
    commands_run: list[str] = []

    def runner(command: str) -> str:
        commands_run.append(command)
        return "Exit code: 0\npassed"

    result = run_verifier(
        "HEAT-001",
        "HEAT-001.1",
        {"verification": ["pytest tests/a.py", "pytest tests/b.py"]},
        command_runner=runner,
    )

    assert result.passed is True
    assert commands_run == ["pytest tests/a.py", "pytest tests/b.py"]


def test_verifier_stops_on_first_failure() -> None:
    commands_run: list[str] = []

    def runner(command: str) -> str:
        commands_run.append(command)
        return "Exit code: 1\nfailed"

    result = run_verifier(
        "HEAT-001",
        "HEAT-001.1",
        {"verification": ["pytest tests/a.py", "pytest tests/b.py"]},
        command_runner=runner,
    )

    assert result.passed is False
    assert commands_run == ["pytest tests/a.py"]
    assert result.blockers == ("Verification failed: pytest tests/a.py",)


def test_verifier_rejects_subtask_without_automated_command() -> None:
    result = run_verifier("HEAT-001", "HEAT-001.1", {})

    assert result.passed is False
    assert result.evidence == ()


def test_verifier_ignores_prose_verification_entries() -> None:
    commands_run: list[str] = []

    def runner(command: str) -> str:
        commands_run.append(command)
        return "Exit code: 0\npassed"

    result = run_verifier(
        "HEAT-001",
        "HEAT-001.1",
        {
            "verification": [
                "python -m pytest tests/a.py -q",
                "Assert the slot toggles.",
            ]
        },
        command_runner=runner,
    )

    assert result.passed is True
    assert commands_run == ["python -m pytest tests/a.py -q"]


def test_verifier_fails_when_no_tests_collected() -> None:
    result = run_verifier(
        "HEAT-001",
        "HEAT-001.1",
        {"verification": ["pytest tests/a.py -k missing -q"]},
        command_runner=lambda _command: "Exit code: 0\ncollected 0 items",
    )

    assert result.passed is False
    assert result.blockers == ("Verification failed: pytest tests/a.py -k missing -q",)


def test_verifier_fails_when_declared_module_file_is_missing() -> None:
    result = run_verifier(
        "HEAT-001",
        "HEAT-001.1",
        {
            "targetArtifacts": {
                "module": "app/domain/__definitely_missing_verifier_gate__.py",
                "test": "tests/__definitely_missing_verifier_gate__.py",
            },
            "verification": ["pytest tests/__definitely_missing_verifier_gate__.py -q"],
        },
        command_runner=lambda _command: "Exit code: 0\npassed",
    )

    assert result.passed is False
    assert any("missing module file" in item for item in result.blockers)


def test_verifier_fails_when_declared_test_does_not_call_target_function(
    tmp_path,
    monkeypatch,
) -> None:
    module = tmp_path / "guard.py"
    test_file = tmp_path / "test_decide.py"
    module.write_text("def same_action():\n    return True\n", encoding="utf-8")
    test_file.write_text(
        "def test_CTX_047_1():\n    from rules import decide\n    assert decide() == 'none'\n",
        encoding="utf-8",
    )

    def fake_safe(relative: str):
        if relative.endswith("guard.py"):
            return module
        return test_file

    monkeypatch.setattr("orchestrator.verifier.safe_path", fake_safe)
    result = run_verifier(
        "CTX-047",
        "CTX-047.1",
        {
            "targetArtifacts": {
                "module": "src/context_mgmt/guard.py",
                "test": "tests/test_ctx_047_1_decide.py",
                "function": "same_action",
            },
            "verification": ["pytest tests/test_ctx_047_1_decide.py -q"],
        },
        command_runner=lambda _command: "Exit code: 0\npassed",
    )

    assert result.passed is False
    assert any("does not call same_action" in item for item in result.blockers)


def test_slice_token_replaces_hyphen_and_dot() -> None:
    assert slice_token("CTX-004.2") == "CTX_004_2"
