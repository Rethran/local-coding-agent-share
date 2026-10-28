"""Coordinator oracle + stub gate for already-green duplicate proof."""

from __future__ import annotations

import subprocess
from pathlib import Path

from orchestrator.contract_gate import (
    duplicate_coverage_blockers,
    module_name_from_path,
    oracle_blockers,
    split_example,
    stub_function_source,
)


MODULE_SOURCE = """
def default_config():
    return object()


def decide(usage, config):
    messages = usage.get("messages") or []
    keys = [item.get("dedupe_key") for item in messages]
    if len(keys) != len(set(keys)):
        return {"action": "trim", "reason": "duplicate_context", "tokens_saved_est": 10, "messages": messages[-1:]}
    return {"action": "none", "reason": "below_threshold", "tokens_saved_est": 0, "messages": messages}
"""

REAL_TEST = """
from rules import decide, default_config

def test_single_copy():
    result = decide({"context_pct": 0.1, "messages": [{"role": "tool", "content": "only", "dedupe_key": "file:a.py"}]}, default_config())
    assert result["action"] == "none"
"""

VACUOUS_TEST = """
def test_vacuous():
    assert True
"""


def _is_module_rel(rel: str) -> bool:
    normalized = rel.replace("\\", "/")
    return normalized == "rules.py" or normalized.endswith("/rules.py")


def _subtask(examples: list[dict], verification: str = "python -m pytest tests/test_rules.py -q") -> dict:
    return {
        "targetArtifacts": {
            "module": "rules.py",
            "test": "tests/test_rules.py",
            "function": "decide",
        },
        "verification": [verification],
        "dataExamples": examples,
    }


def _run(command: str, cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR:\n{result.stderr}"
    return f"Exit code: {result.returncode}\n{output}"


def test_module_name_strips_src_prefix() -> None:
    assert module_name_from_path("src/context_mgmt/rules.py") == "context_mgmt.rules"


def test_split_example_input_output() -> None:
    call, expected = split_example(
        {
            "input": {"context_pct": 0.1, "messages": []},
            "output": {"action": "none", "reason": "below_threshold"},
        },
        {"usage", "config"},
    )
    assert call == {"context_pct": 0.1, "messages": []}
    assert expected == {"action": "none", "reason": "below_threshold"}


def test_split_example_mixed_decide_shape() -> None:
    call, expected = split_example(
        {
            "context_pct": 0.1,
            "messages": [{"content": "only", "dedupe_key": "file:a.py"}],
            "action": "none",
            "reason": "below_threshold",
        },
        {"usage", "config"},
    )
    assert call["context_pct"] == 0.1
    assert expected["action"] == "none"
    assert "messages" not in expected
    assert "token_count" not in expected


def test_split_example_keeps_token_count_as_input() -> None:
    call, expected = split_example(
        {
            "token_count": 4001,
            "context_pct": 0.1,
            "marker": "\n[context-mgmt:trimmed]\n",
            "action": "trim",
            "reason": "tool_output_over_cap",
            "tokens_saved_est": 1,
        },
        {"usage", "config"},
    )
    assert call["token_count"] == 4001
    assert expected["action"] == "trim"
    assert expected["reason"] == "tool_output_over_cap"
    assert expected["tokens_saved_est"] == 1
    assert expected["marker"] == "\n[context-mgmt:trimmed]\n"


def test_split_example_rejects_invalid_only_list() -> None:
    assert split_example({"invalid": ["missing org_id"]}, {"payload", "salt"}) is None


def test_stub_replaces_function_body() -> None:
    source, found = stub_function_source(MODULE_SOURCE, "decide")
    assert found is True
    assert "NotImplementedError" in source
    assert "duplicate_context" not in source


def test_oracle_accepts_matching_decide(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-004.CTX-004.2",
        "rules.py",
        "decide",
        [
            {
                "context_pct": 0.1,
                "messages": [{"role": "tool", "content": "only", "dedupe_key": "file:a.py"}],
                "action": "none",
                "reason": "below_threshold",
            }
        ],
    )
    assert blockers == []


def test_oracle_rejects_wrong_action(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-004.CTX-004.2",
        "rules.py",
        "decide",
        [
            {
                "context_pct": 0.1,
                "messages": [
                    {"role": "tool", "content": "old", "dedupe_key": "file:a.py"},
                    {"role": "tool", "content": "new", "dedupe_key": "file:a.py"},
                ],
                "action": "none",
            }
        ],
    )
    assert any("action expected 'none'" in item for item in blockers)


def test_duplicate_rejects_vacuous_green_tests(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    test_path = tmp_path / "tests" / "test_rules.py"
    test_path.parent.mkdir()
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    test_path.write_text(VACUOUS_TEST, encoding="utf-8")
    original = module_path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        "orchestrator.contract_gate.safe_path",
        lambda rel: module_path if _is_module_rel(rel) else test_path,
    )
    blockers = duplicate_coverage_blockers(
        "CTX-004",
        "CTX-004.2",
        _subtask(
            [
                {
                    "context_pct": 0.1,
                    "messages": [{"role": "tool", "content": "only", "dedupe_key": "file:a.py"}],
                    "action": "none",
                    "reason": "below_threshold",
                }
            ]
        ),
        command_runner=lambda command: _run(command, tmp_path),
    )
    assert any("still pass when decide is stubbed" in item for item in blockers)
    assert module_path.read_text(encoding="utf-8") == original


def test_duplicate_accepts_oracle_and_stub_sensitive_tests(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    test_path = tmp_path / "tests" / "test_rules.py"
    test_path.parent.mkdir()
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    test_path.write_text(REAL_TEST, encoding="utf-8")
    monkeypatch.setattr(
        "orchestrator.contract_gate.safe_path",
        lambda rel: module_path if _is_module_rel(rel) else test_path,
    )
    blockers = duplicate_coverage_blockers(
        "CTX-004",
        "CTX-004.2",
        _subtask(
            [
                {
                    "context_pct": 0.1,
                    "messages": [{"role": "tool", "content": "only", "dedupe_key": "file:a.py"}],
                    "action": "none",
                    "reason": "below_threshold",
                }
            ]
        ),
        command_runner=lambda command: _run(command, tmp_path),
    )
    assert blockers == []
    assert "duplicate_context" in module_path.read_text(encoding="utf-8")


def test_oracle_rejects_non_executable_examples(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-005.CTX-005.2",
        "rules.py",
        "decide",
        [{"invalid": ["missing org_id", "action drop"]}],
    )
    assert any("not executable" in item for item in blockers)


def test_oracle_hash_material(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "events.py"
    module_path.write_text(
        "import hashlib\n"
        "def hash_user_id(user_id, salt):\n"
        "    return hashlib.sha256(f'{salt}:{user_id}'.encode()).hexdigest()\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-005.CTX-005.3",
        "events.py",
        "hash_user_id",
        [{"user_id": "alice", "salt": "test-salt", "algorithm": "sha256", "material": "test-salt:alice"}],
    )
    assert blockers == []


def test_oracle_loads_sibling_config(tmp_path: Path, monkeypatch) -> None:
    pkg = tmp_path / "src" / "context_mgmt"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "config.py").write_text(
        "class OrgConfig:\n    tool_output_token_cap = 4000\n"
        "def default_config():\n    return OrgConfig()\n",
        encoding="utf-8",
    )
    (pkg / "rules.py").write_text(
        "def decide(usage, config):\n"
        "    return {'action': 'none', 'reason': 'below_threshold', 'tokens_saved_est': 0}\n",
        encoding="utf-8",
    )
    module_path = pkg / "rules.py"
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-004.CTX-004.2",
        "src/context_mgmt/rules.py",
        "decide",
        [{"input": {"context_pct": 0.1, "messages": []}, "output": {"action": "none"}}],
    )
    assert blockers == []


GATEWAY_APP = """
def default_config():
    return object()


def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        upstream = client.post(
            "http://localhost/api/chat",
            json={"messages": payload.get("messages", [])},
        )
        body = upstream.json()
        if emit is not None:
            try:
                emit({"action": "none"})
            except Exception:
                pass
        return JSONResponse(status_code=upstream.status_code, content=body)

    return app
"""


def test_oracle_observes_gateway_emit_order(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-011.CTX-011.1",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [{"order": ["upstream", "emit"], "calls": 1}],
    )
    assert blockers == []


def test_oracle_rejects_gateway_emit_before_upstream(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-011.CTX-011.1",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [{"order": ["emit", "upstream"], "calls": 1}],
    )
    assert any("order expected" in item for item in blockers)


METADATA_EXAMPLE = {
    "user": "SECRET-PROMPT",
    "assistant": "SECRET-ANSWER",
    "forbidden_keys": ["content", "messages", "prompt"],
}


LEAKING_GATEWAY_APP = """
def default_config():
    return object()


def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        upstream = client.post(
            "http://localhost/api/chat",
            json={"messages": payload.get("messages", [])},
        )
        body = upstream.json()
        user = (payload.get("messages") or [{}])[-1].get("content")
        assistant = body["choices"][0]["message"]["content"]
        if emit is not None:
            emit({"action": "none", "content": assistant, "prompt": user})
        return JSONResponse(status_code=upstream.status_code, content=body)

    return app
"""


SILENT_GATEWAY_APP = """
def default_config():
    return object()


def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        upstream = client.post(
            "http://localhost/api/chat",
            json={"messages": payload.get("messages", [])},
        )
        return JSONResponse(status_code=upstream.status_code, content=upstream.json())

    return app
"""


def test_oracle_accepts_metadata_event_without_prompt_text(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-013.CTX-013.1",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [METADATA_EXAMPLE],
    )
    assert blockers == []


def test_oracle_rejects_prompt_text_in_emitted_event(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(LEAKING_GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-013.CTX-013.1",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [METADATA_EXAMPLE],
    )
    assert any("forbidden key 'content'" in item for item in blockers)
    assert any("user text" in item for item in blockers)
    assert any("assistant text" in item for item in blockers)
    assert not any("not executable" in item for item in blockers)


def test_oracle_requires_an_emitted_event_for_metadata(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(SILENT_GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-013.CTX-013.1",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [METADATA_EXAMPLE],
    )
    assert any("expected an emitted event" in item for item in blockers)


def test_oracle_accepts_gateway_emit_failure(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(GATEWAY_APP, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-011.CTX-011.2",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [{"emit": "raise RuntimeError", "status": 200, "content": "ok"}],
    )
    assert blockers == []


def test_oracle_folds_token_count_into_usage(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "rules.py"
    module_path.write_text(
        "def default_config():\n"
        "    return type('C', (), {'tool_output_token_cap': 4000})()\n"
        "def decide(usage, config):\n"
        "    messages = usage.get('messages') or []\n"
        "    if any((item.get('token_count') or 0) > config.tool_output_token_cap for item in messages):\n"
        "        return {'action': 'trim', 'reason': 'tool_output_over_cap', 'tokens_saved_est': 1}\n"
        "    return {'action': 'none', 'reason': 'below_threshold', 'tokens_saved_est': 0}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-003.CTX-003.1",
        "rules.py",
        "decide",
        [
            {
                "token_count": 4001,
                "context_pct": 0.1,
                "content_length": 500,
                "action": "trim",
                "reason": "tool_output_over_cap",
                "tokens_saved_est": 1,
            }
        ],
    )
    assert blockers == []


WRAPPER_SOURCE = '''
def make_check_context(config, emit, org_id, user_id, provider, model_id, salt, now):
    def check_context(used_tokens, window_size, last_tool_output_tokens):
        return decide({"context_pct": used_tokens / window_size}, config)
    return check_context


def decide(usage, config):
    return {"action": "compact_recommended"}
'''

WRAPPER_WITH_LITERAL = WRAPPER_SOURCE.replace(
    "return decide(",
    "threshold = 0.60\n        return decide(",
)

CTX_018_2_EXAMPLE = {
    "required": "decide(",
    "forbidden": ["compact_threshold", "0.60", "4000"],
}


def test_oracle_accepts_source_scan_example(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "mcp_server.py"
    module_path.write_text(WRAPPER_SOURCE, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-018.CTX-018.2",
        "src/context_mgmt/mcp_server.py",
        "make_check_context",
        [CTX_018_2_EXAMPLE],
    )
    assert blockers == []


def test_oracle_rejects_missing_required_substring(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "mcp_server.py"
    module_path.write_text(
        WRAPPER_SOURCE.replace("decide(", "choose("),
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-018.CTX-018.2",
        "src/context_mgmt/mcp_server.py",
        "make_check_context",
        [CTX_018_2_EXAMPLE],
    )
    assert any("missing required substring 'decide('" in item for item in blockers)


def test_oracle_rejects_forbidden_source_substring(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "mcp_server.py"
    module_path.write_text(WRAPPER_WITH_LITERAL, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-018.CTX-018.2",
        "src/context_mgmt/mcp_server.py",
        "make_check_context",
        [CTX_018_2_EXAMPLE],
    )
    assert any("forbidden substring '0.60'" in item for item in blockers)


def test_oracle_accepts_forbidden_only_source_example(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "validate_input.py"
    module_path.write_text(
        "def reject_unknown(payload, allowed):\n    return None\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-052.CTX-052.2",
        "src/context_mgmt/validate_input.py",
        "reject_unknown",
        [{"forbidden": ["pickle", "yaml.load"]}],
    )
    assert blockers == []


STANDIN_GATEWAY = '''
def default_config():
    return type("C", (), {"user_hash_salt": "test-salt"})()


def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        upstream = client.post("http://localhost/api/chat", json={"messages": payload.get("messages", [])})
        body = upstream.json()
        if emit is not None:
            usage = body.get("usage")
            emit({"token_source": "provider" if usage else "estimate", "action": "none"})
        return JSONResponse(status_code=upstream.status_code, content=body)

    return app
'''

ESTIMATE_GATEWAY = STANDIN_GATEWAY.replace(
    '"provider" if usage else "estimate"',
    '"estimate"',
)

CTX_020_2_EXAMPLE = {
    "token_source": "provider",
    "contents": ["cloud-ok", "ollama-ok"],
}


def test_oracle_accepts_two_standin_gateway_example(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(STANDIN_GATEWAY, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-020.CTX-020.2",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [CTX_020_2_EXAMPLE],
    )
    assert blockers == []


def test_oracle_rejects_standin_gateway_without_provider_source(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(ESTIMATE_GATEWAY, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-020.CTX-020.2",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [CTX_020_2_EXAMPLE],
    )
    assert any("stand-in 1 token_source expected 'provider'" in item for item in blockers)
    assert any("stand-in 2 token_source expected 'provider'" in item for item in blockers)


FAIL_OPEN_GATEWAY = '''
def default_config():
    return object()


def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        payload = await request.json()
        messages = payload.get("messages", [])
        try:
            if decide is not None:
                decide({}, config)
        except Exception:
            messages = payload.get("messages", [])
        client.post("http://localhost/api/chat", json={"messages": messages})
        return JSONResponse(status_code=200, content={"choices": [{"message": {"content": "ok"}}]})

    return app
'''

TRIMMING_GATEWAY = FAIL_OPEN_GATEWAY.replace(
    '''        try:
            if decide is not None:
                decide({}, config)
        except Exception:
            messages = payload.get("messages", [])
''',
    '''        messages = []
        for message in payload.get("messages", []):
            if message.get("role") == "tool" and (message.get("token_count") or 0) > 4000:
                message = {**message, "content": message["content"] + "\\n[context-mgmt:trimmed]\\n"}
            messages.append(message)
''',
)

CTX_021_2_EXAMPLE = {
    "tool_content": "ORIGINAL",
    "token_count": 4001,
    "status": 200,
}


def test_oracle_accepts_decide_failure_forward(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(FAIL_OPEN_GATEWAY, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-021.CTX-021.2",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [CTX_021_2_EXAMPLE],
    )
    assert blockers == []


def test_oracle_rejects_trim_when_decide_should_fail_open(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "gateway.py"
    module_path.write_text(TRIMMING_GATEWAY, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-021.CTX-021.2",
        "src/context_mgmt/gateway.py",
        "create_gateway",
        [CTX_021_2_EXAMPLE],
    )
    assert any("forwarded tool content expected 'ORIGINAL'" in item for item in blockers)


OBSERVER_OK = '''
def run_observer(observer, events):
    try:
        return observer.analyze(events)
    except Exception:
        return {"guidance": ""}
'''

OBSERVER_LEAK = '''
def run_observer(observer, events):
    return observer.analyze(events)
'''

GATEWAY_OK = '''
def create_gateway(client, config, decide=None, emit=None, salt=None, org_id="org-a", user_id="alice"):
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat():
        upstream = client.post("http://localhost/api/chat", json={})
        return JSONResponse(status_code=upstream.status_code, content=upstream.json())

    return app
'''

GATEWAY_500 = GATEWAY_OK.replace("status_code=upstream.status_code", "status_code=500")

CTX_021_3_EXAMPLE = {"guidance": "", "gateway_status": 200}


def test_oracle_accepts_dead_observer_and_live_gateway(tmp_path: Path, monkeypatch) -> None:
    observer_path = tmp_path / "observer.py"
    observer_path.write_text(OBSERVER_OK, encoding="utf-8")
    (tmp_path / "gateway.py").write_text(GATEWAY_OK, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: observer_path)
    blockers = oracle_blockers(
        "CTX-021.CTX-021.3",
        "src/context_mgmt/observer.py",
        "run_observer",
        [CTX_021_3_EXAMPLE],
    )
    assert blockers == []


def test_oracle_rejects_observer_that_lets_the_crash_escape(tmp_path: Path, monkeypatch) -> None:
    observer_path = tmp_path / "observer.py"
    observer_path.write_text(OBSERVER_LEAK, encoding="utf-8")
    (tmp_path / "gateway.py").write_text(GATEWAY_OK, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: observer_path)
    blockers = oracle_blockers(
        "CTX-021.CTX-021.3",
        "src/context_mgmt/observer.py",
        "run_observer",
        [CTX_021_3_EXAMPLE],
    )
    assert any("run_observer raised RuntimeError" in item for item in blockers)


def test_oracle_rejects_gateway_status_beside_a_dead_observer(tmp_path: Path, monkeypatch) -> None:
    observer_path = tmp_path / "observer.py"
    observer_path.write_text(OBSERVER_OK, encoding="utf-8")
    (tmp_path / "gateway.py").write_text(GATEWAY_500, encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: observer_path)
    blockers = oracle_blockers(
        "CTX-021.CTX-021.3",
        "src/context_mgmt/observer.py",
        "run_observer",
        [CTX_021_3_EXAMPLE],
    )
    assert any("gateway_status expected 200" in item for item in blockers)


PLANE_MODULE = """
def feature_decision(features, component, plane):
    message = "Access to this service is not available. Contact your administrator."
    if component in features:
        return {"allowed": True, "code": "ok", "message": "ok"}
    if plane == "management":
        return {"allowed": False, "code": "feature_not_licensed", "message": message}
    return {"allowed": False, "code": "access_denied", "message": message}
"""

PLANE_EXAMPLE = {
    "component": "mcp",
    "data_code": "access_denied",
    "management_code": "feature_not_licensed",
    "message": "Access to this service is not available. Contact your administrator.",
}

PLANE_TEST = """
from license import feature_decision

def test_deny_CTX_028_2():
    data = feature_decision(["gateway"], "mcp", "data")
    management = feature_decision(["gateway"], "mcp", "management")
    assert data["code"] == "access_denied"
    assert management["code"] == "feature_not_licensed"
    assert "expired" not in data["message"].lower()
"""


def _plane_paths(rel: str, module_path: Path, test_path: Path) -> Path:
    normalized = rel.replace("\\", "/")
    if normalized.endswith("license.py"):
        return module_path
    return test_path


def test_unknown_example_shape_accepts_stub_sensitive_tests(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "license.py"
    test_path = tmp_path / "tests" / "test_deny.py"
    test_path.parent.mkdir()
    module_path.write_text(PLANE_MODULE, encoding="utf-8")
    test_path.write_text(PLANE_TEST, encoding="utf-8")
    monkeypatch.setattr(
        "orchestrator.contract_gate.safe_path",
        lambda rel: _plane_paths(rel, module_path, test_path),
    )
    blockers = duplicate_coverage_blockers(
        "CTX-028",
        "CTX-028.2",
        {
            "targetArtifacts": {
                "module": "license.py",
                "test": "tests/test_deny.py",
                "function": "feature_decision",
            },
            "verification": ["python -m pytest tests/test_deny.py -q"],
            "dataExamples": [PLANE_EXAMPLE],
        },
        command_runner=lambda command: _run(command, tmp_path),
    )
    assert blockers == []


def test_unknown_example_shape_still_rejects_vacuous_tests(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "license.py"
    test_path = tmp_path / "tests" / "test_deny.py"
    test_path.parent.mkdir()
    module_path.write_text(PLANE_MODULE, encoding="utf-8")
    test_path.write_text("def test_vacuous():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(
        "orchestrator.contract_gate.safe_path",
        lambda rel: _plane_paths(rel, module_path, test_path),
    )
    blockers = duplicate_coverage_blockers(
        "CTX-028",
        "CTX-028.2",
        {
            "targetArtifacts": {
                "module": "license.py",
                "test": "tests/test_deny.py",
                "function": "feature_decision",
            },
            "verification": ["python -m pytest tests/test_deny.py -q"],
            "dataExamples": [PLANE_EXAMPLE],
        },
        command_runner=lambda command: _run(command, tmp_path),
    )
    assert any("still pass when feature_decision is stubbed" in item for item in blockers)
    assert not any("not executable" in item for item in blockers)


def test_output_dict_fields_are_assertions(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "models.py"
    module_path.write_text(
        "def models_response(model_ids, enabled):\n"
        "    if not enabled:\n"
        "        return {'status': 404, 'code': 'not_found', 'message': 'Not found.'}\n"
        "    return {'status': 200, 'object': 'list', 'data': [{'id': model_ids[0], 'object': 'model'}]}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-070.CTX-070.1",
        "models.py",
        "models_response",
        [{
            "input": {"model_ids": ["qwen3:8b"], "enabled": True},
            "output": {"status": 200, "object": "list", "data": [{"id": "qwen3:8b", "object": "model"}]},
        }],
    )
    assert blockers == []


def test_flat_example_compares_scalar_return(tmp_path: Path, monkeypatch) -> None:
    module_path = tmp_path / "judge.py"
    module_path.write_text(
        "def quality_ok(before, after, margin):\n"
        "    return (before - after) <= margin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: module_path)
    blockers = oracle_blockers(
        "CTX-069.CTX-069.1",
        "judge.py",
        "quality_ok",
        [{"before": 10, "after": 9, "margin": 2, "ok": True}],
    )
    assert blockers == []


def test_static_page_example_checks_required_text(tmp_path: Path, monkeypatch) -> None:
    page = tmp_path / "index.html"
    page.write_text('<div id="license-banner">The license needs attention. Contact your administrator.</div>', encoding="utf-8")
    monkeypatch.setattr("orchestrator.contract_gate.safe_path", lambda _rel: page)
    blockers = duplicate_coverage_blockers(
        "CTX-032",
        "CTX-032.3",
        {
            "targetArtifacts": {"module": "static/index.html", "test": "tests/test_banner.py"},
            "dataExamples": [{
                "required": ["license-banner", "The license needs attention. Contact your administrator."],
            }],
        },
        command_runner=lambda _command: "Exit code: 0\n",
    )
    assert blockers == []
