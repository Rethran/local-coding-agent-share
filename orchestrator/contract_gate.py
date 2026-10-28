"""Prove an already-green subtask is a real duplicate, not a vacuous pytest pass.

Duplicate proof:
1. When ``dataExamples`` can be executed, the oracle must match.
2. The declared pytest command must fail when ``targetArtifacts.function`` raises NotImplementedError.

An example shape the oracle cannot call does not veto the subtask. The worker cannot edit the spec, and the stub gate is the check it can satisfy.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable

from tools._common import WORKSPACE_ROOT, safe_path

from orchestrator.read_guard import normalize_relative_path
from orchestrator.tool_adapters import safe_run_command
from orchestrator.verifier import is_verification_green, verification_commands

USAGE_FIELDS = frozenset({"context_pct", "messages"})
MESSAGE_FIELDS = frozenset({"token_count", "content", "role", "dedupe_key"})
INPUT_FIELDS = USAGE_FIELDS | MESSAGE_FIELDS | frozenset({
    "payload",
    "user_id",
    "salt",
    "content_length",
})
EXPECT_FIELDS = frozenset({
    "action",
    "reason",
    "tokens_saved_est",
    "stored_keys",
    "raises",
    "algorithm",
    "material",
    "output",
    "expected",
    "value",
})
SPECIAL_EXPECT = frozenset({
    "raises",
    "invalid",
    "algorithm",
    "material",
    "stored_keys",
    "value",
})
GATEWAY_OBSERVE = frozenset({
    "order",
    "calls",
    "status",
    "content",
    "content_type",
    "action",
})


def module_name_from_path(relative: str) -> str:
    rel = normalize_relative_path(relative)
    if rel.startswith("src/"):
        rel = rel[4:]
    if rel.endswith(".py"):
        rel = rel[:-3]
    return rel.replace("/", ".")


def split_example(
    example: dict[str, Any],
    param_names: set[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Split a dataExample into call kwargs and expected assertions."""
    if not isinstance(example, dict):
        return None
    if "invalid" in example and not any(key in example for key in ("input", "payload", "raises")):
        return None
    if "input" in example:
        raw_input = example.get("input")
        call = dict(raw_input) if isinstance(raw_input, dict) else {}
        raw_expected = example.get("output")
        if raw_expected is None:
            raw_expected = example.get("expected")
        if isinstance(raw_expected, dict):
            expected = dict(raw_expected)
        elif raw_expected is not None:
            expected = {"value": raw_expected}
        else:
            expected = {
                key: value
                for key, value in example.items()
                if key not in {"input", "output", "expected"} and key in EXPECT_FIELDS
            }
        return call, expected

    call: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    for key, value in example.items():
        if key in param_names or key in INPUT_FIELDS:
            call[key] = value
        else:
            expected[key] = value
    if not call or not expected:
        return None
    return call, expected


def stub_function_source(source: str, function: str) -> tuple[str, bool]:
    tree = ast.parse(source)

    class _Stub(ast.NodeTransformer):
        def __init__(self) -> None:
            self.found = False

        def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
            return node

        def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
            if node.name != function or self.found:
                return node
            self.found = True
            node.body = [
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                        args=[ast.Constant(value="contract stub")],
                        keywords=[],
                    ),
                    cause=None,
                )
            ]
            return node

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
            if node.name != function or self.found:
                return node
            self.found = True
            node.body = [
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="NotImplementedError", ctx=ast.Load()),
                        args=[ast.Constant(value="contract stub")],
                        keywords=[],
                    ),
                    cause=None,
                )
            ]
            return node

    transformer = _Stub()
    stubbed = transformer.visit(tree)
    ast.fix_missing_locations(stubbed)
    return ast.unparse(stubbed) + "\n", transformer.found


def _static_document_blockers(
    label: str,
    module_rel: str,
    data_examples: Any,
) -> list[str]:
    """Check required and forbidden text in a non-Python target such as HTML."""
    examples = [item for item in data_examples if isinstance(item, dict)] if isinstance(data_examples, list) else []
    blockers: list[str] = []
    ran = 0
    for index, example in enumerate(examples, start=1):
        if not _looks_like_source_example(example, set()):
            continue
        ran += 1
        blockers.extend(_source_example_blockers(label, index, module_rel, example))
    if ran == 0:
        return [f"{label} dataExamples are not executable by the coordinator oracle."]
    return blockers


def _unexecutable_oracle_only(blockers: list[str]) -> bool:
    """True when every oracle blocker is an example shape it cannot call."""
    if not blockers:
        return False
    return all("not executable by the coordinator oracle" in item for item in blockers)


def duplicate_coverage_blockers(
    task_id: str,
    subtask_id: str,
    merged_subtask: dict[str, Any],
    *,
    command_runner: Callable[[str], str] = safe_run_command,
) -> list[str]:
    """Return blockers that prevent marking an already-green subtask as duplicate."""
    label = f"{task_id}.{subtask_id}"
    target = merged_subtask.get("targetArtifacts") or {}
    if not isinstance(target, dict):
        return [f"{label} missing targetArtifacts for duplicate proof."]
    module_rel = str(target.get("module") or "").strip()
    function = str(target.get("function") or "").strip()
    if module_rel and not module_rel.endswith(".py"):
        return _static_document_blockers(
            label,
            module_rel,
            merged_subtask.get("dataExamples") or [],
        )
    if not module_rel or not function:
        return [
            f"{label} missing targetArtifacts.module/function for duplicate proof."
        ]

    oracle = oracle_blockers(
        label, module_rel, function, merged_subtask.get("dataExamples") or []
    )
    stub = stub_verification_blockers(
        label,
        module_rel,
        function,
        merged_subtask,
        command_runner=command_runner,
    )
    if _unexecutable_oracle_only(oracle):
        return stub
    return oracle + stub


def _is_gateway_factory(fn: Callable[..., Any]) -> bool:
    """True when the target builds an app from an injected client and emit hook."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return {"client", "config", "emit"} <= set(params)


def _source_literals(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    return []


def _looks_like_source_example(example: dict[str, Any], param_names: set[str]) -> bool:
    """True when the example checks module source text, not a function return.

    CTX-018.2 and the same shape elsewhere use ``required`` / ``forbidden``
    substrings. Those keys are not arguments of the target function.
    """
    literals = _source_literals(example.get("required")) + _source_literals(example.get("forbidden"))
    if not literals:
        return False
    for key in example:
        if key in {"required", "forbidden"}:
            continue
        if key in param_names or key in INPUT_FIELDS or key in EXPECT_FIELDS:
            return False
        return False
    return True


def _source_example_blockers(
    label: str,
    index: int,
    module_rel: str,
    example: dict[str, Any],
) -> list[str]:
    prefix = f"{label} example {index}"
    path = safe_path(normalize_relative_path(module_rel))
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{prefix} cannot read {module_rel}: {exc}"]
    blockers: list[str] = []
    for literal in _source_literals(example.get("required")):
        if literal not in source:
            blockers.append(f"{prefix} module source is missing required substring {literal!r}.")
    for literal in _source_literals(example.get("forbidden")):
        if literal in source:
            blockers.append(f"{prefix} module source contains forbidden substring {literal!r}.")
    return blockers


def _looks_like_gateway_example(example: dict[str, Any]) -> bool:
    """Packet examples that describe one chat request, not a direct function return."""
    if any(key in example for key in ("order", "calls", "content_type", "stream", "forbidden_keys")):
        return True
    if isinstance(example.get("contents"), list) and example["contents"]:
        return True
    if isinstance(example.get("tool_content"), str) and "status" in example:
        return True
    if isinstance(example.get("user"), str) and isinstance(example.get("assistant"), str):
        return True
    return "status" in example and ("content" in example or "emit" in example)


def _gateway_example_blockers(
    label: str,
    index: int,
    fn: Callable[..., Any],
    module: Any,
    example: dict[str, Any],
) -> list[str]:
    prefix = f"{label} example {index}"
    contents = example.get("contents")
    if isinstance(contents, list) and contents:
        return _standin_gateway_blockers(prefix, fn, module, example, [str(item) for item in contents])
    if isinstance(example.get("tool_content"), str):
        return _fail_open_blockers(prefix, fn, module, example)
    try:
        observed = _observe_gateway(fn, module, example)
    except Exception as exc:
        return [f"{prefix} gateway oracle failed: {type(exc).__name__}: {exc}"]
    blockers: list[str] = []
    for key in sorted(GATEWAY_OBSERVE & set(example)):
        want = example[key]
        got = observed.get(key)
        if key == "content_type":
            actual = str(got or "")
            expected = str(want)
            if actual != expected and not actual.startswith(expected):
                blockers.append(f"{prefix} content_type expected {want!r}, got {got!r}.")
            continue
        if got != want:
            blockers.append(f"{prefix} {key} expected {want!r}, got {got!r}.")
    blockers.extend(_emitted_text_blockers(prefix, example, observed))
    return blockers


def _standin_gateway_blockers(
    prefix: str,
    fn: Callable[..., Any],
    module: Any,
    example: dict[str, Any],
    contents: list[str],
) -> list[str]:
    """POST once per assistant text and check status, content, and token_source."""
    try:
        observed_rows = _observe_gateway_contents(fn, module, contents)
    except Exception as exc:
        return [f"{prefix} gateway oracle failed: {type(exc).__name__}: {exc}"]
    blockers: list[str] = []
    want_source = example.get("token_source")
    for index, (want, observed) in enumerate(zip(contents, observed_rows), start=1):
        if observed["status"] != 200:
            blockers.append(
                f"{prefix} stand-in {index} status expected 200, got {observed['status']}."
            )
        if observed["content"] != want:
            blockers.append(
                f"{prefix} stand-in {index} content expected {want!r}, got {observed['content']!r}."
            )
        if want_source is not None and observed.get("token_source") != want_source:
            blockers.append(
                f"{prefix} stand-in {index} token_source expected {want_source!r}, "
                f"got {observed.get('token_source')!r}."
            )
    return blockers


def _observe_gateway_contents(
    fn: Callable[..., Any],
    module: Any,
    contents: list[str],
) -> list[dict[str, Any]]:
    """One gateway POST per expected assistant string, with provider usage."""
    import httpx
    from fastapi.testclient import TestClient

    config = _default_config(module)
    if config is None:
        raise TypeError("cannot construct the function's config argument")
    rows: list[dict[str, Any]] = []
    for assistant in contents:
        events: list[dict[str, Any]] = []
        upstream_body = {
            "choices": [{"message": {"role": "assistant", "content": assistant}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        def handler(_request: httpx.Request, body: dict[str, Any] = upstream_body) -> httpx.Response:
            return httpx.Response(200, json=body)

        def emit(event: dict[str, Any], bucket: list[dict[str, Any]] = events) -> None:
            bucket.append(event)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        app = fn(client=client, config=config, emit=emit)
        payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        with TestClient(app) as test_client:
            response = test_client.post("/v1/chat/completions", json=payload)
        content: Any = response.text
        try:
            content = response.json()["choices"][0]["message"]["content"]
        except Exception:
            content = response.text
        rows.append(
            {
                "status": response.status_code,
                "content": content,
                "token_source": events[0].get("token_source") if events else None,
            }
        )
    return rows


_TRIM_MARKER = "\n[context-mgmt:trimmed]\n"


def _fail_open_blockers(
    prefix: str,
    fn: Callable[..., Any],
    module: Any,
    example: dict[str, Any],
) -> list[str]:
    """POST one oversized tool message while decide raises, and check the forward."""
    try:
        observed = _observe_decide_failure(fn, module, example)
    except Exception as exc:
        return [f"{prefix} gateway oracle failed: {type(exc).__name__}: {exc}"]
    blockers: list[str] = []
    want_status = example.get("status")
    if want_status is not None and observed["status"] != want_status:
        blockers.append(f"{prefix} status expected {want_status!r}, got {observed['status']!r}.")
    want_content = str(example.get("tool_content"))
    forwarded = observed.get("forwarded_tool_content")
    if forwarded != want_content:
        blockers.append(
            f"{prefix} forwarded tool content expected {want_content!r}, got {forwarded!r}."
        )
    if isinstance(forwarded, str) and _TRIM_MARKER in forwarded:
        blockers.append(f"{prefix} forwarded tool content contains the trim marker.")
    return blockers


def _observe_decide_failure(
    fn: Callable[..., Any],
    module: Any,
    example: dict[str, Any],
) -> dict[str, Any]:
    """Call the gateway with a decide that raises and record the upstream tool content."""
    import httpx
    from fastapi.testclient import TestClient

    config = _default_config(module)
    if config is None:
        raise TypeError("cannot construct the function's config argument")
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["json"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
        )

    def decide(_usage: Any, _config: Any) -> dict[str, Any]:
        raise RuntimeError("rule engine down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    app = fn(client=client, config=config, decide=decide, emit=None)
    tool_message = {
        "role": "tool",
        "content": str(example["tool_content"]),
        "token_count": example.get("token_count", 4001),
    }
    payload = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}, tool_message],
    }
    with TestClient(app) as test_client:
        response = test_client.post("/v1/chat/completions", json=payload)
    forwarded = None
    body = captured.get("json") or {}
    for message in body.get("messages") or []:
        if isinstance(message, dict) and message.get("role") == "tool":
            forwarded = message.get("content")
            break
    return {"status": response.status_code, "forwarded_tool_content": forwarded}


def _observe_gateway(fn: Callable[..., Any], module: Any, example: dict[str, Any]) -> dict[str, Any]:
    """POST one chat completion and record upstream/emit order plus the response."""
    import httpx
    from fastapi.testclient import TestClient

    order: list[str] = []
    events: list[dict[str, Any]] = []
    if isinstance(example.get("assistant"), str):
        assistant = example["assistant"]
    elif isinstance(example.get("content"), str):
        assistant = example["content"]
    else:
        assistant = "ok"
    user_text = example["user"] if isinstance(example.get("user"), str) else "hi"
    upstream_body = {
        "choices": [{"message": {"role": "assistant", "content": assistant}}],
    }
    stream = bool(example.get("stream"))
    raw_body = example.get("body") if stream and isinstance(example.get("body"), str) else None
    media_type = str(example.get("content_type") or "text/event-stream")

    def handler(_request: httpx.Request) -> httpx.Response:
        order.append("upstream")
        if raw_body is not None:
            return httpx.Response(
                200,
                content=raw_body.encode("utf-8"),
                headers={"content-type": media_type},
            )
        return httpx.Response(200, json=upstream_body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    raise_emit = str(example.get("emit") or "").lower().startswith("raise")

    def emit(event: dict[str, Any]) -> None:
        order.append("emit")
        events.append(event)
        if raise_emit:
            raise RuntimeError("c3 down")

    config = _default_config(module)
    if config is None:
        raise TypeError("cannot construct the function's config argument")
    app = fn(client=client, config=config, emit=emit)
    payload: dict[str, Any] = {
        "model": "m",
        "messages": [{"role": "user", "content": user_text}],
    }
    if stream:
        payload["stream"] = True
    with TestClient(app) as test_client:
        response = test_client.post("/v1/chat/completions", json=payload)
    content: Any = response.text
    try:
        parsed = response.json()
        content = parsed["choices"][0]["message"]["content"]
    except Exception:
        content = response.text
    return {
        "order": order,
        "calls": order.count("emit"),
        "status": response.status_code,
        "content": content,
        "content_type": response.headers.get("content-type", ""),
        "action": events[0].get("action") if events else None,
        "event": events[0] if events else None,
    }


def _emitted_text_blockers(
    prefix: str,
    example: dict[str, Any],
    observed: dict[str, Any],
) -> list[str]:
    """Reject prompt or completion text on the emitted event."""
    checks_text = (
        "forbidden_keys" in example
        or isinstance(example.get("user"), str)
        or isinstance(example.get("assistant"), str)
    )
    if not checks_text:
        return []
    event = observed.get("event")
    if not isinstance(event, dict):
        return [f"{prefix} expected an emitted event to check prompt text."]
    blockers: list[str] = []
    forbidden = example.get("forbidden_keys")
    if isinstance(forbidden, list):
        for key in forbidden:
            if str(key) in event:
                blockers.append(f"{prefix} emitted event contains forbidden key {key!r}.")
    serialized = json.dumps(event)
    for field in ("user", "assistant"):
        secret = example.get(field)
        if isinstance(secret, str) and secret and secret in serialized:
            blockers.append(f"{prefix} emitted event contains {field} text.")
    return blockers


def _looks_like_observer_kill_example(example: dict[str, Any], function: str) -> bool:
    """True when the example is a dead observer plus a separate gateway status."""
    return (
        function == "run_observer"
        and "guidance" in example
        and "gateway_status" in example
    )


def _observer_kill_blockers(
    label: str,
    index: int,
    fn: Callable[..., Any],
    module: Any,
    example: dict[str, Any],
) -> list[str]:
    """Call run_observer with a raising observer, then POST the sibling gateway."""
    prefix = f"{label} example {index}"
    blockers: list[str] = []

    class _RaisingObserver:
        def analyze(self, _events: list[Any]) -> dict[str, Any]:
            raise RuntimeError("observer crashed")

    try:
        result = fn(_RaisingObserver(), [{"org_id": "org-a"}])
    except Exception as exc:
        return [f"{prefix} run_observer raised {type(exc).__name__}: {exc}"]
    guidance = result.get("guidance") if isinstance(result, dict) else None
    if guidance != example.get("guidance"):
        blockers.append(
            f"{prefix} guidance expected {example.get('guidance')!r}, got {guidance!r}."
        )

    try:
        status = _sibling_gateway_status(module)
    except Exception as exc:
        return blockers + [f"{prefix} gateway oracle failed: {type(exc).__name__}: {exc}"]
    want_status = example.get("gateway_status")
    if status != want_status:
        blockers.append(f"{prefix} gateway_status expected {want_status!r}, got {status!r}.")
    return blockers


def _sibling_gateway_status(observer_module: Any) -> int:
    """POST /v1/chat/completions on create_gateway next to the observer module."""
    from fastapi.testclient import TestClient

    file = getattr(observer_module, "__file__", None)
    if not file:
        raise FileNotFoundError("observer module has no file")
    gateway_path = Path(file).with_name("gateway.py")
    if not gateway_path.is_file():
        raise FileNotFoundError(str(gateway_path))
    _ensure_import_paths()
    gateway_module = _load_by_file(gateway_path)
    create_gateway = getattr(gateway_module, "create_gateway", None)
    if not callable(create_gateway):
        raise AttributeError("gateway.py has no create_gateway")

    class _Upstream:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}

    class _Client:
        def post(self, _url: str, json: Any = None) -> _Upstream:
            return _Upstream()

    config = _default_config(gateway_module)
    if config is None:
        config = object()
    app = create_gateway(client=_Client(), config=config)
    with TestClient(app) as test_client:
        response = test_client.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
    return response.status_code


def oracle_blockers(
    label: str,
    module_rel: str,
    function: str,
    data_examples: Any,
) -> list[str]:
    examples = [item for item in data_examples if isinstance(item, dict)] if isinstance(data_examples, list) else []
    if not examples:
        return [f"{label} has no dataExamples for the coordinator oracle."]

    try:
        module, fn = load_target_function(module_rel, function)
    except Exception as exc:
        return [f"{label} cannot load {function} from {module_rel}: {exc}"]

    param_names = set(inspect.signature(fn).parameters)
    gateway = _is_gateway_factory(fn)
    ran = 0
    blockers: list[str] = []
    for index, example in enumerate(examples, start=1):
        if gateway and _looks_like_gateway_example(example):
            ran += 1
            blockers.extend(_gateway_example_blockers(label, index, fn, module, example))
            continue
        if _looks_like_source_example(example, param_names):
            ran += 1
            blockers.extend(_source_example_blockers(label, index, module_rel, example))
            continue
        if _looks_like_observer_kill_example(example, function):
            ran += 1
            blockers.extend(_observer_kill_blockers(label, index, fn, module, example))
            continue
        split = split_example(example, param_names)
        if split is None:
            continue
        call, expected = split
        if not _has_assertions(expected):
            continue
        ran += 1
        try:
            kwargs = bind_call_kwargs(fn, call, module)
        except Exception as exc:
            if "missing required arguments" in str(exc):
                continue
            blockers.append(f"{label} example {index} cannot bind arguments: {exc}")
            continue
        try:
            result = fn(**kwargs)
            raised: BaseException | None = None
        except Exception as exc:
            result = None
            raised = exc
        blockers.extend(
            _compare_example(label, index, result, expected, raised)
        )
    if ran == 0:
        return [f"{label} dataExamples are not executable by the coordinator oracle."]
    return blockers


def stub_verification_blockers(
    label: str,
    module_rel: str,
    function: str,
    merged_subtask: dict[str, Any],
    *,
    command_runner: Callable[[str], str],
) -> list[str]:
    commands = verification_commands(merged_subtask)
    if not commands:
        return [f"{label} declares no verification command for the stub gate."]
    module_path = safe_path(normalize_relative_path(module_rel))
    if not module_path.is_file():
        return [f"{label} missing module file: {normalize_relative_path(module_rel)}"]
    original = module_path.read_bytes()
    try:
        stubbed, found = stub_function_source(original.decode("utf-8"), function)
    except SyntaxError as exc:
        return [f"{label} cannot parse {module_rel} to stub {function}: {exc}"]
    if not found:
        return [f"{label} cannot stub missing function {function} in {module_rel}."]
    try:
        module_path.write_text(stubbed, encoding="utf-8")
        _drop_pycache(module_path)
        for command in commands:
            output = command_runner(command)
            if is_verification_green(output):
                return [
                    f"{label} declared tests still pass when {function} is stubbed ({command})."
                ]
    except Exception as exc:
        return [f"{label} stub gate failed: {exc}"]
    finally:
        module_path.write_bytes(original)
        _drop_pycache(module_path)
    return []


def load_target_function(module_rel: str, function: str) -> tuple[Any, Callable[..., Any]]:
    relative = normalize_relative_path(module_rel)
    path = safe_path(relative)
    if not path.is_file():
        raise FileNotFoundError(relative)
    errors: list[str] = []
    try:
        module = _load_by_file(path)
        fn = getattr(module, function, None)
        if callable(fn):
            return module, fn
        errors.append(f"file load has no callable {function}")
    except Exception as exc:
        errors.append(f"file load: {exc}")
    try:
        module = _load_by_package(relative)
        fn = getattr(module, function, None)
        if callable(fn):
            return module, fn
        errors.append(f"package import has no callable {function}")
    except Exception as exc:
        errors.append(f"package import: {exc}")
    raise AttributeError("; ".join(errors) or f"{relative} has no callable {function}")


def bind_call_kwargs(fn: Callable[..., Any], call: dict[str, Any], module: Any) -> dict[str, Any]:
    params = inspect.signature(fn).parameters
    names = [name for name in params if name != "self"]
    remaining = dict(call)
    kwargs: dict[str, Any] = {}
    for name in names:
        if name in remaining:
            kwargs[name] = remaining.pop(name)
    if "usage" in names:
        usage = dict(kwargs.get("usage") or {})
        for key in list(remaining):
            if key in USAGE_FIELDS:
                usage[key] = remaining.pop(key)
        msg_fields = {key: remaining.pop(key) for key in list(remaining) if key in MESSAGE_FIELDS}
        if msg_fields and not usage.get("messages"):
            msg_fields.setdefault("role", "tool")
            msg_fields.setdefault("content", "x")
            usage["messages"] = [msg_fields]
        if usage:
            kwargs["usage"] = usage
    remaining.clear()
    if "config" in names and "config" not in kwargs:
        config = _default_config(module)
        if config is None:
            raise TypeError("cannot construct the function's config argument")
        kwargs["config"] = config
    if "salt" in names and "salt" not in kwargs:
        kwargs["salt"] = "test-salt"
    missing = [
        name
        for name, param in params.items()
        if name not in kwargs
        and name != "self"
        and param.default is inspect.Parameter.empty
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if missing:
        raise TypeError(f"missing required arguments: {missing}")
    return kwargs


def _has_assertions(expected: dict[str, Any]) -> bool:
    return bool(expected)


def _compare_example(
    label: str,
    index: int,
    result: Any,
    expected: dict[str, Any],
    raised: BaseException | None,
) -> list[str]:
    prefix = f"{label} example {index}"
    want_exc = expected.get("raises")
    if want_exc:
        want_name = str(want_exc)
        if raised is None:
            return [f"{prefix} expected {want_name} but the call succeeded."]
        if type(raised).__name__ != want_name:
            return [f"{prefix} expected {want_name}, got {type(raised).__name__}."]
        return []
    if raised is not None:
        return [f"{prefix} raised {type(raised).__name__}: {raised}"]

    blockers: list[str] = []
    if "material" in expected:
        algorithm = str(expected.get("algorithm") or "sha256").lower()
        if algorithm == "sha256":
            want = hashlib.sha256(str(expected["material"]).encode("utf-8")).hexdigest()
            if result != want:
                blockers.append(f"{prefix} hash mismatch for material {expected['material']!r}.")
        else:
            blockers.append(f"{prefix} unsupported hash algorithm {algorithm}.")
    if "stored_keys" in expected:
        keys = expected["stored_keys"]
        if not isinstance(result, dict):
            blockers.append(f"{prefix} expected a dict with stored_keys, got {type(result).__name__}.")
        else:
            missing = [key for key in keys if key not in result]
            if missing:
                blockers.append(f"{prefix} missing stored keys: {missing}.")
            if "user_id_hash" in keys and "user_id" in result:
                blockers.append(f"{prefix} stored event still contains user_id.")
    if "value" in expected and result != expected["value"]:
        blockers.append(f"{prefix} expected {expected['value']!r}, got {result!r}.")
    plain = {
        key: want
        for key, want in expected.items()
        if key not in SPECIAL_EXPECT
    }
    if not isinstance(result, dict):
        if len(plain) == 1:
            key, want = next(iter(plain.items()))
            if result != want:
                blockers.append(f"{prefix} {key} expected {want!r}, got {result!r}.")
        elif plain:
            names = ", ".join(sorted(plain))
            blockers.append(
                f"{prefix} expected {names} but the result was {result!r}."
            )
        return blockers
    if isinstance(result, dict):
        for key, want in expected.items():
            if key in SPECIAL_EXPECT:
                continue
            if key not in result:
                blockers.append(f"{prefix} missing result key {key}.")
            elif result[key] != want:
                blockers.append(f"{prefix} {key} expected {want!r}, got {result[key]!r}.")
    return blockers


def _instantiate_config_module(cfg: Any) -> Any:
    if callable(getattr(cfg, "default_config", None)):
        return cfg.default_config()
    org = getattr(cfg, "OrgConfig", None)
    if callable(org):
        try:
            return org()
        except TypeError:
            return None
    return None


def _default_config(module: Any) -> Any:
    for attr in ("default_config", "OrgConfig"):
        obj = getattr(module, attr, None)
        if callable(obj):
            try:
                return obj()
            except TypeError:
                continue
    file = getattr(module, "__file__", None)
    if file:
        cfg_path = Path(file).with_name("config.py")
        if cfg_path.is_file():
            try:
                instantiated = _instantiate_config_module(_load_by_file(cfg_path))
            except Exception:
                instantiated = None
            if instantiated is not None:
                return instantiated
    name = str(getattr(module, "__name__", "") or "")
    pkg = name.rsplit(".", 1)[0] if "." in name else ""
    if not pkg:
        return None
    try:
        cfg = importlib.import_module(f"{pkg}.config")
    except Exception:
        return None
    return _instantiate_config_module(cfg)


def _ensure_import_paths() -> None:
    for path in (WORKSPACE_ROOT / "src", WORKSPACE_ROOT):
        text = str(path)
        if path.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def _load_by_package(module_rel: str) -> Any:
    _ensure_import_paths()
    name = module_name_from_path(module_rel)
    for key in list(sys.modules):
        if key == name or key.startswith(f"{name}."):
            sys.modules.pop(key, None)
    return importlib.import_module(name)


def _load_by_file(path: Path) -> Any:
    unique = f"contract_oracle_{path.stem}_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(unique, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    spec.loader.exec_module(module)
    return module


def _drop_pycache(module_path: Path) -> None:
    cache = module_path.parent / "__pycache__"
    if not cache.is_dir():
        return
    stem = module_path.stem
    for item in cache.glob(f"{stem}*.pyc"):
        try:
            item.unlink()
        except OSError:
            continue
