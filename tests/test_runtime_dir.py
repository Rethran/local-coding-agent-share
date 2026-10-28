"""LOCAL_AGENT_RUNTIME_DIR must relocate handoffs and recovery artifacts.

The paths are module-level constants resolved at import time, so the test
spawns a fresh interpreter with the environment variable set instead of
monkeypatching an already-imported module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent

_PROBE = (
    "import json\n"
    "from tools import _handoffs\n"
    "from orchestrator import config, continuity, session\n"
    "print(json.dumps({\n"
    "  'handoffs': str(_handoffs.HANDOFFS_DIR),\n"
    "  'consumed': str(_handoffs.CONSUMED_DIR),\n"
    "  'recovery': str(config.ATTEMPT_REPORTS_DIR),\n"
    "  'plans': str(config.PLANS_DIR),\n"
    "  'continuity': str(continuity.CONTINUITY_DIR),\n"
    "  'sessions': str(session.SESSIONS_DIR),\n"
    "}))\n"
)


def _probe(env_override: dict[str, str]) -> dict[str, str]:
    env = {**os.environ, **env_override}
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=AGENT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_default_runtime_dir_is_agent_root() -> None:
    env = {k: v for k, v in os.environ.items() if k != "LOCAL_AGENT_RUNTIME_DIR"}
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=AGENT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    paths = json.loads(completed.stdout.strip().splitlines()[-1])
    assert Path(paths["handoffs"]) == AGENT_ROOT / "handoffs"
    assert Path(paths["recovery"]) == AGENT_ROOT / "recovery"


def test_runtime_dir_override_relocates_every_runtime_artifact(tmp_path: Path) -> None:
    runtime = tmp_path / "eval-run-1"
    paths = _probe({"LOCAL_AGENT_RUNTIME_DIR": str(runtime)})

    assert Path(paths["handoffs"]) == runtime / "handoffs"
    assert Path(paths["consumed"]) == runtime / "handoffs" / ".consumed"
    assert Path(paths["recovery"]) == runtime / "recovery"
    assert Path(paths["plans"]) == runtime / "recovery" / "plans"
    assert Path(paths["continuity"]) == runtime / "recovery" / "continuity"
    assert Path(paths["sessions"]) == runtime / "recovery" / "sessions"
