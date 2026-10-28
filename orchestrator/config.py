from __future__ import annotations

from pathlib import Path

from orchestrator.project import PROJECT_PROFILE
from tools._handoffs import runtime_root

DEFAULT_MODEL = PROJECT_PROFILE.default_model
DEFAULT_FROM_ID = PROJECT_PROFILE.default_from_id
DEFAULT_TO_ID = PROJECT_PROFILE.default_to_id
DEFAULT_MAX_TOOL_STEPS = 80
DEFAULT_MAX_ATTEMPTS = 100
DEFAULT_MAX_SUBTASK_ATTEMPTS = 5
DEFAULT_MAX_PLANNER_ROUNDS = 3
DEFAULT_NUM_CTX = 24576
DEFAULT_NUM_PREDICT = 8192
DEFAULT_CONTEXT_THRESHOLD = 0.85
DEFAULT_MAX_CONTEXT_HANDOVERS = 3
DEFAULT_CHARS_PER_TOKEN = 4
DEFAULT_GIT_TIMEOUT_SECONDS = 120
MAX_COMPACT_EXCERPT_CHARS = 5000
ATTEMPT_REPORTS_DIR = runtime_root() / "recovery"
PLANS_DIR = ATTEMPT_REPORTS_DIR / "plans"

# read_file guardrails (orchestrator only; MCP read_file stays unrestricted)
READ_FILE_MAX_PY_LINES = 1500
READ_FILE_MAX_PY_BYTES = 30_720
READ_FILE_SMALL_FILE_BYTES = 32_768
READ_FILE_ALLOWED_PREFIXES = (
    *PROJECT_PROFILE.read_file_allowed_prefixes,
)
READ_FILE_ALLOWED_SUFFIXES = (
    *PROJECT_PROFILE.read_file_allowed_suffixes,
)
IMPLEMENTATION_PREFIXES = PROJECT_PROFILE.implementation_prefixes
LEGACY_PREFIXES = PROJECT_PROFILE.legacy_prefixes
PROJECT_NAME = PROJECT_PROFILE.name
PROJECT_PROMPT_GUIDANCE = PROJECT_PROFILE.prompt_guidance
TASK_PREFIX = PROJECT_PROFILE.task_prefix
DEFAULT_AUTO_PUSH = PROJECT_PROFILE.auto_push

SEARCH_MAX_MATCHES = 40
SEARCH_FORBIDDEN_LEGACY_QUERIES = (
    r"^def\s*$",
)
TEXT_ONLY_RETRY_MAX = 1
SYNTAX_WRITE_RETRY_MAX = 3
MALFORMED_TOOL_CALL_RETRY_MAX = 1
MAX_CONTEXT_PACK_SNIPPET_CHARS = 10_000
REPO_CONTEXT_TOKEN_BUDGET = 2000
MAX_REPO_CONTEXT_MATCHES = 12
TARGET_SNAPSHOT_EXCERPT_LINES = 40
MODULE_PREAMBLE_MAX_LINES = 25
TRUNCATED_EXCERPT_MARKER = "[truncated; full output stored in handoff]"
DEFAULT_MAX_READ_ONLY_BEFORE_WRITE = 6
DEFAULT_READ_ONLY_WARN_THRESHOLD = 4
MAX_PARALLEL_READONLY_TOOLS = 4

READ_ONLY_TOOLS = frozenset({
    "read_file",
    "search_files",
    "extract_symbol",
    "find_symbol",
    "list_directory",
    "read_handoff",
    "get_subtask_context",
    "get_active_subtask",
    "get_verification_checklist",
})

FORBIDDEN_COMMAND_PATTERNS = (
    r"\brm\s+-rf\b",
    r"\bRemove-Item\b",
    r"\bdel\s+",
    r"\brmdir\b",
    r"\bgit\s+reset\b",
    r"\bgit\s+checkout\b",
    r"\bgit\s+clean\b",
    r"\bdrop\s+database\b",
)
