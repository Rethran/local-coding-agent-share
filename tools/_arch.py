import hashlib
import json
from pathlib import Path

from tools._common import WORKSPACE_ROOT  # noqa: F401 — ensures arch-mcp is on sys.path

from cache import ModuleCache
from ollama_client import reduce_architecture, summarize_module
from structure_extractor import build_structure_map

cache = ModuleCache(Path(".arch_cache"))


def hash_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_structure_file(root: Path, structure: dict) -> Path:
    out_path = root / ".structure.json"
    out_path.write_text(json.dumps(structure, indent=2))
    return out_path
