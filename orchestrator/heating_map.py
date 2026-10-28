"""Heating-app repo map: app/ only, workspace-relative paths, HEAD-cached."""
from __future__ import annotations

import sys
from pathlib import Path

from tools._common import WORKSPACE_ROOT

_REPMAP_ROOT = Path(__file__).resolve().parent.parent / "repomap"
if str(_REPMAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPMAP_ROOT))

HEATING_PREFIX = "app"
HEATING_SCAN_ROOT = WORKSPACE_ROOT / HEATING_PREFIX
HEATING_CACHE_DIR = WORKSPACE_ROOT / ".repomap_cache"


def get_heating_repo_map() -> dict:
    """Return the cached call-graph map of app/, with files like app/domain/foo.py."""
    from repomap.scoped import get_or_build_prefixed_map

    if not HEATING_SCAN_ROOT.is_dir():
        return {"symbols": [], "edges": []}
    return get_or_build_prefixed_map(HEATING_SCAN_ROOT, HEATING_PREFIX, HEATING_CACHE_DIR)
