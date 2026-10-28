# repomap

A minimal, Python-native, deterministic structural map of a codebase for AI
coding agents — same idea as `ripwire` (Red Hat Emerging Technologies),
scoped down to your own repos instead of "any repo, any language."

Built test-first: every module below was implemented against a failing
test suite before the implementation existed (see `tests/`).

## Pipeline

1. `repomap/symbols.py` — Tree-sitter symbol extraction (name, signature, line range)
2. `repomap/graph.py` — call graph construction + PageRank ranking
3. `repomap/churn.py` — git churn + change amplification (co-change) from `git log`
4. `repomap/coverage_join.py` — joins a `coverage.py` JSON report onto symbols by line range
5. `repomap/blast_radius.py` — reverse-BFS reachability: "who is affected if this changes"
6. `repomap/query.py` — ties it all together: `build_repo_map()` (cache-once, JSON-serializable)
   + `query()` (cheap, repeatable per agent turn, token-budgeted)
7. `repomap/mcp_tool.py` — FastMCP wrapper (`get_repo_context` tool), with HEAD-commit-keyed
   caching and a slot for your existing `ValidationMiddleware`

## Determinism

`tests/test_determinism.py` runs `build_repo_map()` three times over the fixture
repo and requires byte-identical JSON — the same guarantee ripwire enforces with
`diff -q a b` in its own CI. Wire this into your pipeline as a required check, not
just a local test: it's the property that makes caching by HEAD commit safe.

## Usage

```python
from pathlib import Path
from repomap.query import build_repo_map, query

repo_map = build_repo_map(Path("."))          # do this once per HEAD commit
result = query(repo_map, "cache invalidation", max_hops=2, token_budget=2000)
# result["matches"] -> ranked symbols with signature, churn, score, blast_radius
```

As an MCP tool:

```python
from repomap.mcp_tool import register_repo_context_tool
register_repo_context_tool(mcp, repo_root=PROJECT_ROOT,
                            validation_middleware=your_existing_middleware)
```

## What's intentionally left out (vs. ripwire)

- Only Python is wired up (`tree_sitter` + `tree_sitter_python`).
  Adding a language means adding a parser + a symbol-walker for that
  grammar's function/class node types in `symbols.py` — `graph.py`,
  `churn.py`, `blast_radius.py`, and `query.py` are language-agnostic.
- Call resolution is by simple name match, not real import/type
  resolution. Fine for ranking; will occasionally link two same-named
  functions in different files that aren't actually related. If that
  becomes a real problem, resolve `import` statements in `symbols.py`
  and thread the resolved module path into `graph.py`'s matching step.
- No XML output, no C++, no single-binary distribution — this runs as
  a library inside your existing FastMCP server.

## Running the tests

```bash
pip install -r requirements.txt
pytest -v
```
