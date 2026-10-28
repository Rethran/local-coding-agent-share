Python AST call-graph rules for GRAPH-006 through GRAPH-008:

- Parse with ast.parse. Walk FunctionDef bodies with ast.walk, not only direct body statements.
- A call like `connect()` is ast.Expr wrapping ast.Call, not a top-level ast.Call statement.
- Record an edge only when the callee is an ast.Name whose id is another FunctionDef in the same file.
- Edge ids are `{source_path}::{name}` with forward slashes. Copy field construction from extract_robot_edges.
- In tests, assert caller_name and callee_name (and resolved). Do not compare Edge objects with `==`.
- Do not parse Robot files in these tasks.
