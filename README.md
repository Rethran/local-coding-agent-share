# Local Coding Agent

Reusable local Ollama coding orchestrator.

The agent code lives in this repository. Application workspaces live elsewhere,
and `projects/*.json` profiles decide which workspace and requirement folder are
active.

## Profiles

Living Graph:

```powershell
cd C:\Users\santt\Desktop\local-coding-agent
$env:LOCAL_AGENT_PROFILE = "living-graph"
python main.py --from GRAPH-001 --to GRAPH-003
```

The Living Graph profile defaults to `qwen3:8b`. Override the model when your
local Ollama model has a different name:

```powershell
python main.py --model qwen3:8b --from GRAPH-001 --to GRAPH-003
```

## Layout

```text
requirements/
  living-graph/
    GRAPH-*.json
projects/
  living-graph.json
```

Requirement state, handoffs, and recovery files belong to the agent. Code edits
and subtask commits happen in the profile's configured `workspaceRoot`.

## Smoke Checks

```powershell
python -m pytest tests/test_project_profile.py tests/test_task_prefix_config.py -q

$env:LOCAL_AGENT_PROFILE = "living-graph"
python -c "from tools import _tasks; print(sorted(_tasks.load_task_definitions()))"
```
