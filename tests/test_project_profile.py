from __future__ import annotations

import importlib
from pathlib import Path


def test_living_graph_profile_file_points_to_graph_workspace() -> None:
    from orchestrator.project import PROFILES_DIR

    profile_path = PROFILES_DIR / "living-graph.json"
    assert profile_path.exists()
    text = profile_path.read_text(encoding="utf-8")
    assert "living-3d-graph" in text
    assert '"taskPrefix": "GRAPH"' in text


def test_project_profile_can_load_living_graph(monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_AGENT_PROFILE", "living-graph")

    import orchestrator.project as project

    try:
        reloaded = importlib.reload(project)
        profile = reloaded.load_project_profile()

        assert profile.task_prefix == "GRAPH"
        assert profile.default_model == "qwen3:8b"
        assert profile.workspace_root == Path(r"C:\Users\santt\Desktop\living-3d-graph")
        assert profile.requirements_dir.name == "living-graph"
        assert "src/" in profile.implementation_prefixes
        assert profile.legacy_prefixes == ()
        assert profile.knowledge_packs
        assert profile.knowledge_packs[0]["id"] == "python-ast-edges"
        assert "ast.parse" in profile.task_prompt_guidance[-1]["guidance"]
    finally:
        monkeypatch.delenv("LOCAL_AGENT_PROFILE", raising=False)
        importlib.reload(project)
