from types import SimpleNamespace

from orchestrator import knowledge


def test_task_number_parses_graph_ids() -> None:
    assert knowledge.task_number("GRAPH-006") == 6
    assert knowledge.task_number("GRAPH-006.1") == 6
    assert knowledge.task_number("") is None


def test_guidance_for_uses_task_range(monkeypatch) -> None:
    fake = SimpleNamespace(
        prompt_guidance="default guidance",
        task_prompt_guidance=(
            {"from": "GRAPH-003", "to": "GRAPH-005", "guidance": "robot edges"},
            {"from": "GRAPH-006", "to": "GRAPH-008", "guidance": "use ast.walk"},
        ),
        knowledge_packs=(),
    )
    monkeypatch.setattr(knowledge, "PROJECT_PROFILE", fake)
    assert knowledge.guidance_for("GRAPH-006.1") == "use ast.walk"
    assert knowledge.guidance_for("GRAPH-003") == "robot edges"
    assert knowledge.guidance_for("GRAPH-001") == "default guidance"


def test_knowledge_pack_text_loads_markdown(tmp_path, monkeypatch) -> None:
    pack = tmp_path / "python-ast-edges.md"
    pack.write_text("Call is often under Expr.", encoding="utf-8")
    fake = SimpleNamespace(
        prompt_guidance="",
        task_prompt_guidance=(),
        knowledge_packs=(
            {
                "id": "python-ast-edges",
                "from": "GRAPH-006",
                "to": "GRAPH-008",
                "path": str(pack),
            },
        ),
    )
    monkeypatch.setattr(knowledge, "PROJECT_PROFILE", fake)
    text = knowledge.knowledge_pack_text("GRAPH-006")
    assert "python-ast-edges" in text
    assert "Expr" in text
    assert knowledge.knowledge_pack_text("GRAPH-003") == ""
