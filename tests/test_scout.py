"""Tests for the generic parallel scout phase."""

from orchestrator.scout import attach_scout_brief, build_scout_plan


def test_build_scout_plan_is_generic_and_deduplicates_paths() -> None:
    subtask = {
        "targetArtifacts": {
            "module": "app/api/schedule.py",
            "test": "tests/test_schedule_toggle.py",
        },
        "relatedArtifacts": [
            {"path": "app/api/schedule.py", "purpose": "duplicate"},
            {"path": "app/services/schedule_store.py", "purpose": "store"},
        ],
    }

    plan = build_scout_plan(subtask)

    assert [request.path for request in plan] == [
        "app/api/schedule.py",
        "tests/test_schedule_toggle.py",
        "app/services/schedule_store.py",
    ]


def test_attach_scout_brief_reuses_preloaded_sources_in_parallel() -> None:
    subtask = {
        "targetArtifacts": {
            "module": "app/example.py",
            "test": "tests/test_example.py",
        },
    }
    context_pack = {
        "referenceSnippets": [
            {"path": "app/example.py", "source": "def example():\n    return 1\n"},
            {"path": "tests/test_example.py", "source": "def test_example():\n    pass\n"},
        ],
    }

    enriched = attach_scout_brief(context_pack, subtask)
    brief = enriched["scoutBrief"]

    assert brief["parallel"] is True
    assert brief["coveredPaths"] == ["app/example.py", "tests/test_example.py"]
    assert all(finding["status"] == "ok" for finding in brief["findings"])
    assert "scoutBrief" not in context_pack
