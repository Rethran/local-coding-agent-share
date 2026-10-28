from orchestrator.write_guard import refuse_worker_write


def test_test_writer_cannot_write_production_module() -> None:
    merged = {
        "targetArtifacts": {
            "module": "src/living_graph/robot_extractor.py",
            "test": "tests/test_robot_extractor.py",
        }
    }
    refusal = refuse_worker_write(
        "src/living_graph/robot_extractor.py",
        role="test_writer",
        merged_subtask=merged,
    )
    assert refusal is not None
    assert "production file" in refusal


def test_test_writer_can_write_tests() -> None:
    merged = {
        "targetArtifacts": {
            "module": "src/living_graph/robot_extractor.py",
            "test": "tests/test_robot_extractor.py",
        }
    }
    assert refuse_worker_write(
        "tests/test_robot_extractor.py",
        role="test_writer",
        merged_subtask=merged,
    ) is None
    assert refuse_worker_write(
        "tests/fixtures/simple_test.robot",
        role="test_writer",
        merged_subtask=merged,
    ) is None


def test_implementer_cannot_edit_test_file() -> None:
    merged = {
        "targetArtifacts": {
            "module": "app/domain/schedule.py",
            "test": "tests/test_schedule_toggle.py",
        }
    }
    refusal = refuse_worker_write(
        "tests/test_schedule_toggle.py",
        role="implementer",
        merged_subtask=merged,
    )
    assert refusal is not None
    assert "test file" in refusal
    assert refuse_worker_write(
        "app/domain/schedule.py",
        role="implementer",
        merged_subtask=merged,
    ) is None
