"""Eval harness: dataset, isolation, scorers, report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.build_dataset import REGRESSION_IDS, SMOKE_IDS, SUBJECT_RE, build_cases, tags_for, write_cases
from evals.criteria import CRITERIA, PROCESS_BUDGET
from evals.isolation import (
    IsolatedRun,
    IsolationError,
    overlay_reference_tests,
    prepare_isolated_run,
    summary_excerpt,
)
from evals.report import (
    bernoulli_stderr,
    build_index,
    build_report,
    build_review,
    compare_reports,
    failures_first,
    finalize_log,
    find_report_for_log,
    flaky_cases,
    index_rows,
    is_dry_run_report,
    primary_failure,
    report_stem_for_log,
    summarize,
    write_pdf,
    write_report,
)

DRY_RUN_LOG = Path(__file__).resolve().parent.parent / "evals" / "logs" / (
    "2026-09-30T19-59-12-00-00_tdd-coder_X8q6NmCMg2EBumAtqvVhXn.eval"
)
from evals.scorers import score_process_budget, score_subtask_done

AGENT_ROOT = Path(__file__).resolve().parent.parent
LIVING_GRAPH = Path(r"C:\Users\santt\Desktop\living-3d-graph")


def test_subject_regex_parses_subtask_commits() -> None:
    match = SUBJECT_RE.match("GRAPH-047.1: Resource callers are user_keyword")
    assert match is not None
    assert match.group(1) == "GRAPH-047"
    assert match.group(2) == "1"


def test_tags_cover_smoke_and_regression() -> None:
    assert set(tags_for("GRAPH-011.1")) == {"smoke", "regression"}
    assert tags_for("GRAPH-054.1") == ["regression"]
    assert tags_for("GRAPH-040.1") == []
    assert len(SMOKE_IDS) == 3
    assert "GRAPH-051.1" not in REGRESSION_IDS
    assert "GRAPH-050.1" in REGRESSION_IDS


def test_build_dataset_from_living_graph_repo() -> None:
    if not LIVING_GRAPH.exists():
        pytest.skip("Living Graph workspace is not on this machine")
    cases = build_cases()
    ids = {case["id"] for case in cases}
    assert "GRAPH-047.1" in ids
    graph_047 = next(case for case in cases if case["id"] == "GRAPH-047.1")
    assert graph_047["metadata"]["base_commit"]
    assert graph_047["metadata"]["accepted_commit"]
    assert "tests/test_resource_caller_type.py" in graph_047["metadata"]["reference_tests"]
    assert "smoke" in graph_047["metadata"]["tags"]
    smoke = [case for case in cases if "smoke" in case["metadata"]["tags"]]
    assert {case["id"] for case in smoke} == set(SMOKE_IDS)


def test_write_cases_roundtrip(tmp_path: Path) -> None:
    path = write_cases(
        tmp_path / "cases.jsonl",
        [{
            "id": "GRAPH-011.1",
            "input": "fan-in",
            "target": "",
            "metadata": {"tags": ["smoke"], "task_id": "GRAPH-011"},
        }],
    )
    line = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert line["id"] == "GRAPH-011.1"


def test_subtask_done_reads_status_not_exit_code() -> None:
    ok, why = score_subtask_done(
        {"status": "FAILED", "subtasks": [{"id": "GRAPH-011.1", "status": "DONE"}]},
        "GRAPH-011.1",
    )
    assert ok
    failed, explanation = score_subtask_done(
        {
            "status": "FAILED",
            "lastFailureReason": "tool step budget",
            "subtasks": [{"id": "GRAPH-011.1", "status": "FAILED"}],
        },
        "GRAPH-011.1",
    )
    assert not failed
    assert "tool step budget" in explanation


def test_process_budget_flags_loops() -> None:
    summary = {
        "totals": {
            "attempts": 2,
            "steps": 10,
            "toolCalls": 10,
            "toolErrors": 1,
        },
        "attempts": [{"maxRepeatOfOneCall": PROCESS_BUDGET["max_repeat_of_one_call"] + 1}],
    }
    ok, why, checks = score_process_budget(summary)
    assert not ok
    assert checks["maxRepeatOfOneCall"]["ok"] is False
    assert "maxRepeatOfOneCall" in why


def test_process_budget_passes_clean_run() -> None:
    ok, _, checks = score_process_budget({
        "totals": {"attempts": 2, "steps": 12, "toolCalls": 12, "toolErrors": 0},
        "attempts": [{"maxRepeatOfOneCall": 2}],
    })
    assert ok
    assert all(row["ok"] for row in checks.values())


def test_report_failures_first_flaky_and_noise(tmp_path: Path) -> None:
    trials = [
        {
            "sampleId": "GRAPH-011.1",
            "epoch": 1,
            "title": "fan-in",
            "input": "Count distinct callers",
            "scores": {
                "subtask_done": {"ok": True, "explanation": "DONE"},
                "process_budget": {"ok": True, "explanation": "ok"},
            },
            "wallSeconds": 10,
            "tokens": 100,
        },
        {
            "sampleId": "GRAPH-011.1",
            "epoch": 2,
            "title": "fan-in",
            "input": "Count distinct callers",
            "scores": {
                "subtask_done": {"ok": False, "explanation": "budget"},
                "process_budget": {"ok": False, "explanation": "loops"},
            },
            "wallSeconds": 40,
            "tokens": 200,
        },
        {
            "sampleId": "GRAPH-013.1",
            "epoch": 1,
            "title": "cycles",
            "input": "True on a two-node cycle",
            "scores": {
                "subtask_done": {"ok": True, "explanation": "DONE"},
                "process_budget": {"ok": True, "explanation": "ok"},
            },
            "wallSeconds": 12,
            "tokens": 80,
        },
    ]
    summary = summarize(trials)
    assert summary["scorers"]["subtask_done"]["passed"] == 2
    assert summary["scorers"]["subtask_done"]["n"] == 3
    assert summary["nSamples"] == 2
    assert summary["scorers"]["subtask_done"]["stderr"] == round(bernoulli_stderr(2 / 3, 3), 4)
    fails = failures_first(trials)
    assert fails[0]["criterion"] in {"subtask_done", "process_budget"}
    flaky = flaky_cases(trials)
    assert flaky == [{"sampleId": "GRAPH-011.1", "scorer": "subtask_done", "passed": 1, "epochs": 2}]

    report = build_report(
        trials,
        header={"sourceLog": "fixture", "model": "night-runner:latest", "promptVariant": "baseline"},
    )
    assert "subtask_done" in report["criteria"]
    html_path, json_path = write_report(report, tmp_path / "run")
    html = html_path.read_text(encoding="utf-8")
    assert "Epäonnistumiset ensin" in html
    assert "GRAPH-011.1" in html
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema"] == "eval-report/v1"

    noisy = compare_reports(report, report)
    assert all(row["delta"] == 0 for row in noisy)
    assert all(row["signal"] is False for row in noisy)


def _review_trial(
    sample_id: str,
    epoch: int,
    *,
    tests: bool,
    done: bool,
    wall: float,
    steps: int,
    checks: dict | None = None,
    timed_out: bool = False,
) -> dict:
    return {
        "sampleId": sample_id,
        "epoch": epoch,
        "title": sample_id,
        "input": sample_id,
        "scores": {
            "subtask_done": {"ok": done, "explanation": "DONE" if done else "not done"},
            "reference_tests_pass": {"ok": tests, "explanation": "pass" if tests else "fail"},
            "process_budget": {"ok": True, "explanation": "ok", **({"checks": checks} if checks else {})},
        },
        "wallSeconds": wall,
        "steps": steps,
        "totals": {"steps": steps},
        "timedOut": timed_out,
        "stopReason": "timeout" if timed_out else "",
    }


def test_review_metrics_buckets_cost_false_done_and_noise(tmp_path: Path) -> None:
    reliable = [
        _review_trial("GRAPH-003.1", epoch, tests=True, done=True, wall=60, steps=10)
        for epoch in (1, 2, 3)
    ]
    fragile = [
        _review_trial("GRAPH-006.1", 1, tests=True, done=True, wall=90, steps=20),
        _review_trial("GRAPH-006.1", 2, tests=True, done=True, wall=90, steps=20),
        _review_trial("GRAPH-006.1", 3, tests=False, done=True, wall=180, steps=50),
    ]
    cannot = [
        _review_trial(
            "GRAPH-011.1",
            1,
            tests=False,
            done=False,
            wall=30,
            steps=5,
            checks={"maxRepeatOfOneCall": {"ok": False, "value": 9, "limit": 5}},
        ),
        _review_trial(
            "GRAPH-011.1",
            2,
            tests=False,
            done=False,
            wall=30,
            steps=5,
            timed_out=True,
            checks={"maxRepeatOfOneCall": {"ok": False, "value": 9, "limit": 5}},
        ),
        _review_trial(
            "GRAPH-011.1",
            3,
            tests=False,
            done=False,
            wall=30,
            steps=5,
            checks={"toolErrorRate": {"ok": False, "value": 0.8, "limit": 0.4}},
        ),
    ]
    trials = reliable + fragile + cannot
    false_done = next(row for row in trials if row["sampleId"] == "GRAPH-006.1" and row["epoch"] == 3)
    looped = next(row for row in trials if row["sampleId"] == "GRAPH-011.1" and row["epoch"] == 1)
    timed = next(row for row in trials if row["sampleId"] == "GRAPH-011.1" and row["epoch"] == 2)
    assert primary_failure(false_done) == "reference_tests"
    assert primary_failure(looped) == "loop"
    assert primary_failure(timed) == "timeout"

    review = build_review(trials)
    assert [row["sampleId"] for row in review["consistency"]["reliable"]] == ["GRAPH-003.1"]
    assert [row["passed"] for row in review["consistency"]["fragile"]] == [2]
    assert [row["sampleId"] for row in review["consistency"]["cannot"]] == ["GRAPH-011.1"]
    assert review["falseDone"]["falseDone"] == 1
    assert review["falseDone"]["claimedDone"] == 6
    assert review["falseDone"]["rate"] == round(1 / 6, 4)
    assert review["cost"]["gpuMinutes"] == 10.5
    assert review["cost"]["fullSuccesses"] == 1
    assert review["cost"]["minutesPerSuccess"] == 10.5
    assert review["timing"]["wallSeconds"]["median"] == 60
    assert review["timing"]["wallSeconds"]["worst"] == 180
    assert review["timing"]["steps"]["median"] == 10
    assert review["timing"]["steps"]["worst"] == 50
    counts = review["failureTypes"]["counts"]
    assert counts["timeout"] == 1
    assert counts["loop"] == 1
    assert counts["tool_errors"] == 1
    assert counts["reference_tests"] == 1
    assert review["failureTypes"]["failedTrials"] == sum(counts.values())

    same = build_review(trials, review)
    assert same["deltas"]["falseDoneRate"]["delta"] == 0
    assert same["deltas"]["falseDoneRate"]["noise"] is True
    assert same["deltas"]["referenceTestsPassRate"]["noise"] is True
    moved = json.loads(json.dumps(review))
    moved["falseDone"]["rate"] = 1.0
    shifted = build_review(trials, moved)
    assert shifted["deltas"]["falseDoneRate"]["noise"] is False

    report = build_report(
        trials,
        header={"sourceLog": "fixture", "model": "night-runner:latest", "promptVariant": "baseline"},
    )
    assert report["schema"] == "eval-report/v1"
    assert report["trend"][-1]["reliableCases"] == 1
    assert report["trend"][-1]["falseDoneRate"] == round(1 / 6, 4)
    html_path, _json_path = write_report(report, tmp_path / "review")
    html = html_path.read_text(encoding="utf-8")
    assert "Katselmoijalle" in html
    assert "Tekijälle" in html
    assert "GRAPH-003.1" in html
    assert "10,5" in html
    assert "ei tallennettu" not in html


def test_summary_excerpt_is_copied_before_cleanup(tmp_path: Path) -> None:
    missing = summary_excerpt(tmp_path / "gone.json")
    assert missing == {"totals": {}, "stopReasons": []}
    path = tmp_path / "run_summary.json"
    path.write_text(
        json.dumps({
            "totals": {
                "steps": 12,
                "toolCalls": 4,
                "toolErrors": 1,
                "attempts": 2,
                "repeatedIdenticalCalls": 3,
                "promptTokens": 10,
                "completionTokens": 5,
                "wallSeconds": 99,
            },
            "attempts": [{"stopReason": "maxRepeatOfOneCall"}, {"stopReason": ""}],
        }),
        encoding="utf-8",
    )
    excerpt = summary_excerpt(path)
    assert excerpt["totals"]["steps"] == 12
    assert excerpt["totals"]["toolErrors"] == 1
    assert "wallSeconds" not in excerpt["totals"]
    assert excerpt["stopReasons"] == ["maxRepeatOfOneCall"]
    packed = IsolatedRun(
        sample_id="GRAPH-003.1",
        worktree=tmp_path,
        requirements_dir=tmp_path,
        runtime_dir=tmp_path,
        repo=tmp_path,
        run_summary_path=path,
        state_path=tmp_path / "state.json",
    ).to_dict()
    assert packed["totals"] == excerpt["totals"]
    assert packed["stopReasons"] == ["maxRepeatOfOneCall"]


def test_write_pdf_reports_missing_browser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evals.report.find_browser", lambda: None)
    html = tmp_path / "report.html"
    html.write_text("<html><body>ok</body></html>", encoding="utf-8")
    assert write_pdf(html) is None


SUMMA_FIXTURE = Path(__file__).resolve().parent.parent / "evals" / "examples" / "summa.json"


def test_summa_example_answers_in_finnish(tmp_path: Path) -> None:
    from evals.report import load_trials, render_html

    trials, header = load_trials(SUMMA_FIXTURE)
    dry = {
        "dryRun": True,
        "sourceLog": "dry.eval",
        "trials": trials[:1],
        "summary": {"scorers": {}, "totalTokens": 0},
    }
    report = build_report(trials, header=header, trend_reports=[dry])
    brief = report["brief"]
    assert "summa onnistui kaikilla kolmella kerralla" in brief["assign"].lower()
    assert "erotus onnistui vain joskus, 2 kertaa 3 kerrasta" in brief["assign"].lower()
    assert "tulo ei onnistunut kertaakaan" in brief["assign"].lower()
    assert "9 minuuttia" in brief["cost"]
    assert "Yksi tehtävä onnistui joka kerralla" in brief["cost"]
    assert "ei voi päätellä" in brief["trend"]
    assert "4 kertaa 9 kerrasta" in brief["trust"]
    assert "1 minuutin" in brief["time"]
    assert "Askeleita ei ole kirjattu" in brief["time"]
    assert "ohjeelle, jonka malli sai" in brief["fix"]
    assert brief["headline"].startswith("1 luotettava, 1 hauras, 1 ei osaa")

    one_try = [row for row in trials if row["epoch"] == 1]
    partial = build_report(one_try, header=header)
    assert "ei vielä ole kolme kertaa kolmesta" in partial["brief"]["assign"]

    same = dict(report)
    same["sourceLog"] = header["sourceLog"]
    again = build_report(trials, header=header, trend_reports=[same])
    assert "ei voi päätellä" in again["brief"]["trend"]
    alien = {
        "sourceLog": "graph.eval",
        "trials": [{"sampleId": "GRAPH-003.1", "epoch": 1, "scores": {}, "wallSeconds": 1}],
        "review": {"reliableShare": 0.0, "nSamples": 1, "consistency": {"reliable": [{"sampleId": "GRAPH-003.1"}]}},
    }
    other_tasks = build_report(trials, header=header, trend_reports=[alien])
    assert "ei voi päätellä" in other_tasks["brief"]["trend"]

    html = render_html(report)
    lowered = html.lower()
    for word in ("stderr", "epoch", "false-done", "pass rate", "reference_tests"):
        assert word not in lowered
    written, _json_path = write_report(report, tmp_path / "esimerkki-summa")
    assert "Mihin tehtäviin agentin voi laittaa" in written.read_text(encoding="utf-8")


def test_laskin_example_fails_before_implementation_then_checks_again() -> None:
    from evals.example_run import run_trial

    good = run_trial("summa", 1, "return a + b")
    assert good["redFailed"] is True
    assert good["scores"]["reference_tests_pass"]["ok"] is True
    assert good["scores"]["subtask_done"]["ok"] is True

    bad = run_trial("tulo", 1, "return 0")
    assert bad["redFailed"] is True
    assert bad["scores"]["reference_tests_pass"]["ok"] is False
    assert "assert" in bad["scores"]["reference_tests_pass"]["explanation"].lower()


def test_report_stem_is_deterministic(tmp_path: Path) -> None:
    log = tmp_path / "2026-10-01T17-59-24-00-00_tdd-coder_M7D7.eval"
    assert report_stem_for_log(log, tmp_path / "reports").name == "2026-10-01T17-59-24-00-00_tdd-coder_M7D7"


def test_finalize_log_writes_html_json_index_and_pdf_only_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not DRY_RUN_LOG.exists():
        pytest.skip("dry-run smoke log is not on this machine")
    calls: list[Path] = []

    def fake_pdf(html_path: Path) -> Path:
        calls.append(html_path)
        pdf = html_path.with_suffix(".pdf")
        pdf.write_bytes(b"%PDF-1.4 fake")
        return pdf

    monkeypatch.setattr("evals.report.write_pdf", fake_pdf)
    reports_dir = tmp_path / "reports"
    result = finalize_log(DRY_RUN_LOG, reports_dir=reports_dir)
    assert result["skipped"] is False
    assert result["runStatus"] == "success"
    assert Path(result["html"]).exists()
    assert Path(result["json"]).exists()
    assert result["pdf"] and Path(result["pdf"]).exists()
    assert (reports_dir / "index.html").exists()
    assert len(calls) == 1
    report = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert report["runStatus"] == "success"
    assert report["dryRun"] is True
    assert report["plannedTrials"] == 3
    assert is_dry_run_report(report)

    again = finalize_log(DRY_RUN_LOG, reports_dir=reports_dir)
    assert again["skipped"] is True
    assert len(calls) == 1
    assert find_report_for_log(DRY_RUN_LOG, reports_dir) == Path(result["json"])

    # Not successful: HTML + JSON, no PDF unless pdf_always.
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(DRY_RUN_LOG))
    log.status = "cancelled"
    partial_dir = tmp_path / "partial"
    partial = finalize_log(DRY_RUN_LOG, reports_dir=partial_dir, log=log, force=True)
    assert partial["pdf"] is None
    assert partial["runStatus"] == "cancelled"
    html = Path(partial["html"]).read_text(encoding="utf-8")
    assert "Ajo ei valmistunut" in html
    assert len(calls) == 1
    forced = finalize_log(DRY_RUN_LOG, reports_dir=partial_dir, log=log, force=True, pdf_always=True)
    assert forced["pdf"]
    assert len(calls) == 2


def test_index_rows_list_every_report_with_links(tmp_path: Path) -> None:
    trials = [
        _review_trial("GRAPH-003.1", 1, tests=True, done=True, wall=60, steps=10),
        _review_trial("GRAPH-006.1", 1, tests=False, done=True, wall=90, steps=20),
    ]
    first = build_report(
        trials,
        header={"sourceLog": "a.eval", "model": "night-runner:latest", "promptVariant": "baseline", "runStatus": "success"},
    )
    second = build_report(
        trials,
        header={"sourceLog": "b.eval", "model": "qwen3:8b", "promptVariant": "baseline", "runStatus": "cancelled"},
    )
    second["createdAt"] = "2030-01-01T00:00:00+00:00"
    write_report(first, tmp_path / "a")
    write_report(second, tmp_path / "b")
    (tmp_path / "a.pdf").write_bytes(b"%PDF")
    rows = index_rows(tmp_path)
    assert [row["name"] for row in rows] == ["b", "a"]
    by_name = {row["name"]: row for row in rows}
    assert by_name["a"]["pdf"] == "a.pdf"
    assert by_name["b"]["pdf"] is None
    assert by_name["a"]["html"] == "a.html"
    assert by_name["b"]["runStatus"] == "cancelled"
    assert by_name["a"]["referenceTestsPassRate"] == 0.5
    assert by_name["a"]["dryRun"] is False
    out = build_index(tmp_path)
    html = out.read_text(encoding="utf-8")
    assert "qwen3:8b" in html
    assert 'href="a.pdf"' in html
    assert find_report_for_log(tmp_path / "logs" / "b.eval", tmp_path) == tmp_path / "b.json"


def test_run_progress_counts_eta_and_handles_empty() -> None:
    from datetime import datetime, timezone

    from evals.dashboard import run_progress

    empty = run_progress([], {"plannedEpochs": 3, "plannedSamples": 12, "runStatus": "started"})
    assert empty["done"] == 0
    assert empty["total"] == 36
    assert empty["percent"] == 0.0
    assert empty["etaSeconds"] is None
    assert empty["medianWallSeconds"] is None

    trials = [
        _review_trial("GRAPH-003.1", 1, tests=True, done=True, wall=100, steps=10),
        _review_trial("GRAPH-006.1", 1, tests=False, done=True, wall=300, steps=20),
        _review_trial("GRAPH-011.1", 1, tests=True, done=True, wall=200, steps=5),
    ]
    now = datetime(2026, 10, 3, 12, 0, 30, tzinfo=timezone.utc)
    heartbeat = {
        "updatedAt": "2026-10-03T12:00:00+00:00",
        "current": {"sampleId": "GRAPH-013.1", "epoch": 1, "startedAt": "2026-10-03T12:00:00+00:00"},
    }
    progress = run_progress(
        trials,
        {"plannedEpochs": 1, "plannedSamples": 12, "runStatus": "started"},
        heartbeat,
        now=now,
    )
    assert progress["done"] == 3
    assert progress["total"] == 12
    assert progress["percent"] == 25.0
    assert progress["medianWallSeconds"] == 200.0
    assert progress["etaSeconds"] == 9 * 200
    assert progress["current"] == {"sampleId": "GRAPH-013.1", "epoch": 1, "elapsedSeconds": 30}
    assert progress["consistency"] == {"reliable": 2, "fragile": 0, "cannot": 1}
    assert progress["falseDone"]["falseDone"] == 1
    assert progress["scorers"]["reference_tests_pass"]["n"] == 3
    assert [row["sampleId"] for row in progress["recent"]][-1] == "GRAPH-011.1"

    no_plan = run_progress(trials, {"runStatus": "started"}, {"total": 24})
    assert no_plan["total"] == 24
    finished = run_progress(trials, {"plannedEpochs": 1, "plannedSamples": 3, "runStatus": "success"})
    assert finished["etaSeconds"] == 0.0
    assert finished["percent"] == 100.0


def test_dashboard_server_serves_api_and_reports(tmp_path: Path) -> None:
    import json as _json
    import urllib.error
    import urllib.request

    from evals.dashboard import serve

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    (reports_dir / "note.html").write_text("<p>hello</p>", encoding="utf-8")
    server, stop = serve(
        port=0,
        interval=60,
        finalize=False,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        live_dir=tmp_path / "live",
        block=False,
    )
    try:
        port = server.server_address[1]
        payload = _json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs").read())
        assert payload["runs"] == []
        assert payload["lastError"] is None
        page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode("utf-8")
        assert "Ajon eteneminen" in page
        served = urllib.request.urlopen(f"http://127.0.0.1:{port}/reports/note.html").read().decode("utf-8")
        assert served == "<p>hello</p>"
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/reports/..%2Fsecret.txt")
        assert missing.value.code == 404
    finally:
        stop.set()
        server.shutdown()
        server.server_close()


def test_hook_writes_heartbeat_and_finalizes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from types import SimpleNamespace

    import evals.report as report_module
    from evals.hooks import ReportHooks, heartbeat_path, read_heartbeat

    live_dir = tmp_path / "live"
    hook = ReportHooks(live_dir=live_dir)
    spec = SimpleNamespace(
        config=SimpleNamespace(epochs=2),
        dataset=SimpleNamespace(samples=3),
        metadata={"model_tag": "night-runner:latest", "prompt_variant": "baseline"},
    )
    asyncio.run(hook.on_task_start(SimpleNamespace(eval_id="E1", run_id="R1", spec=spec, plan=None, eval_set_id=None)))
    beat = read_heartbeat(heartbeat_path("E1", live_dir))
    assert beat["total"] == 6
    assert beat["done"] == 0
    assert beat["modelTag"] == "night-runner:latest"

    summary = SimpleNamespace(id="GRAPH-003.1", epoch=1)
    asyncio.run(hook.on_sample_start(SimpleNamespace(eval_id="E1", run_id="R1", sample_id="s", summary=summary, eval_set_id=None)))
    beat = read_heartbeat(heartbeat_path("E1", live_dir))
    assert beat["current"]["sampleId"] == "GRAPH-003.1"

    sample = SimpleNamespace(id="GRAPH-003.1", epoch=1, error=None)
    asyncio.run(hook.on_sample_end(SimpleNamespace(eval_id="E1", run_id="R1", sample_id="s", sample=sample, eval_set_id=None)))
    beat = read_heartbeat(heartbeat_path("E1", live_dir))
    assert beat["done"] == 1
    assert beat["current"] is None
    assert beat["last"]["sampleId"] == "GRAPH-003.1"

    calls: list[Path] = []

    def fake_finalize(path: Path, **kwargs: object) -> dict:
        calls.append(path)
        return {"skipped": False, "html": "x.html", "pdf": None}

    monkeypatch.setattr(report_module, "finalize_log", fake_finalize)
    log = SimpleNamespace(location=str(tmp_path / "run.eval"), status="success")
    asyncio.run(hook.on_task_end(SimpleNamespace(eval_id="E1", run_id="R1", log=log, eval_set_id=None)))
    assert calls == [tmp_path / "run.eval"]
    assert not heartbeat_path("E1", live_dir).exists()

    # A failing finalize must not raise out of the hook.
    def broken(path: Path, **kwargs: object) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(report_module, "finalize_log", broken)
    asyncio.run(hook.on_task_end(SimpleNamespace(eval_id="E2", run_id="R1", log=log, eval_set_id=None)))


def test_criteria_cover_the_three_code_scorers() -> None:
    assert set(CRITERIA) == {"subtask_done", "reference_tests_pass", "process_budget"}
    for row in CRITERIA.values():
        assert row["what"]
        assert row["why"]
        assert row["how"]


def test_worktree_isolation_roundtrip() -> None:
    if not LIVING_GRAPH.exists():
        pytest.skip("Living Graph workspace is not on this machine")
    cases = build_cases()
    case = next(case for case in cases if case["id"] == "GRAPH-013.1")
    meta = case["metadata"]
    isolated = prepare_isolated_run(
        sample_id="GRAPH-013.1",
        task_id=meta["task_id"],
        base_commit=meta["base_commit"],
        repo=LIVING_GRAPH,
        source_requirements=AGENT_ROOT / "requirements" / "living-graph",
    )
    try:
        head = (isolated.worktree / ".git").exists() or (isolated.worktree / "src").exists()
        assert head
        assert not isolated.state_path.exists()
        dep_state = isolated.requirements_dir / "GRAPH-009.state.json"
        assert dep_state.exists()
        assert json.loads(dep_state.read_text(encoding="utf-8")).get("status") == "DONE"
        overlaid = overlay_reference_tests(
            isolated.worktree,
            LIVING_GRAPH,
            meta["accepted_commit"],
            meta["reference_tests"],
        )
        assert overlaid
        assert (isolated.worktree / overlaid[0]).exists()
    finally:
        isolated.cleanup()
    assert not isolated.worktree.exists()


def test_prepare_isolated_run_rejects_missing_base(tmp_path: Path) -> None:
    with pytest.raises(IsolationError):
        prepare_isolated_run(
            sample_id="X",
            task_id="X",
            base_commit="",
            repo=tmp_path,
            source_requirements=tmp_path,
        )
