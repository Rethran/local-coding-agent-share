# TDD coder evals

Assess properties of orchestrator runs over Living Graph subtasks. Pass rates,
not exact strings. Model under test: `night-runner:latest`.

## What a case is

`evals/cases/living-graph.jsonl` is built from git history of the Living Graph
workspace. Each `GRAPH-N.M: title` commit becomes one case: start from the
parent commit, run `--from GRAPH-N.M --to GRAPH-N.M`, score DONE / gold tests /
process budget.

Rebuild:

```powershell
python -m evals build-dataset
```

Tags: `smoke` (3 cases, plumbing), `regression` (~12 cases). GRAPH-051.1 is
historically BLOCKED and never committed, so it is not in the jsonl. Untagged
rows are the held-out set — do not iterate prompts against them.

Reference tests come from the accepted commit. They are regression tests, not
independently authored hidden tests.

## Isolation

Every sample gets:

- a detached git worktree at `base_commit`
- a copy of `requirements/living-graph/` with only that task's `.state.json`
  removed (dependencies stay DONE)
- a fresh `LOCAL_AGENT_RUNTIME_DIR` so handoffs and continuity cannot leak
- `--no-auto-push` and `--run-summary`

## Run

The task file inserts the agent root onto `sys.path`, so this works from the
repo root:

```powershell
inspect eval evals/tdd_coder_task.py `
  -T tags=smoke `
  -T prompt_variant=baseline `
  -T model_tag=night-runner:latest `
  --epochs 1 `
  --max-samples 1 `
  --max-connections 1 `
  --model mockllm/model `
  --log-dir evals/logs
```

Plumbing without a GPU (worktrees + scorers, no Ollama):

```powershell
python -m evals run --tags smoke --epochs 1 --dry-run
inspect eval evals/tdd_coder_task.py -T tags=smoke -T dry_run=true --epochs 1 --max-connections 1 --model mockllm/model --log-dir evals/logs
```

Overnight baseline (one GPU, do not parallelize):

```powershell
inspect eval evals/tdd_coder_task.py `
  -T tags=regression `
  -T prompt_variant=baseline `
  -T model_tag=night-runner:latest `
  --epochs 3 `
  --max-connections 1 `
  --model mockllm/model `
  --log-dir evals/logs
```

Prompt A/B (change one clause: `orchestrator/prompt_variants/v2.json`):

```powershell
inspect eval evals/tdd_coder_task.py -T tags=regression -T prompt_variant=v2 --epochs 3 --max-connections 1 --model mockllm/model --log-dir evals/logs
```

Fallback runner if Inspect is not installed:

```powershell
python -m evals run --tags smoke --epochs 1 --max-samples 1 --model night-runner:latest
```

## Plumbing that already ran

Inspect AI 0.3.273 is installed. A dry-run of the 3 smoke cases through
`inspect eval ... -T dry_run=true` finished in 8s: `subtask_done` 0%,
`reference_tests_pass` 0%, `process_budget` 100%. That is the correct
signature for "isolation works, the model did not run." The log is under
`evals/logs/`. Open it with `inspect view`. The HTML report is
`evals/reports/inspect-smoke-dry.html`.

`--compare` of two dry-runs (baseline vs v2) reports **noise** on every
scorer, which is the point of the check: a prompt change cannot move the
score if the solver never called Ollama.

A live `night-runner:latest` baseline (12 cases × 3 epochs) is several hours
on one GPU. Use the overnight command below. Do not raise `--max-connections`.

## Arithmetic example

A real red-then-green run, no model and no GPU. Each try copies
`evals/examples/laskin`, runs pytest (the stub fails), writes an
implementation, and runs pytest again. summa passes all three tries,
erotus fails the third, tulo never passes but is marked done.

```powershell
python -m evals example
```

Open `evals/reports/esimerkki-summa.html`. `evals/examples/summa.json` is the
same story with fixed one-minute timings for the unit test. The command
above replaces the HTML with the timings from the pytest you just ran.

## Report

```powershell
python -m evals report evals/logs\<run>.eval --out evals/reports/baseline-night-runner
python -m evals report evals/logs\<run>.eval --out evals/reports/baseline-night-runner --pdf
python -m evals report --compare evals/reports\baseline.json evals/reports\v2.json
```

HTML + JSON land in `evals/reports/`. The page leads with the reviewer
block, then the maker block. Criteria text comes from `evals/criteria.py`.

**Katselmoijalle:** consistency (reliable / fragile / cannot by how many
epochs `reference_tests_pass` succeeded), GPU-minutes per case that passed
on every epoch, and trend versus the previous report. A delta inside the
combined stderr is noise.

**Tekijälle:** false-done rate, median and worst wall time and step count,
and one primary failure label per failed trial. False-done is not a new
scorer. It is `subtask_done` true and `reference_tests_pass` false, divided
by trials where the agent said DONE.

`--pdf` prints that HTML with Edge or Chrome (`--headless --print-to-pdf`).
If neither is installed, the HTML is still A4 print-ready and the command
says so.

Logs finished before totals were stored in the `.eval` file still produce
false-done, consistency, GPU-minutes, and trend. Step counts show as
unknown when the log has no `process_budget` check metadata.

## Dashboard, index, PDF on completion

```powershell
python -m evals dashboard            # http://127.0.0.1:8765/
python -m evals finalize evals/logs\<run>.eval [--pdf-always] [--force]
python -m evals index                # rebuild evals/reports/index.html
```

`inspect view` already browses transcripts. The dashboard shows what it
does not: trials done against the planned count (samples × epochs), ETA
from the median trial so far, the current sample and how long it has been
running, consistency buckets and false-done so far, and links to each
run's HTML / PDF / JSON. The page polls `/api/runs` every 5 s; the server
re-reads `evals/logs/*.eval` every 10 s (`--interval`). Binds localhost only.

The task file imports `evals/hooks.py`, so every `inspect eval` of
`tdd_coder_task.py` registers a hook that:

- writes a heartbeat to `evals/logs/.live/<eval_id>.json` on each sample
  start and end (the dashboard reads the current sample from it);
- on task end calls `finalize_log()`: HTML + JSON at
  `evals/reports/<log name>.{html,json}`, a PDF next to them when the run
  status is `success`, then rebuilds `index.html`.

Cancelled or errored runs get HTML + JSON with a yellow "ajo ei
valmistunut" notice and no PDF. `finalize --pdf-always` overrides that.
Ctrl+C does not fire the task-end hook; the dashboard poller finalizes any
finished log that has no report yet (`--no-finalize` turns that off), so
the report appears the next time the dashboard runs.

`evals/reports/index.html` is one row per `eval-report/v1` JSON: model,
prompt variant, run status, trials/planned, reference-test pass rate ±
stderr, false-done, reliable cases, GPU-minutes per success, file links.
Dry-run reports are labelled "kuivaharjoitus".

If `inspect.exe` is blocked by an application-control policy,
`python -m inspect_ai eval ...` is the same command.

## Education

Read [LEARNING.md](LEARNING.md) in order. Each chapter points at a file and
asks one question you can answer from the data.
