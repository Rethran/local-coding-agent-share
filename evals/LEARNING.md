# How evals work in this repo

This file is the education track. Each chapter matches one implementation
step. After a chapter, answer the question by looking at the artifact it
points to — not by guessing.

Evals here do **not** assert an exact model string. They assess properties of
a full TDD-coder run (done / tests / process) over a dataset of subtasks, and
report pass rates with error bars.

---

## 1. Isolation and observability before grading

**Concept.** A number is only as honest as the run that produced it. Two
things have to be true before any scorer is worth writing:

1. Epoch 2 cannot see epoch 1. Continuity packets, handoffs, recovery
   sessions, and leftover files would otherwise inflate the pass rate.
2. Process metrics cannot be scraped from stdout. Scorers need a file.

**Where it lives.**

- `tools/_handoffs.runtime_root()` and `orchestrator.config.ATTEMPT_REPORTS_DIR`
  honor `LOCAL_AGENT_RUNTIME_DIR`. Default is still the agent repo root, so
  interactive use is unchanged.
- `orchestrator/run_metrics.py` collects steps, tool fingerprints, tokens,
  and verifier results. `python main.py --run-summary PATH` writes them.
- `orchestrator/prompt_variants.py` plus `orchestrator/prompt_variants/*.json`
  let you change clauses without editing `prompts.py`. `baseline.json` is
  empty on purpose: the snapshot in
  `tests/fixtures/prompt_baseline_snapshot.json` must stay byte-identical.

**Look at.** `tests/test_runtime_dir.py`, `tests/test_run_metrics.py`,
`tests/test_prompt_variants.py`.

**Question.** If you ran two eval epochs without `LOCAL_AGENT_RUNTIME_DIR`,
which file under `recovery/` would leak the first epoch into the second
prompt, and why does that make `subtask_done` easier the second time?

---

## 2. A case is input + properties, not a golden string

**Concept.** One eval case is one historically accepted subtask commit.

- `input` — the spec the agent sees (`--from GRAPH-047.1 --to GRAPH-047.1`).
- `base_commit` — parent SHA; the worktree starts here, *without* the
  accepted implementation.
- `accepted_commit` — the SHA that landed. Its `tests/` files are the
  reference tests.
- `tags` — `smoke` (3 cases, plumbing) and `regression` (~12 cases).
  GRAPH-051.1 is historically BLOCKED and has no landing commit, so it is
  not in the jsonl yet.

These reference tests were written by the agent that originally landed the
commit. They are **regression tests**, not hidden tests. A pass means "the
new run matches what a previous run thought was correct," not "an independent
author would accept this." Hidden tests come later, and only for cases that
look suspiciously good.

**Where it lives.** `evals/build_dataset.py` walks
`git log` in the Living Graph workspace. Output:
`evals/cases/living-graph.jsonl`.

**Look at.** One jsonl line for `GRAPH-047.1`. Check that `base_commit` is
the parent of `accepted_commit` (`git -C living-3d-graph rev-parse accepted^`).

**Question.** Why is a 48-attempt task with seven subtasks a bad eval
unit, and why is `GRAPH-047.1` a better one?

---

## 3. Dataset → solver → scorer, and why epochs exist

**Concept.** Every eval framework is the same three objects:

- **Dataset** — the cases.
- **Solver** — "run my app." Here: git worktree + env overrides +
  `python main.py`. Inspect's `generate()` is unused; Ollama is inside the
  orchestrator. Pass `--model mockllm/model` so Inspect does not load a
  second copy of the weights.
- **Scorer** — code checks, not a judge. `subtask_done`,
  `reference_tests_pass`, `process_budget`.

Local models are stochastic. `epochs=3` runs each case three times and
reports a pass rate, not a single pass/fail. `accuracy()` is the mean;
`stderr()` is √(p(1−p)/n). A prompt that is "3 points better" with stderr
8 points is noise. Change one variable at a time (`prompt_variant` **or**
`model_tag`, never both) and keep `--max-samples 1` / `--max-connections 1`
on one GPU.

**Where it lives.**

- Solver/isolation: `evals/isolation.py`, `evals/tdd_coder_task.py`
- Graders: `evals/scorers.py`, thresholds in `evals/criteria.py`
- Fallback without Inspect: `python -m evals run --tags smoke --dry-run`

**Look at.** After a dry-run, open the report JSON and find `n` for
`subtask_done`. If you had `epochs=3` and 3 smoke cases, `n` should be 9.

**Question.** The solver copies all of `requirements/living-graph/` but
deletes only `<task_id>.state.json`. Why must GRAPH-010.state.json stay
DONE when evaluating GRAPH-047.1?

---

## 4. A report a person can read without opening the code

**Concept.** The report has four sections on purpose:

1. **Summary** — pass rate per criterion, stderr, delta vs the last report,
   p95 latency, tokens. Not one blended score.
2. **Criteria in plain language** — copied from `evals/criteria.py` so the
   text cannot drift from the grader.
3. **Failures first** — input, failed criterion, scorer explanation.
4. **Flaky** — same sample, mixed epochs (e.g. 2/5). These are usually the
   most informative trials; read those transcripts first.

`--compare a.json b.json` prints per-scorer deltas and marks a change as
**noise** when |Δ| ≤ combined stderr.

The same page now opens with reviewer numbers, then maker numbers.
Reviewer: which cases are reliable (reference tests passed every epoch),
fragile, or cannot; GPU-minutes per fully successful case, with failed
trials still in the numerator; trend of reference-test pass rate, false-done
rate, and reliable-case count, marked noise when the delta sits inside the
combined stderr. Maker: false-done rate, median and worst time and steps,
and one primary failure type so the counts add up. False-done is not a new
scorer — it is the two existing scores, `subtask_done` true and
`reference_tests_pass` false. Old logs still answer the reviewer questions.
Steps stay blank until a run copies `run_summary` totals into sample
metadata before the worktree is deleted.

The page a person opens leads with three sentences for the reviewer: which tasks to assign, what one success cost, and whether the change is larger than the sampling wobble. Under that, three sentences for the maker: whether a done-claim can be trusted, whether the wait is usable, and whether the fix is the model or the tools. The tables under those sentences are the evidence. `evals/examples/summa.json` is the small input you can render to see this without Living Graph: one question, three tries, the check is whether the number is right.

The dashboard (`python -m evals dashboard`) runs these same functions over
a log that is still being written. That is the whole trick: there is no
second set of "live" metrics. A case shows as reliable after one passing
epoch and can drop to fragile when epoch 2 lands; ETA is remaining trials
times the median so far, so a slow case in the middle moves it a lot; the
false-done denominator is tiny early on. Treat the live numbers as a
progress bar, and the report written at task end (HTML, JSON, PDF when the
run status is `success`) as the result.

**Where it lives.** `evals/report.py`, template
`evals/templates/report.html.j2`. Command:
`python -m evals report <log-or-json>`.

**Look at.** `evals/reports/*.html` after a dry-run. Confirm the criteria
paragraphs match `evals/criteria.py`.

**Question.** Why would averaging `subtask_done` and `process_budget` into
one "eval score" hide a regression you actually care about?

---

## 5. Change one variable, then decide if the delta is real

**Concept.** Baseline is `prompt_variants/baseline.json` (empty) +
`night-runner:latest` + the regression tag set + 3 epochs. `v2.json`
overrides only `system.tdd`. If pass rate moves by less than combined
stderr, keep baseline. If it moves on `process_budget` but not
`subtask_done`, you learned something specific: the sermon was wasting
steps, not buying correctness.

Do not look at the held-out cases (untagged jsonl rows) while iterating on
v2. That is how you overfit the prompt to the eval set.

Inspect's log viewer (`inspect view`) is for reading a single transcript.
The HTML report is for comparing runs.

**Where it lives.** `orchestrator/prompt_variants/v2.json`. Commands in
`evals/README.md`.

**Look at.** `python -m evals report --compare` on two dry-run JSONs of the
same cases. Every row should say `noise` and `delta=0` if nothing changed
but the filename.

**Question.** The accepted GRAPH commits were produced under a profile that
named `qwen3:8b`. Tonight's baseline uses `night-runner:latest`. If a case
fails now, is that a harness bug, a prompt regression, or a model
difference — and how does the run summary let you tell?
