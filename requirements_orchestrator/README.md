# Vision-to-Requirements Orchestrator

This package is a separate, single-model pipeline that turns an interactive
project vision into task JSON consumed by the existing code orchestrator. It
does not execute generated tasks and does not import the existing
`orchestrator` or `tools` packages.

## Prerequisites

- A running Ollama server
- A local Qwen model, defaulting to `qwen3:8b`
- Python packages from `requirements_orchestrator/requirements.txt`
- An existing target workspace directory

## Start a new run

```powershell
python -m requirements_orchestrator
```

The CLI asks one question at a time. Separate list values with semicolons.
Every answer and accepted draft task is persisted under
`requirements_orchestrator/runs/<project-slug>/`.

Useful options:

```powershell
python -m requirements_orchestrator --model qwen3:8b --num-ctx 16384
python -m requirements_orchestrator --yes-publish
```

## Resume

```powershell
python -m requirements_orchestrator --resume <project-slug>
```

Rejected proposals and their exact validator errors are retained and included
in the next one-task decomposition request.

## Output

Publishing is explicit and refuses to overwrite existing output. It creates:

- `requirements/<project-slug>/<PREFIX>-001.json` and later tasks
- `projects/<project-slug>.json`

Generation-only acceptance coverage metadata remains in the run state and is
not copied into published task files. No runtime `.state.json` files are
created; the code orchestrator owns those.

After publication, use the commands printed by the CLI:

```powershell
$env:LOCAL_AGENT_PROFILE = "<project-slug>"
python main.py --from <PREFIX>-001 --to <PREFIX>-NNN
```

## Guarantees and limits

- Ollama JSON Schema output is mandatory; malformed free-form output is not
  accepted.
- `dependsOn` is never model-authored. It is inferred from prior target and
  related artifact references.
- Completion is rejected until all elicited acceptance signals are covered.
- Verification commands must use a supported runner and identify a focused
  test artifact or selector.
- Acceptance coverage is structurally enforced, but its semantic correctness
  still depends on the local model and human review before publishing.

## Tests

```powershell
python -m pytest requirements_orchestrator/tests -q
```
