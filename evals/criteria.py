"""Plain-language criteria shared by scorers, the HTML report, and LEARNING.md.

Keep thresholds here so a report can quote them without reading scorer code.
"""

from __future__ import annotations

from typing import Any

PROCESS_BUDGET: dict[str, Any] = {
    "max_attempts": 10,  # test-writer 5 + implementer 5
    "max_steps": 160,
    "max_repeat_of_one_call": 5,
    "max_tool_error_rate": 0.40,
}

CRITERIA: dict[str, dict[str, Any]] = {
    "subtask_done": {
        "what": "The orchestrator marked the pinned subtask DONE after the independent verifier passed.",
        "why": "This is the outcome that matters: the agent's own declared verification succeeded and the coordinator agreed. A fluent transcript that never reaches DONE is still a failure.",
        "how": "Code. Read the copied requirements <task_id>.state.json after the run. Pass iff the matching subtask status is DONE.",
        "grader": "code",
    },
    "reference_tests_pass": {
        "what": "Tests from the historically accepted commit still pass against the code the agent just wrote.",
        "why": "The agent's own tests can be too weak or accidentally rewritten. Overlaying the accepted commit's test files is a cheap substitute for hidden tests. These are regression tests, not independently authored hidden tests.",
        "how": "Code. After the agent finishes, overlay tests/ files from accepted_commit, then run pytest on those paths in the worktree.",
        "grader": "code",
    },
    "process_budget": {
        "what": "The agent finished without looping, blowing the step budget, or drowning in tool errors.",
        "why": "Local models often fail on process before they fail on correctness. A pass that took 80 identical read_file calls is not a prompt you want to keep.",
        "how": (
            "Code, from run_summary.json. Fail if attempts > {max_attempts}, "
            "steps > {max_steps}, any identical tool fingerprint repeats more than "
            "{max_repeat_of_one_call} times, or tool error rate > {max_tool_error_rate:.0%}."
        ).format(**PROCESS_BUDGET),
        "grader": "code",
        "thresholds": PROCESS_BUDGET,
    },
}
