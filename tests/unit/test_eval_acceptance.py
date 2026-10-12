"""The gate's acceptance tooling: the probe corpus puts each rule where only the default
judge can see it, and the R0 candidates are valid workflow scripts."""

from __future__ import annotations

import pytest
from evals.acceptance import candidates, probe_corpus
from evals.workflows import eval_item, load

from aios.workflows.script_validation import validate_workflow_script

_REF = {"id": "wf_x", "version": 1}
ITEM = load(eval_item.build(r0=_REF, judge=_REF))


def test_every_turn_but_the_last_two_sets_a_rule_for_two_turns_later() -> None:
    pool = ["What is 2 + 2?"]
    turns = [probe_corpus.turn_text("s", t, pool) for t in range(1, probe_corpus.TURNS + 1)]
    with_rule = [i + 1 for i, text in enumerate(turns) if "two messages from now" in text]
    assert with_rule == list(range(1, probe_corpus.TURNS - 1))
    assert all(text.startswith("Problem:\n") for text in turns)


@pytest.mark.parametrize("turn", [5, 6])
def test_the_rule_is_inside_the_default_tail_and_outside_a_tail_of_three(turn: int) -> None:
    """At turn ``turn`` the governing rule was set at turn ``turn - 2``. The gate's own
    window helpers put it inside the default judge's 8-message tail, outside a 3-message
    tail, and outside the control's view (the last user turn); and the control item is
    eligible (control view < judge tail < whole request)."""
    pool = ["What is 2 + 2?"]
    rest: list[dict[str, str]] = []
    for t in range(1, turn + 1):
        rest.append({"role": "user", "content": probe_corpus.turn_text("s", t, pool)})
        if t < turn:
            rest.append({"role": "assistant", "content": "4\nANSWER: 4"})
    rule_turn = rest[2 * (turn - 3)]
    assert "two messages from now" in rule_turn["content"]

    default = ITEM["judge_tail"](rest, 8)
    blind = ITEM["judge_tail"](rest, 3)
    control = rest[ITEM["last_user"](rest) :]
    assert rule_turn in default
    assert rule_turn not in blind
    assert rule_turn not in control
    assert len(control) < len(default) < len(rest)
    assert len(control) < len(blind) < len(rest)


def test_the_sessions_are_drawn_deterministically_from_the_reasoning_suites() -> None:
    pool = probe_corpus.problems()
    assert len(pool) == 310
    assert probe_corpus.turn_text("k", 3, pool) == probe_corpus.turn_text("k", 3, pool)


@pytest.mark.parametrize("name", sorted(candidates.CANDIDATES))
def test_each_candidate_is_a_valid_workflow_with_no_tools(name: str) -> None:
    model, max_tokens, _ = candidates.CANDIDATES[name]
    validate_workflow_script(candidates.R0 % {"model": model, "max_tokens": max_tokens}, [])
