"""Unit tests for the execution-verified-selection P0 pure logic (no network, no spend).

Run standalone:  python3 test_selection.py   (or pytest test_selection.py)
"""

from __future__ import annotations

import sys

from coding_prompt import build_prompt
from selection_arms import (
    build_judge_prompt,
    canary_tokens,
    changed_lines,
    leak_scan,
    one_sided_sign_test,
    parse_judge_pick,
    pick_exec_verified,
    pick_index0,
    pick_seeded_random,
    unified_diff_text,
)


def test_build_prompt_new_file_item():
    """A new-file-creation item has an EMPTY pre-PR snapshot (the s01/gate_reaper
    confirmatory-run crash): build_prompt must name the file(s) to create instead
    of raising StopIteration on next(iter({}))."""
    p = build_prompt("create the gate reaper", {}, expected_paths=["src/aios/workflows/gate_reaper.py"])
    assert "FILE(S) TO CREATE" in p
    assert "src/aios/workflows/gate_reaper.py" in p
    assert "CURRENT SOURCE FILE(S)" not in p
    assert "```src/aios/workflows/gate_reaper.py" in p  # the example info-string line
    # existing-file behavior unchanged
    p2 = build_prompt("fix it", {"src/x.py": "a\n"})
    assert "CURRENT SOURCE FILE(S)" in p2 and "FILE(S) TO CREATE" not in p2
    assert "```src/x.py" in p2
    # mixed: shown files + a to-create file
    p3 = build_prompt("t", {"src/x.py": "a\n"}, expected_paths=["src/x.py", "src/y.py"])
    assert "FILE(S) TO CREATE" in p3 and "- src/y.py" in p3
    # caller bug is loud
    try:
        build_prompt("t", {})
        raise AssertionError("expected ValueError on empty base + no expected paths")
    except ValueError:
        pass


def test_changed_lines_and_diff_for_new_file_candidate():
    """Empty base: diff is vs the empty file, so the tie-break stays meaningful
    and the judge sees an all-added diff (not 'no usable patch')."""
    cand = {"src/new.py": "a\nb\nc\n"}
    assert changed_lines({}, cand) == 3
    assert changed_lines({}, {}) >= 10**9  # no-output candidate keeps the sentinel
    d = unified_diff_text({}, cand)
    assert "b/src/new.py" in d and "+a" in d and "+c" in d


def test_index0():
    assert pick_index0(4) == 0


def test_seeded_random_deterministic_and_uniformish():
    a = pick_seeded_random("c05", 4, 123)
    assert a == pick_seeded_random("c05", 4, 123), "same seed+item must reproduce"
    assert 0 <= a < 4
    picks = {pick_seeded_random(f"i{k}", 4, 123) for k in range(64)}
    assert picks == {0, 1, 2, 3}, "all indices reachable across items"
    assert pick_seeded_random("c05", 4, 123) is not None
    b = pick_seeded_random("c05", 4, 456)
    assert isinstance(b, int)  # different seed may differ; just typed sanity


def test_changed_lines_and_missing_candidate():
    base = {"src/x.py": "a\nb\nc\n"}
    assert changed_lines(base, {"src/x.py": "a\nb\nc\n"}) == 0
    assert changed_lines(base, {"src/x.py": "a\nB\nc\n"}) == 2  # one - one +
    assert changed_lines(base, {}) >= 10**9
    assert changed_lines(base, {"src/x.py": "  "}) >= 10**9


def test_pick_exec_verified_tiebreak_and_no_passer():
    # passers 1 and 3; 3 has fewer changed lines -> 3 wins
    idx, detail = pick_exec_verified([False, True, False, True], [5, 9, 2, 4])
    assert idx == 3 and "passer" in detail
    # tie on changed lines -> lowest index
    idx, _ = pick_exec_verified([True, True, False, False], [4, 4, 1, 1])
    assert idx == 0
    # no passer -> deterministic fewest-changed-lines fallback
    idx, detail = pick_exec_verified([False, False, False, False], [5, 2, 9, 7])
    assert idx == 1 and "NO PASSER" in detail


def test_judge_prompt_never_contains_tests_and_parse():
    prompt = build_judge_prompt("fix the off-by-one", ["--- a/src/x.py\n+++ b/src/x.py\n-a\n+b", ""])
    assert "CANDIDATE 0" in prompt and "CANDIDATE 1" in prompt
    assert "test" not in prompt.lower().replace("fix the off-by-one", "")  # no test material
    assert parse_judge_pick("blah\nWINNER: 1", 4) == (1, "judge picked 1")
    assert parse_judge_pick("WINNER: 0\n... final WINNER: 3", 4)[0] == 3  # last wins
    assert parse_judge_pick("I like candidate two", 4)[0] is None
    assert parse_judge_pick("WINNER: 9", 4)[0] is None
    assert parse_judge_pick(None, 4)[0] is None


def test_canary_tokens_and_leak_scan():
    oracle = "import x\n\ndef test_attenuation_clamps():\n    pass\n\nclass TestEdge:\n    def test_zero(self):\n        pass\n"
    toks = canary_tokens(
        ["tests/unit/test_attenuation.py"], [oracle],
        {"src/aios/models/attenuation.py": "def clamp(): ..."},
        "fix clamping in attenuation",
    )
    assert "tests/unit/test_attenuation.py" in toks
    assert "test_attenuation.py" in toks
    assert "test_attenuation_clamps" in toks and "TestEdge" in toks and "test_zero" in toks
    # a candidate that names the hidden test file = leak
    hits = leak_scan("I checked tests/unit/test_attenuation.py and ...", toks)
    assert "tests/unit/test_attenuation.py" in hits
    assert leak_scan("clean candidate content", toks) == []
    # tokens visible in the model's INPUT are excluded (not evidence of leak)
    toks2 = canary_tokens(
        ["tests/unit/test_attenuation.py"], [oracle],
        {"src/x.py": "see test_zero used here"},
        "please also create tests/unit/test_attenuation.py",
    )
    assert "tests/unit/test_attenuation.py" not in toks2
    assert "test_zero" not in toks2
    assert "test_attenuation_clamps" in toks2  # never shown to the model -> still a canary


def test_ts_canary_identifiers():
    oracle = 'describe("agent draft form", () => { it("rejects an empty model id", () => {}) })'
    toks = canary_tokens(["src/components/agents/agent-draft.test.ts"], [oracle], {}, "task")
    assert "agent-draft.test.ts" in toks
    assert "rejects an empty model id" in toks
    assert "agent draft form" in toks


def test_one_sided_sign_test():
    # structure: discordants all favor c -> small p at k>=5
    b = [False] * 5 + [True] * 3
    c = [True] * 5 + [True] * 3
    r = one_sided_sign_test("t", b, c)
    assert r.n_discordant == 5 and r.c_only == 5 and r.b_only == 0
    assert abs(r.p_one_sided - 0.03125) < 1e-9, r.p_one_sided
    # no discordance -> p=1
    r2 = one_sided_sign_test("t", [True, False], [True, False])
    assert r2.p_one_sided == 1.0
    # a b-only win raises p (flakiness tracked, not hidden)
    r3 = one_sided_sign_test("t", [True, False, False], [False, True, True])
    assert r3.b_only == 1 and r3.c_only == 2 and r3.p_one_sided > 0.03125


def test_unified_diff_text_shape():
    d = unified_diff_text({"src/x.py": "a\nb\n"}, {"src/x.py": "a\nc\n"})
    assert "a/src/x.py" in d and "b/src/x.py" in d and "+c" in d and "-b" in d
    assert unified_diff_text({"src/x.py": "a\n"}, {"src/x.py": "a\n"}) == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL {fn.__name__}: {e}")
    print(f"{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
