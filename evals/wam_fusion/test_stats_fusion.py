"""Validate the paired-binary fusion statistics against known values.

Run: python3 test_stats_fusion.py  (or pytest). No scipy in the prod venv, so we
anchor the exact McNemar tail + Holm step-down against hand-computable cases.
"""

from __future__ import annotations

from stats_fusion import binom_two_sided_p, holm_correction, mcnemar


def test_binom_tail_known() -> None:
    # b=c -> perfectly balanced -> p = 1.0
    assert abs(binom_two_sided_p(5, 5) - 1.0) < 1e-9
    # all discordant in one direction, n=6: two-sided p = 2 * (0.5^6) = 2/64 = 0.03125
    assert abs(binom_two_sided_p(6, 0) - 2 * (0.5**6)) < 1e-9
    # n=0 discordant -> p=1
    assert binom_two_sided_p(0, 0) == 1.0
    # b=8,c=2 (n=10): two-sided exact ~ 0.109 (known McNemar value)
    p = binom_two_sided_p(8, 2)
    assert 0.10 < p < 0.115


def test_mcnemar_clear_win() -> None:
    # B fixes 9 items A got wrong, loses 0 -> strong, significant.
    a = [False] * 9 + [True] * 11
    b = [True] * 9 + [True] * 11
    r = mcnemar("test", a, b, n_boot=2000, seed=1)
    assert r.only_b_correct == 9 and r.only_a_correct == 0
    assert r.delta > 0.4
    assert r.p_exact < 0.01
    assert r.ci_low > 0  # CI excludes 0


def test_mcnemar_no_difference() -> None:
    # identical -> delta 0, p=1, CI contains 0
    a = [True, False, True, False, True, False, True, False]
    r = mcnemar("test", a, a, n_boot=2000, seed=1)
    assert r.delta == 0.0
    assert r.p_exact == 1.0
    assert r.ci_low <= 0 <= r.ci_high


def test_mcnemar_win_hiding_losses() -> None:
    # Aggregate accuracy up, but with REGRESSIONS — discordant cells must expose it.
    # A correct on items 0-9 (10), wrong 10-19. B correct on 5-14 (10), wrong else.
    a = [i < 10 for i in range(20)]
    b = [5 <= i < 15 for i in range(20)]
    r = mcnemar("test", a, b, n_boot=2000, seed=1)
    assert r.acc_a == r.acc_b  # same aggregate accuracy (10/20 each)
    assert r.only_b_correct == 5 and r.only_a_correct == 5  # 5 wins, 5 losses both visible
    assert r.delta == 0.0


def test_holm_step_down() -> None:
    # 3 comparisons. Holm: sort asc, multiply by (m - i).
    pvals = [("c1", 0.01), ("c2", 0.04), ("c3", 0.20)]
    out = holm_correction(pvals, alpha=0.05)
    # c1: 0.01*3 = 0.03 < 0.05 reject
    assert out["c1"]["reject"] and abs(out["c1"]["p_holm"] - 0.03) < 1e-9
    # c2: 0.04*2 = 0.08 >= 0.05 -> not rejected; stops the chain
    assert not out["c2"]["reject"] and abs(out["c2"]["p_holm"] - 0.08) < 1e-9
    # c3: 0.20*1 = 0.20, monotone >= 0.08, not rejected
    assert not out["c3"]["reject"]


def test_holm_all_reject() -> None:
    out = holm_correction([("a", 0.001), ("b", 0.002), ("c", 0.003)], alpha=0.05)
    assert all(out[k]["reject"] for k in ("a", "b", "c"))


def test_holm_monotone_enforced() -> None:
    # An out-of-order case where raw adj would dip — monotonicity must hold.
    out = holm_correction([("a", 0.02), ("b", 0.04)], alpha=0.05)
    # a: 0.02*2=0.04 ; b: 0.04*1=0.04 ; both equal, monotone preserved
    assert out["a"]["p_holm"] <= out["b"]["p_holm"]


if __name__ == "__main__":
    import sys
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
