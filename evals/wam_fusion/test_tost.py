"""Validate the pure-Python TOST implementation against known values.

Run: python -m pytest evals/wam_fusion/test_tost.py  (or execute directly).
No scipy in the prod venv, so we check the Student-t CDF + the TOST decision
against analytically-known anchors and a worked equivalence example.
"""

from __future__ import annotations

import math

from tost import paired_tost, student_t_cdf


def test_t_cdf_symmetry_and_median() -> None:
    # t-CDF is 0.5 at 0 for any df, and symmetric: F(-t) = 1 - F(t).
    for df in (1, 5, 30, 100):
        assert abs(student_t_cdf(0.0, df) - 0.5) < 1e-9
        assert abs(student_t_cdf(-1.3, df) - (1.0 - student_t_cdf(1.3, df))) < 1e-9


def test_t_cdf_known_critical_values() -> None:
    # Two-sided 95% t critical values (upper-tail 0.975): df=10 -> 2.228, df=inf -> 1.96.
    assert abs(student_t_cdf(2.228, 10) - 0.975) < 1e-3
    assert abs(student_t_cdf(1.960, 1_000_000) - 0.975) < 1e-3
    # df=1 (Cauchy): F(1) = 0.75.
    assert abs(student_t_cdf(1.0, 1) - 0.75) < 1e-6


def test_tost_clearly_equivalent() -> None:
    # Tiny noise around 0 with a generous margin -> equivalence accepted.
    diffs = [0.01, -0.02, 0.0, 0.015, -0.01, 0.005, -0.005, 0.02, -0.015, 0.0]
    r = paired_tost(diffs, margin=0.5, alpha=0.05)
    assert r.equivalent
    assert r.p_tost < 0.05
    assert r.ci_low > -0.5 and r.ci_high < 0.5


def test_tost_clearly_not_equivalent() -> None:
    # Differences centered far from 0 relative to a tight margin -> not equivalent.
    diffs = [1.0, 1.1, 0.9, 1.05, 0.95, 1.0, 1.02, 0.98, 1.0, 1.0]
    r = paired_tost(diffs, margin=0.1, alpha=0.05)
    assert not r.equivalent
    assert r.p_tost >= 0.05


def test_tost_zero_variance_inside_margin() -> None:
    # All diffs identical and inside margin -> trivially equivalent.
    r = paired_tost([0.0] * 8, margin=1.0)
    assert r.equivalent and r.sd_diff == 0.0


def test_tost_zero_variance_outside_margin() -> None:
    r = paired_tost([2.0] * 8, margin=1.0)
    assert not r.equivalent


def test_tost_borderline_matches_manual() -> None:
    # Worked example: n=20, mean diff = 0.10, sd = 0.20, margin = 0.30, alpha=0.05.
    # se = 0.20/sqrt(20) = 0.044721; df=19.
    # t_lower = (0.10+0.30)/0.044721 = 8.944 (p_lower ~ 0)
    # t_upper = (0.10-0.30)/0.044721 = -4.472 -> p_upper = F(-4.472, 19) ~ 0.00013
    # => equivalent.
    import random

    random.seed(0)
    # Construct a sample with the target mean/sd exactly.
    base = [-1, -0.5, 0, 0.5, 1] * 4  # mean 0, known spread
    m = sum(base) / len(base)
    s = math.sqrt(sum((x - m) ** 2 for x in base) / (len(base) - 1))
    diffs = [0.10 + (x - m) * (0.20 / s) for x in base]
    r = paired_tost(diffs, margin=0.30, alpha=0.05)
    assert abs(r.mean_diff - 0.10) < 1e-9
    assert abs(r.sd_diff - 0.20) < 1e-9
    assert r.equivalent
    assert r.p_upper < 0.01


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
