"""Paired-binary statistics for the WaM fusion MEASURING harness — pure stdlib.

The foundation's ``tost.py`` proves R0 ≈ native (a continuous-metric *equivalence*
test). Phase A asks the opposite question on a BINARY correctness metric: does
heterogeneous R1 *beat* the matched-fan-in baseline (self-fusion of the best single
model at equal chain depth), past a pre-declared MDE, with the family-wise error
rate controlled across the comparisons?

The red-team's non-negotiables this module enforces:

  * **Matched-fan-in, not a lone sample.** The caller measures three conditions
    (best-single / self-fusion-of-best-single / heterogeneous-R1) on the SAME
    task set. The headline comparison is R1 vs self-fusion (equal compute/structure),
    NOT R1 vs a single call — an aggregate win over a single call is confounded by
    fan-in and is exactly the fraud to avoid.
  * **Paired** tests (McNemar): every condition runs the SAME items, so we test the
    per-item correctness *difference*, which removes item-difficulty variance and is
    far more powerful than comparing two independent accuracy rates.
  * **FWER control (Holm).** Multiple pairwise comparisons inflate the false-positive
    rate; Holm-Bonferroni controls it without assuming independence.
  * **Per-item deltas including LOSSES.** The caller reports the discordant cells
    (b, c) so an aggregate win that hides regressions is visible (a routing bug).
  * **A pre-declared MDE.** "Not significant" at this n could just be underpower;
    the caller states the minimum detectable effect up front and the report says
    whether the observed CI excludes / includes it.

No scipy in the prod venv, so the binomial tail (exact McNemar) and the normal
approx are implemented directly. Validated in test_stats_fusion.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


# ── exact binomial tail (for exact McNemar) ──────────────────────────────────
def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1 - p) ** (n - k))


def binom_two_sided_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant pair (b, c).

    Under H0 (the two conditions are equally likely to flip a discordant item),
    b ~ Binomial(n=b+c, p=0.5). The two-sided p is the probability of an outcome
    at least as extreme (in either tail) as observed. b+c == 0 → p = 1.0 (no
    discordance, no evidence either way).
    """
    n = b + c
    if n == 0:
        return 1.0
    obs = _binom_pmf(b, n, 0.5)
    # Sum probabilities of all outcomes no more likely than the observed one
    # (the standard exact two-sided "method of small p-values").
    p = sum(
        _binom_pmf(k, n, 0.5) for k in range(n + 1) if _binom_pmf(k, n, 0.5) <= obs * (1 + 1e-9)
    )
    return min(1.0, p)


@dataclass(frozen=True)
class McNemarResult:
    name: str
    n: int  # paired items
    acc_a: float  # accuracy of condition A
    acc_b: float  # accuracy of condition B
    delta: float  # acc_b - acc_a  (B is the "treatment", A the "baseline")
    both_correct: int
    only_a_correct: int  # c: A right, B wrong  (B's losses vs A)
    only_b_correct: int  # b: B right, A wrong  (B's wins vs A)
    both_wrong: int
    p_exact: float  # exact McNemar two-sided p (uncorrected)
    ci_low: float  # bootstrap CI on delta
    ci_high: float

    def summary(self) -> str:
        return (
            f"{self.name}: B-A delta={self.delta:+.3f}  "
            f"(accA={self.acc_a:.3f} accB={self.acc_b:.3f}, n={self.n})\n"
            f"    discordant: B-wins(b)={self.only_b_correct}  B-losses(c)={self.only_a_correct}  "
            f"both✓={self.both_correct} both✗={self.both_wrong}\n"
            f"    McNemar exact p={self.p_exact:.4g}   95% bootstrap CI on delta: "
            f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}]"
        )


def mcnemar(
    name: str,
    correct_a: list[bool],
    correct_b: list[bool],
    *,
    n_boot: int = 10000,
    seed: int = 12345,
) -> McNemarResult:
    """Paired McNemar (exact) + paired bootstrap CI on the accuracy difference.

    ``correct_a`` / ``correct_b`` are per-item correctness for the SAME items
    (baseline A vs treatment B). ``delta = acc_b - acc_a``. The bootstrap resamples
    item INDICES (preserving pairing) to CI the difference.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("paired conditions must have equal length")
    n = len(correct_a)
    if n == 0:
        raise ValueError("need >= 1 paired item")
    both = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and b)
    only_a = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if a and not b)
    only_b = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if b and not a)
    neither = sum(1 for a, b in zip(correct_a, correct_b, strict=True) if not a and not b)
    acc_a = sum(correct_a) / n
    acc_b = sum(correct_b) / n
    delta = acc_b - acc_a
    p = binom_two_sided_p(only_b, only_a)

    rng = random.Random(seed)
    diffs = [int(b) - int(a) for a, b in zip(correct_a, correct_b, strict=True)]
    boots = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n))
        boots.append(s / n)
    boots.sort()
    ci_low = boots[int(0.025 * n_boot)]
    ci_high = boots[min(n_boot - 1, int(0.975 * n_boot))]
    return McNemarResult(
        name=name,
        n=n,
        acc_a=acc_a,
        acc_b=acc_b,
        delta=delta,
        both_correct=both,
        only_a_correct=only_a,
        only_b_correct=only_b,
        both_wrong=neither,
        p_exact=p,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def holm_correction(named_pvalues: list[tuple[str, float]], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down FWER control over a family of p-values.

    Returns, per comparison name: its raw p, its Holm-adjusted p, and whether it is
    rejected at family-wise ``alpha``. Holm makes no independence assumption.
    """
    m = len(named_pvalues)
    ordered = sorted(named_pvalues, key=lambda kv: kv[1])
    out: dict[str, dict] = {}
    max_adj = 0.0
    still_rejecting = True
    for i, (name, p) in enumerate(ordered):
        adj = min(1.0, (m - i) * p)
        adj = max(adj, max_adj)  # enforce monotonic non-decreasing adjusted p
        max_adj = adj
        if not (still_rejecting and adj < alpha):
            still_rejecting = False
        out[name] = {"p_raw": p, "p_holm": adj, "reject": still_rejecting and adj < alpha}
    return out


def mde_note(n: int, baseline_acc: float, mde_pp: float, alpha: float = 0.05) -> str:
    """A plain-language power note: is n plausibly enough to detect an mde_pp effect?

    Uses the normal approximation for a paired-proportion (McNemar) test, with the
    conservative assumption that the effect manifests entirely as discordant items.
    This is an orientation heuristic, NOT a substitute for the observed CI — the
    report leads with the CI vs the MDE.
    """
    # Approx discordant rate if the true per-item flip prob ~ mde; under the crude
    # model the expected discordant count ~ n * mde_pp/100 (one-directional). Power
    # to see it at alpha via a sign-test normal approx:
    d = n * (mde_pp / 100.0)
    if d <= 0:
        return f"MDE {mde_pp:.1f}pp: degenerate (n or mde is 0)."
    se = math.sqrt(n * 0.25)  # sd of discordant-balanced count under H0
    z = d / (2 * se) if se > 0 else 0.0
    # crude power at this z vs the alpha critical (~1.96 two-sided)
    powerish = "likely powered" if z >= 1.96 else "UNDERPOWERED (n too small to be sure)"
    return (
        f"MDE {mde_pp:.1f}pp on n={n} (baseline acc {baseline_acc:.2f}): "
        f"normal-approx z~{z:.2f} ⇒ {powerish}. Lead with the bootstrap CI vs the MDE."
    )
