"""TOST (Two One-Sided Tests) for paired equivalence — pure stdlib.

The foundation question the WaM fusion eval must answer is NOT "is there no
significant difference between R0 and native" (absence of evidence) but the
positive claim "R0 is equivalent to native within a pre-declared margin." That
is exactly what TOST establishes: equivalence is **accepted** only when the
(1-2alpha) CI of the paired mean difference lies entirely inside (-margin,
+margin).

We implement the paired TOST on the per-task differences d_i = R0_i - native_i:

  H0_lower: mu_d <= -margin     vs  H1_lower: mu_d > -margin
  H0_upper: mu_d >= +margin     vs  H1_upper: mu_d < +margin

Equivalence is concluded iff BOTH one-sided nulls are rejected at alpha, i.e.
p_lower < alpha AND p_upper < alpha (equivalently, max(p_lower, p_upper) < alpha).

No scipy in the prod venv, so the Student-t CDF is computed from the regularized
incomplete beta function (continued-fraction, Numerical Recipes). Validated in
test_tost.py against known critical values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    MAXIT, EPS, FPMIN = 200, 3.0e-12, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """P(T <= t) for a Student-t with df degrees of freedom."""
    x = df / (df + t * t)
    ib = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - ib if t > 0 else ib


@dataclass(frozen=True)
class TostResult:
    n: int
    mean_diff: float
    sd_diff: float
    se_diff: float
    margin: float
    alpha: float
    t_lower: float
    p_lower: float
    t_upper: float
    p_upper: float
    p_tost: float  # max(p_lower, p_upper)
    ci_low: float  # (1 - 2*alpha) CI bounds
    ci_high: float
    equivalent: bool

    def summary(self) -> str:
        verdict = "EQUIVALENT (positively established)" if self.equivalent else "NOT established"
        return (
            f"TOST paired equivalence: {verdict}\n"
            f"  n={self.n}  margin=±{self.margin:g}  alpha={self.alpha:g}\n"
            f"  mean_diff(R0-native)={self.mean_diff:+.6g}  sd={self.sd_diff:.6g}  se={self.se_diff:.6g}\n"
            f"  {(1 - 2 * self.alpha) * 100:g}% CI of mean diff: [{self.ci_low:+.6g}, {self.ci_high:+.6g}]"
            f"  (must lie within (-{self.margin:g}, +{self.margin:g}))\n"
            f"  lower test: t={self.t_lower:.4g} p={self.p_lower:.4g}   "
            f"upper test: t={self.t_upper:.4g} p={self.p_upper:.4g}   p_TOST={self.p_tost:.4g}"
        )


def paired_tost(diffs: list[float], margin: float, alpha: float = 0.05) -> TostResult:
    """Paired TOST on differences ``diffs`` against an equivalence ``margin``.

    Equivalence is concluded iff both one-sided tests reject at ``alpha``
    (p_TOST = max(p_lower, p_upper) < alpha). A degenerate sample (all diffs
    identical, sd=0) is handled: if |mean| < margin it is trivially equivalent
    (the difference is a constant strictly inside the margin), else not.
    """
    n = len(diffs)
    if n < 2:
        raise ValueError("paired_tost needs n >= 2")
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    se = sd / math.sqrt(n)
    df = n - 1

    if se == 0.0:
        # Zero variance: the paired difference is a constant. Equivalent iff that
        # constant is strictly inside the margin (no sampling uncertainty).
        equivalent = abs(mean) < margin
        p = 0.0 if equivalent else 1.0
        return TostResult(
            n=n,
            mean_diff=mean,
            sd_diff=0.0,
            se_diff=0.0,
            margin=margin,
            alpha=alpha,
            t_lower=math.inf if equivalent else -math.inf,
            p_lower=p,
            t_upper=math.inf if equivalent else -math.inf,
            p_upper=p,
            p_tost=p,
            ci_low=mean,
            ci_high=mean,
            equivalent=equivalent,
        )

    # Lower one-sided: H0 mu <= -margin; reject (mu > -margin) when t_lower large +.
    t_lower = (mean - (-margin)) / se
    p_lower = 1.0 - student_t_cdf(t_lower, df)
    # Upper one-sided: H0 mu >= +margin; reject (mu < +margin) when t_upper large -.
    t_upper = (mean - margin) / se
    p_upper = student_t_cdf(t_upper, df)

    p_tost = max(p_lower, p_upper)
    # (1 - 2 alpha) two-sided CI ⇔ the TOST equivalence interval.
    t_crit = _inv_student_t(1.0 - alpha, df)
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    equivalent = p_tost < alpha

    return TostResult(
        n=n,
        mean_diff=mean,
        sd_diff=sd,
        se_diff=se,
        margin=margin,
        alpha=alpha,
        t_lower=t_lower,
        p_lower=p_lower,
        t_upper=t_upper,
        p_upper=p_upper,
        p_tost=p_tost,
        ci_low=ci_low,
        ci_high=ci_high,
        equivalent=equivalent,
    )


def _inv_student_t(p: float, df: float) -> float:
    """Inverse Student-t CDF via bisection (p in (0,1)); adequate for CI bounds."""
    lo, hi = -100.0, 100.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if student_t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0
