"""``eval_analysis``: the eval statistics, pure compute. No capability is called.

Two modes, chosen by the input's ``mode``:

* ``power``: ``{"mode": "power", "bar"}`` returns the smallest number of items, up to
  the sample cap, with which the gate has its target power to pass a candidate that
  meets the bar's planning values, every limit and the negative control included:
  ``{"n": int | None, "power": float}``. Nothing is spent to find it.
* ``gate``: ``{"mode": "gate", "bar", "alpha", "records", "exclusions", "attributed",
  "considered", "flags", "seed"}`` returns the verdict and its statistics.
  ``attributed`` holds the cluster of each excluded item the candidate could have
  caused.

The primary statistic is the candidate's win rate W against the baseline (a tie is
half a win). Its one-sided bound is cluster-robust over (session, UTC day): the mean
plus or minus t(G-1) times SE, with SE from the clusters' summed residuals. The negative control's win
rate W_neg is bounded the same way and must be shown below 0.5 for the judge to count
as able to tell replies apart. W is bounded a second time with every attributed
exclusion scored as a candidate loss (``w_worst``), and PASS needs both bounds, so a
candidate can't lift its win rate by getting its losses excluded. Cost uses the same cluster-robust bound on the ratio of
summed uncached costs (delta method); p95 latency uses a cluster bootstrap, since a
quantile has no simple standard error. The two binary limits (the candidate degenerate
where the baseline isn't; the candidate's tool calls invalid where the baseline's are
valid) use the exact Clopper-Pearson bound.

The verdict: INVALID if the judge's family overlaps an arm's, the control isn't shown
worse, or the control pair's judged ties pass the ceiling. Otherwise INCONCLUSIVE if
there are too few clusters (overall or for the control), too many exclusions, or a
flag (a budget stop, a full run cap, too few items to have power). Otherwise PASS when
the primary bound and every limit hold, else FAIL.
"""

from __future__ import annotations

NAME = "eval-analysis"
TOOLS: list[dict[str, str]] = []

SCRIPT = '''
import hashlib
import math
import statistics

_NORMAL = statistics.NormalDist()
SCORE = {"win": 1.0, "tie": 0.5, "loss": 0.0}


# ── special functions (no scipy in a workflow) ────────────────────────────────


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3.0e-14:
            break
    return h


def betai(a, b, x):
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    ib = 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - ib if t > 0 else ib


def t_quantile(p, df):
    """The p-quantile (p > 0.5) of Student's t with ``df`` degrees of freedom."""
    hi = 1.0
    while t_cdf(hi, df) < p:
        hi *= 2.0
    lo = 0.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return hi


def binom_cdf(k, n, p):
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return betai(n - k, k + 1, 1.0 - p)


def cp_within(x, n, eps, alpha):
    """Whether the exact (Clopper-Pearson) one-sided 1-alpha upper bound of x/n is at
    most eps: P(Beta(x+1, n-x) <= eps) >= 1-alpha."""
    if x >= n:
        return eps >= 1.0
    return betai(x + 1, n - x, eps) >= 1.0 - alpha


def cp_upper(x, n, alpha):
    if x >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if betai(x + 1, n - x, mid) >= 1.0 - alpha:
            hi = mid
        else:
            lo = mid
    return hi


def cp_lower(x, n, alpha):
    if x <= 0:
        return 0.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if betai(x, n - x + 1, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def x_max(n, eps, alpha):
    """The most discordant items out of n for which the limit still holds (-1 when
    none can); within-ness only shrinks as x grows, so bisect."""
    lo, hi = -1, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cp_within(mid, n, eps, alpha):
            lo = mid
        else:
            hi = mid - 1
    return lo


# ── bounds ────────────────────────────────────────────────────────────────────


def _clusters(clusters):
    order = {}
    for c in clusters:
        order.setdefault(c, len(order))
    return order


def cluster_mean(values, clusters, alpha):
    """The mean and its cluster-robust one-sided 1-alpha bounds, as
    ``{"value", "lower", "upper", "clusters"}`` (bounds None with < 2 clusters)."""
    n = len(values)
    mean = sum(values) / n
    sums = {}
    for v, c in zip(values, clusters):
        sums[c] = sums.get(c, 0.0) + (v - mean)
    g = len(sums)
    if g < 2:
        return {"value": mean, "lower": None, "upper": None, "clusters": g}
    se = math.sqrt(g / (g - 1) * sum(s * s for s in sums.values())) / n
    t = t_quantile(1.0 - alpha, g - 1)
    return {"value": mean, "lower": mean - t * se, "upper": mean + t * se, "clusters": g}


def cluster_ratio(num, den, clusters, alpha):
    """sum(num)/sum(den) with cluster-robust one-sided bounds (delta method)."""
    total = sum(den)
    if total <= 0:
        return {"value": None, "lower": None, "upper": None, "clusters": len(set(clusters))}
    ratio = sum(num) / total
    sums = {}
    for a, b, c in zip(num, den, clusters):
        sums[c] = sums.get(c, 0.0) + (a - ratio * b)
    g = len(sums)
    if g < 2:
        return {"value": ratio, "lower": None, "upper": None, "clusters": g}
    se = math.sqrt(g / (g - 1) * sum(s * s for s in sums.values())) / total
    t = t_quantile(1.0 - alpha, g - 1)
    return {"value": ratio, "lower": ratio - t * se, "upper": ratio + t * se, "clusters": g}


def uniforms(seed):
    """An endless stream of uniforms in [0, 1): SHA-256 in counter mode."""
    i = 0
    while True:
        digest = hashlib.sha256((str(seed) + "|" + str(i)).encode()).digest()
        for k in range(0, 32, 8):
            yield int.from_bytes(digest[k:k + 8], "big") / 2.0**64
        i += 1


def p95(values):
    ordered = sorted(values)
    return ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]


def p95_ratio(base, cand, clusters, alpha, seed, rounds):
    """The ratio of p95 latencies with cluster-bootstrap one-sided bounds."""
    groups = {}
    for b, c, k in zip(base, cand, clusters):
        groups.setdefault(k, []).append((b, c))
    keys = sorted(groups)
    point = p95(cand) / max(p95(base), 1)
    if len(keys) < 2:
        return {"value": point, "lower": None, "upper": None, "clusters": len(keys)}
    draws = uniforms(seed)
    ratios = []
    for _ in range(rounds):
        pairs = []
        for _ in keys:
            pairs.extend(groups[keys[int(next(draws) * len(keys))]])
        ratios.append(p95([p[1] for p in pairs]) / max(p95([p[0] for p in pairs]), 1))
    ratios.sort()
    lo = ratios[max(math.floor(alpha * rounds) - 1, 0)]
    hi = ratios[min(math.ceil((1.0 - alpha) * rounds) - 1, rounds - 1)]
    return {"value": point, "lower": lo, "upper": hi, "clusters": len(keys)}


def length_controlled(scores, base_chars, cand_chars):
    """W at no length difference: the intercept of a linear fit of the score on the
    standardized difference in reply length."""
    diffs = [c - b for b, c in zip(base_chars, cand_chars)]
    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / n)
    if sd == 0:
        return sum(scores) / n
    z = [(d - mean_d) / sd for d in diffs]
    mean_s = sum(scores) / n
    slope = sum(zi * (s - mean_s) for zi, s in zip(z, scores)) / sum(zi * zi for zi in z)
    # The fit at a zero difference, where z = -mean_d / sd.
    return mean_s - slope * mean_d / sd


# ── power ─────────────────────────────────────────────────────────────────────


def power_at(bar, n):
    """Joint power at n items under the bar's planning values: the product of each
    test's power, conservative when the tests are positively dependent."""
    plan = bar["planning"]
    alpha = bar["alpha"]
    z = _NORMAL.inv_cdf(1.0 - alpha)
    deff = 1.0 + (bar["cluster_cap"] - 1) * plan["icc"]
    se = math.sqrt(plan["score_var"] * deff / n)
    power = _NORMAL.cdf((plan["w"] - (0.5 - bar["delta"])) / se - z)
    n_control = n * plan["control_eligible"]
    se_neg = math.sqrt(plan["control_var"] * deff / n_control)
    power *= _NORMAL.cdf((0.5 - plan["w_neg"]) / se_neg - z)
    for limit, rate in (("degenerate", "degenerate_rate"), ("tool_calls", "tool_call_rate")):
        power *= binom_cdf(x_max(n, bar["limits"][limit], alpha), n, plan[rate])
    for limit, ratio, cv in (("cost", "cost_ratio", "cost_cv"), ("latency", "latency_ratio", "latency_cv")):
        se_r = plan[cv] * math.sqrt(deff / n)
        power *= _NORMAL.cdf((1.0 + bar["limits"][limit] - plan[ratio]) / se_r - z)
    return power


def required_n(bar):
    """The smallest n, from the minimum cluster count up to the sample size, with
    joint power at the target; None when none reaches it."""
    for n in range(bar["min_clusters"], bar["sample_size"] + 1):
        power = power_at(bar, n)
        if power >= bar["power"]:
            return {"n": n, "power": power}
    return {"n": None, "power": power_at(bar, bar["sample_size"])}


# ── the verdict ───────────────────────────────────────────────────────────────


def _count(records, test):
    return sum(1 for r in records if test(r))


def _kinds(arms):
    """An arm's errors by kind (``budget`` among them: losses the budget forced)."""
    out = {}
    for arm in arms:
        if arm["error_kind"] is not None:
            out[arm["error_kind"]] = out.get(arm["error_kind"], 0) + 1
    return out


def analyze(input):
    bar = input["bar"]
    alpha = input["alpha"]
    limits = bar["limits"]
    invalid, inconclusive, failed = [], [], []

    if any(r["family_violation"] for r in input["records"]):
        invalid.append("judge_family")
    records = [r for r in input["records"] if not r["family_violation"]]
    excluded = sum(input["exclusions"].values())
    considered = input["considered"]
    for flag in sorted(set(input["flags"])):
        inconclusive.append(flag)
    if considered and excluded / considered > bar["max_exclusion_rate"]:
        inconclusive.append("exclusion_rate")
    clusters = [r["cluster"] for r in records]
    if len(set(clusters)) < bar["min_clusters"]:
        inconclusive.append("too_few_clusters")
    if not records:
        return verdict(invalid, inconclusive + ["no_records"], failed, {}, {})

    stats = {"n": len(records), "clusters": len(set(clusters))}
    scores = [SCORE[r["outcomes"]["cand"]] for r in records]
    w = cluster_mean(scores, clusters, alpha)
    stats["w"] = w
    if w["lower"] is None or w["lower"] < 0.5 - bar["delta"]:
        failed.append("win_rate")
    # The worst case: every exclusion the candidate could have caused is its loss, so
    # turning losses into exclusions can't lift the verdict.
    attributed = input.get("attributed", [])
    w_worst = cluster_mean(scores + [0.0] * len(attributed), clusters + attributed, alpha)
    stats["w_worst"] = dict(w_worst, attributed=len(attributed))
    if attributed and (w_worst["lower"] is None or w_worst["lower"] < 0.5 - bar["delta"]):
        failed.append("win_rate_worst_case")

    control = [r for r in records if r["outcomes"]["neg"] is not None]
    control_clusters = [r["cluster"] for r in control]
    stats["control"] = {"n": len(control), "clusters": len(set(control_clusters))}
    if len(set(control_clusters)) < bar["control"]["min_clusters"]:
        inconclusive.append("control_unpowered")
    else:
        w_neg = cluster_mean(
            [SCORE[r["outcomes"]["neg"]] for r in control], control_clusters, alpha
        )
        judged = [r for r in control if not r["identical"]["neg"]]
        ties = _count(judged, lambda r: r["outcomes"]["neg"] == "tie")
        tie_rate = ties / len(judged) if judged else 0.0
        stats["control"].update(w=w_neg, judged=len(judged), tie_rate=tie_rate)
        if w_neg["upper"] is None or w_neg["upper"] >= 0.5:
            invalid.append("control_not_worse")
        if tie_rate > bar["control"]["tie_ceiling"]:
            invalid.append("control_ties")

    arms = [(r["arms"]["base"], r["arms"]["cand"]) for r in records]
    n = len(arms)
    worse_degenerate = _count(arms, lambda a: a[1]["degenerate"] and not a[0]["degenerate"])
    worse_tools = _count(
        arms, lambda a: a[0]["tool_calls_valid"] and not a[1]["tool_calls_valid"]
        and a[1]["n_tool_calls"] > 0
    )
    stats["degenerate"] = {
        "worse": worse_degenerate, "n": n, "upper": cp_upper(worse_degenerate, n, alpha)
    }
    stats["tool_calls"] = {"worse": worse_tools, "n": n, "upper": cp_upper(worse_tools, n, alpha)}
    if not cp_within(worse_degenerate, n, limits["degenerate"], alpha):
        failed.append("degenerate")
    if not cp_within(worse_tools, n, limits["tool_calls"], alpha):
        failed.append("tool_calls")

    priced = [
        (a[0]["uncached_cost_microusd"], a[1]["uncached_cost_microusd"], r["cluster"])
        for a, r in zip(arms, records)
    ]
    if any(b is None or c is None for b, c, _ in priced):
        inconclusive.append("cost_unpriced")
    else:
        cost = cluster_ratio(
            [c for _, c, _ in priced], [b for b, _, _ in priced],
            [k for _, _, k in priced], alpha,
        )
        stats["cost"] = cost
        if cost["upper"] is None or cost["upper"] > 1.0 + limits["cost"]:
            failed.append("cost")

    timed = [
        (a[0]["duration_ms"], a[1]["duration_ms"], r["cluster"])
        for a, r in zip(arms, records)
        if a[0]["duration_ms"] is not None and a[1]["duration_ms"] is not None
    ]
    if len(set(k for _, _, k in timed)) < 2:
        inconclusive.append("latency_untimed")
    else:
        latency = p95_ratio(
            [b for b, _, _ in timed], [c for _, c, _ in timed], [k for _, _, k in timed],
            alpha, input["seed"], bar["bootstrap_rounds"],
        )
        stats["latency_p95"] = latency
        if latency["upper"] > 1.0 + limits["latency"]:
            failed.append("latency")

    fidelity = {}
    for r in records:
        fidelity[r["fidelity"]] = fidelity.get(r["fidelity"], 0) + 1
    # Why eligible items have no control outcome: the control arm's errors by kind.
    control_errors = {}
    for r in records:
        neg = r["arms"].get("neg")
        if r["control_eligible"] and r["outcomes"]["neg"] is None and neg is not None:
            kind = neg["error_kind"] or "invalid_output"
            control_errors[kind] = control_errors.get(kind, 0) + 1
    cand_judged = [r for r in records if not r["identical"]["cand"]]
    diagnostics = {
        "w_length_controlled": length_controlled(
            scores, [a[0]["chars"] for a in arms], [a[1]["chars"] for a in arms]
        ),
        "agreement": _count(records, lambda r: r["identical"]["cand"]) / n,
        "candidate_tie_rate": (
            _count(cand_judged, lambda r: r["outcomes"]["cand"] == "tie") / len(cand_judged)
            if cand_judged else 0.0
        ),
        "exclusions": dict(sorted(input["exclusions"].items())),
        "excluded": excluded,
        "attributed": len(attributed),
        "control_errors": dict(sorted(control_errors.items())),
        "candidate_errors": dict(sorted(_kinds(a[1] for a in arms).items())),
        "considered": considered,
        "fidelity": dict(sorted(fidelity.items())),
        "baseline_rates": {
            "degenerate": _count(arms, lambda a: a[0]["degenerate"]) / n,
            "candidate_degenerate": _count(arms, lambda a: a[1]["degenerate"]) / n,
        },
        "items": [
            {
                "ref": r["ref"],
                "score": s,
                "cost_delta_microusd": (
                    None if a[1]["uncached_cost_microusd"] is None
                    or a[0]["uncached_cost_microusd"] is None
                    else a[1]["uncached_cost_microusd"] - a[0]["uncached_cost_microusd"]
                ),
                "latency_delta_ms": (
                    None if a[1]["duration_ms"] is None or a[0]["duration_ms"] is None
                    else a[1]["duration_ms"] - a[0]["duration_ms"]
                ),
            }
            for r, s, a in zip(records, scores, arms)
        ],
    }
    return verdict(invalid, inconclusive, failed, stats, diagnostics)


def verdict(invalid, inconclusive, failed, stats, diagnostics):
    if invalid:
        name = "INVALID"
    elif inconclusive:
        name = "INCONCLUSIVE"
    elif failed:
        name = "FAIL"
    else:
        name = "PASS"
    return {
        "verdict": name,
        "reasons": {"invalid": invalid, "inconclusive": inconclusive, "failed": failed},
        "stats": stats,
        "diagnostics": diagnostics,
    }


async def main(input):
    if input["mode"] == "power":
        return required_n(input["bar"])
    return analyze(input)
'''


def build() -> str:
    return SCRIPT
