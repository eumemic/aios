"""The eval statistics (``evals/workflows/eval_analysis``), called outside a run.

The script's helpers are loaded from its rendered text, so these test exactly what a
registered version runs."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import pytest
from evals.workflows import eval_analysis, load

NS = load(eval_analysis.build())
BAR: dict[str, Any] = json.loads(
    (Path(__file__).parents[2] / "evals" / "bars" / "wam_gate.json").read_text()
)


# ── special functions ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("p", "df", "expected"),
    [(0.95, 29, 1.6991), (0.975, 10, 2.2281), (0.95, 1, 6.3138), (0.99, 100, 2.3642)],
)
def test_t_quantile_matches_tables(p: float, df: int, expected: float) -> None:
    assert NS["t_quantile"](p, df) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize(("k", "n", "p"), [(0, 10, 0.3), (3, 10, 0.5), (7, 25, 0.2), (9, 10, 0.9)])
def test_binomial_cdf_matches_the_sum(k: int, n: int, p: float) -> None:
    exact = sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))
    assert NS["binom_cdf"](k, n, p) == pytest.approx(exact, abs=1e-10)


def test_clopper_pearson_bounds() -> None:
    # x = 0 and x = n have closed forms.
    assert NS["cp_upper"](0, 30, 0.05) == pytest.approx(1 - 0.05 ** (1 / 30), abs=1e-9)
    assert NS["cp_lower"](30, 30, 0.05) == pytest.approx(0.05 ** (1 / 30), abs=1e-9)
    # Otherwise the bound is the Beta quantile: P(Beta(x+1, n-x) <= upper) = 1 - alpha.
    upper = NS["cp_upper"](2, 20, 0.05)
    assert NS["betai"](3, 18, upper) == pytest.approx(0.95, abs=1e-9)
    lower = NS["cp_lower"](5, 30, 0.05)
    assert NS["betai"](5, 26, lower) == pytest.approx(0.05, abs=1e-9)
    assert lower < 5 / 30 < NS["cp_upper"](5, 30, 0.05)


def test_x_max_is_the_last_count_within_the_limit() -> None:
    for n in (100, 200, 500):
        x = NS["x_max"](n, 0.05, 0.05)
        assert x >= 0
        assert NS["cp_within"](x, n, 0.05, 0.05)
        assert not NS["cp_within"](x + 1, n, 0.05, 0.05)
    # 0 of n shows a rate under 5% only from n = 59 (1 - 0.05 ** (1 / n) <= 0.05).
    assert NS["x_max"](58, 0.05, 0.05) == -1
    assert NS["x_max"](59, 0.05, 0.05) == 0


# ── bounds ────────────────────────────────────────────────────────────────────


def test_the_cluster_bound_uses_cluster_residuals() -> None:
    values = [1.0, 1.0, 0.0, 0.0, 0.5, 0.5]
    clusters = ["a", "a", "b", "b", "c", "c"]
    out = NS["cluster_mean"](values, clusters, 0.05)
    # Residual sums per cluster: +1, -1, 0; SE = sqrt(3/2 * 2) / 6 = sqrt(3) / 6.
    se = math.sqrt(3) / 6
    t = NS["t_quantile"](0.95, 2)
    assert out["value"] == 0.5 and out["clusters"] == 3
    assert out["lower"] == pytest.approx(0.5 - t * se)
    assert out["upper"] == pytest.approx(0.5 + t * se)


def test_clustering_widens_the_bound_when_items_in_a_cluster_agree() -> None:
    values = [1.0, 1.0, 0.0, 0.0] * 5
    by_item = NS["cluster_mean"](values, list(range(20)), 0.05)
    by_pair = NS["cluster_mean"](values, [i // 2 for i in range(20)], 0.05)
    assert by_pair["upper"] - by_pair["value"] > by_item["upper"] - by_item["value"]


def test_identical_values_have_no_spread() -> None:
    out = NS["cluster_mean"]([0.5] * 10, list(range(10)), 0.05)
    assert out["lower"] == out["upper"] == 0.5


def test_one_cluster_has_no_bound() -> None:
    assert NS["cluster_mean"]([1.0, 0.0], ["a", "a"], 0.05)["lower"] is None


def test_the_cost_ratio_bound() -> None:
    same = NS["cluster_ratio"]([10, 20, 30], [10, 20, 30], ["a", "b", "c"], 0.05)
    assert same["value"] == same["lower"] == same["upper"] == 1.0
    double = NS["cluster_ratio"]([20, 38, 64], [10, 20, 30], ["a", "b", "c"], 0.05)
    assert double["value"] == pytest.approx(122 / 60)
    assert double["lower"] < double["value"] < double["upper"]


def test_the_latency_bootstrap_is_seeded() -> None:
    base = [100.0 + i for i in range(40)]
    cand = [150.0 + 2 * i for i in range(40)]
    clusters = [i // 2 for i in range(40)]
    first = NS["p95_ratio"](base, cand, clusters, 0.05, "s", 300)
    assert first == NS["p95_ratio"](base, cand, clusters, 0.05, "s", 300)
    assert first != NS["p95_ratio"](base, cand, clusters, 0.05, "t", 300)
    assert first["lower"] <= first["value"] <= first["upper"]


def test_length_control_removes_a_pure_length_effect() -> None:
    base = [100] * 8
    cand = [80, 90, 100, 110, 120, 130, 140, 150]
    # Longer wins, with no effect at equal length.
    scores = [0.5 + (c - 100) / 200 for c in cand]
    assert NS["length_controlled"](scores, base, cand) == pytest.approx(0.5)


# ── power ─────────────────────────────────────────────────────────────────────


def test_the_default_bar_reaches_its_power_within_one_sample() -> None:
    needed = NS["required_n"](BAR)
    assert needed["n"] is not None
    assert BAR["min_clusters"] <= needed["n"] <= BAR["sample_size"]
    assert needed["power"] >= BAR["power"]
    assert NS["power_at"](BAR, needed["n"] - 1) < BAR["power"]


def test_the_power_search_is_fast_at_the_sample_cap() -> None:
    unreachable = copy.deepcopy(BAR)
    unreachable["delta"] = 0.01  # forces the search through every n up to 500
    started = time.monotonic()
    assert NS["required_n"](unreachable)["n"] is None
    assert time.monotonic() - started < 5.0


def test_power_grows_with_n() -> None:
    powers = [NS["power_at"](BAR, n) for n in (50, 100, 200, 400)]
    assert powers == sorted(powers)


# ── the verdict ───────────────────────────────────────────────────────────────


def _arm(**changes: Any) -> dict[str, Any]:
    arm = {
        "error_kind": None,
        "invalid_output": False,
        "degenerate": False,
        "tool_calls_valid": True,
        "n_tool_calls": 0,
        "chars": 100,
        "cost_microusd": 1000,
        "uncached_cost_microusd": 1000,
        "duration_ms": 1000,
        "models": ["openai/gpt-4o"],
    }
    arm.update(changes)
    return arm


def _record(
    i: int, cand: str = "tie", neg: str | None = "loss", cand_arm: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "family_violation": False,
        "ref": {"session_id": f"s{i}", "request_id": f"r{i}"},
        "cluster": f"s{i}|2026-09-01",
        "fidelity": "exact",
        "control_eligible": neg is not None,
        "outcomes": {"cand": cand, "neg": neg},
        "identical": {"cand": cand == "tie", "neg": False},
        "orders": {},
        "arms": {
            "base": _arm(),
            "cand": cand_arm or _arm(),
            "neg": _arm() if neg is not None else None,
        },
        "candidate_resolved": {"workflows": [], "agents": [], "models": []},
        "judge_models": [],
    }


def _analyze(records: list[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    bar = copy.deepcopy(BAR)
    bar["bootstrap_rounds"] = 200
    # 0 of 40 shows a rate under 10% (not under the default 5%, which needs 59 items).
    bar["limits"].update(degenerate=0.1, tool_calls=0.1)
    inputs = {
        "mode": "gate",
        "bar": bar,
        "alpha": bar["alpha"],
        "records": records,
        "exclusions": {},
        "considered": len(records),
        "flags": [],
        "seed": "s",
    }
    inputs.update(changes)
    return dict(NS["analyze"](inputs))


def test_identical_arms_pass() -> None:
    out = _analyze([_record(i) for i in range(40)])
    assert out["verdict"] == "PASS", out["reasons"]
    assert out["diagnostics"]["agreement"] == 1.0
    assert out["stats"]["control"]["w"]["upper"] == 0.0


def test_a_losing_candidate_fails() -> None:
    out = _analyze([_record(i, cand="loss") for i in range(40)])
    assert out["verdict"] == "FAIL"
    assert out["reasons"]["failed"] == ["win_rate"]


def test_exclusions_the_candidate_could_cause_count_as_losses_in_the_worst_case() -> None:
    """A candidate that gets its losses excluded (a judge failure on its reply) can't
    lift W: the worst case scores each such exclusion as a loss, and PASS needs it."""
    records = [_record(i) for i in range(40)]
    attributed = [f"x{i}|2026-09-02" for i in range(8)]
    out = _analyze(records, exclusions={"judge_error": 8}, attributed=attributed, considered=48)
    assert out["stats"]["w"]["value"] == 0.5
    assert out["stats"]["w_worst"]["value"] == 20 / 48
    assert out["stats"]["w_worst"]["attributed"] == 8
    assert out["verdict"] == "FAIL"
    assert out["reasons"]["failed"] == ["win_rate_worst_case"]
    assert out["diagnostics"]["attributed"] == 8


def test_without_attributed_exclusions_the_worst_case_is_w() -> None:
    out = _analyze([_record(i) for i in range(40)])
    assert out["stats"]["w_worst"]["value"] == out["stats"]["w"]["value"]
    assert out["verdict"] == "PASS"


def test_arm_errors_are_counted_by_kind() -> None:
    records = [_record(i) for i in range(40)]
    for r in records[:3]:
        r["outcomes"]["neg"] = None
        r["arms"]["neg"] = _arm(error_kind="author_exception", degenerate=True)
    for r in records[3:5]:
        r["outcomes"]["cand"] = "loss"
        r["identical"]["cand"] = False
        r["arms"]["cand"] = _arm(error_kind="budget", degenerate=True)
    out = _analyze(records)
    assert out["diagnostics"]["control_errors"] == {"author_exception": 3}
    assert out["diagnostics"]["candidate_errors"] == {"budget": 2}


def test_a_control_not_shown_worse_is_invalid() -> None:
    records = [_record(i, neg="win" if i % 2 else "loss") for i in range(40)]
    out = _analyze(records)
    assert out["verdict"] == "INVALID"
    assert out["reasons"]["invalid"] == ["control_not_worse"]


def test_a_tie_heavy_control_is_invalid() -> None:
    records = [_record(i, neg="tie" if i % 4 else "loss") for i in range(40)]
    out = _analyze(records)
    assert "control_ties" in out["reasons"]["invalid"]


def test_too_few_control_items_is_inconclusive_not_invalid() -> None:
    records = [_record(i, neg="loss" if i < 5 else None) for i in range(40)]
    out = _analyze(records)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reasons"]["inconclusive"] == ["control_unpowered"]
    assert out["reasons"]["invalid"] == []


def test_invalid_outranks_inconclusive_and_fail() -> None:
    records = [_record(i, cand="loss") for i in range(10)]
    records.append({"family_violation": True, "ref": {}})
    out = _analyze(records, flags=["budget_stop"])
    assert out["verdict"] == "INVALID"
    assert out["reasons"]["invalid"] == ["judge_family"]
    assert "budget_stop" in out["reasons"]["inconclusive"]
    assert "too_few_clusters" in out["reasons"]["inconclusive"]
    assert "win_rate" in out["reasons"]["failed"]  # with 10 items the binary limits fail too


def test_inconclusive_outranks_fail() -> None:
    records = [_record(i, cand="loss") for i in range(40)]
    out = _analyze(records, exclusions={"unavailable": 20}, considered=60)
    assert out["verdict"] == "INCONCLUSIVE"
    assert out["reasons"]["inconclusive"] == ["exclusion_rate"]


def test_more_degenerate_turns_fail_the_limit() -> None:
    records = [
        _record(i, cand_arm=_arm(degenerate=True) if i % 3 == 0 else _arm()) for i in range(60)
    ]
    out = _analyze(records)
    assert out["verdict"] == "FAIL"
    assert out["reasons"]["failed"] == ["degenerate"]
    assert out["stats"]["degenerate"]["worse"] == 20


def test_invalid_tool_calls_fail_the_limit() -> None:
    bad = _arm(tool_calls_valid=False, n_tool_calls=1)
    records = [_record(i, cand_arm=bad if i % 3 == 0 else _arm()) for i in range(60)]
    out = _analyze(records)
    assert out["reasons"]["failed"] == ["tool_calls"]


def test_a_dearer_candidate_fails_the_cost_limit() -> None:
    dear = _arm(uncached_cost_microusd=3000)
    out = _analyze([_record(i, cand_arm=dear) for i in range(40)])
    assert out["reasons"]["failed"] == ["cost"]
    assert out["stats"]["cost"]["value"] == 3.0


def test_unpriced_cost_is_inconclusive() -> None:
    unpriced = _arm(uncached_cost_microusd=None)
    out = _analyze([_record(i, cand_arm=unpriced) for i in range(40)])
    assert out["reasons"]["inconclusive"] == ["cost_unpriced"]


def test_a_slower_candidate_fails_the_latency_limit() -> None:
    slow = _arm(duration_ms=5000)
    out = _analyze([_record(i, cand_arm=slow) for i in range(40)])
    assert out["reasons"]["failed"] == ["latency"]


def test_outputs_are_finite() -> None:
    """A run's output must be finite JSON: no infinities from a degenerate bound."""
    out = _analyze([_record(i) for i in range(40)])
    text = json.dumps(out)
    assert "Infinity" not in text and "NaN" not in text


# ── the monitor's alarm ───────────────────────────────────────────────────────

_WEEKLY = 0.05 / 52


def _analyze_bar() -> dict[str, Any]:
    """The bar ``_analyze`` uses."""
    bar = copy.deepcopy(BAR)
    bar["limits"].update(degenerate=0.1, tool_calls=0.1)
    return bar


def _alarms(records: list[dict[str, Any]]) -> list[str]:
    out = _analyze(records, mode="monitor", alpha=_WEEKLY)
    return list(NS["alarms"](out, _analyze_bar()))


def test_an_unchanged_candidate_raises_no_alarm() -> None:
    assert _alarms([_record(i) for i in range(40)]) == []


def test_failing_to_re_prove_is_not_an_alarm() -> None:
    """W = 0.45 with a wide bound: the gate's FAIL, but nothing shown worse."""
    outcomes = ["loss"] * 20 + ["tie"] * 4 + ["win"] * 16
    records = [_record(i, cand=o) for i, o in enumerate(outcomes)]
    assert _analyze(records)["verdict"] == "FAIL"
    assert _alarms(records) == []


def test_a_candidate_shown_worse_alarms() -> None:
    assert _alarms([_record(i, cand="loss") for i in range(40)]) == ["win_rate"]


def test_an_invalid_judge_cannot_raise_the_win_rate_alarm() -> None:
    records = [_record(i, cand="loss", neg="win") for i in range(40)]
    assert _alarms(records) == []


def test_a_limit_shown_worse_alarms_whatever_the_judge() -> None:
    records = [
        _record(i, cand="loss", neg="win", cand_arm=_arm(degenerate=True)) for i in range(40)
    ]
    assert _alarms(records) == ["degenerate"]


def test_too_few_clusters_never_alarm() -> None:
    assert _alarms([_record(i, cand="loss") for i in range(10)]) == []


def test_the_monitor_main_reports_its_alarms() -> None:
    records = [_record(i, cand="loss") for i in range(40)]
    bar = _analyze_bar()
    bar["bootstrap_rounds"] = 200
    inputs = {
        "mode": "monitor",
        "bar": bar,
        "alpha": _WEEKLY,
        "records": records,
        "exclusions": {},
        "considered": 40,
        "flags": [],
        "seed": "s",
    }
    out = _monitor_main(inputs)
    assert out["alarm"] is True
    assert out["alarms"] == ["win_rate"]
    # The week's false-alarm rate is split across the alarm tests (Bonferroni).
    assert out["stats"]["alarm_alpha"] == pytest.approx(_WEEKLY / 4)
    assert list(NS["ALARM_TESTS"]) == ["win_rate", "degenerate", "tool_calls", "cost"]
    assert 0.0 < out["stats"]["detectable_w"] < 0.5 - bar["delta"]


def _monitor_main(inputs: dict[str, Any]) -> dict[str, Any]:
    coroutine = NS["main"](inputs)
    with pytest.raises(StopIteration) as done:
        coroutine.send(None)
    return dict(done.value.value)


def test_the_monitor_tests_the_judge_at_the_bars_level_not_the_alarms() -> None:
    """A control shown worse at the bar's level (W_neg = 0.35 over 40 clusters) is a
    valid judge. At the alarm's level, about 1e-4, the same control would read as
    not shown worse, and the week would be INVALID and deaf to a regression."""
    outcomes = ["loss"] * 26 + ["win"] * 14
    records = [_record(i, neg=o) for i, o in enumerate(outcomes)]
    out = _analyze(records, mode="monitor", alpha=_WEEKLY / 4)
    assert out["stats"]["control"]["w"]["upper"] < 0.5
    assert out["reasons"]["invalid"] == []
    strict = NS["cluster_mean"](
        [NS["SCORE"][o] for o in outcomes], [r["cluster"] for r in records], _WEEKLY / 4
    )
    assert strict["upper"] >= 0.5


def test_latency_is_not_a_monitor_alarm() -> None:
    """A percentile bootstrap can't bound a tail at the alarm's level, so a slower
    candidate is reported, not alarmed on."""
    records = [_record(i, cand_arm=_arm(duration_ms=50_000)) for i in range(40)]
    out = _analyze(records, mode="monitor", alpha=_WEEKLY / 4)
    assert out["stats"]["latency_p95"]["lower"] > 2.0
    assert NS["alarms"](out, _analyze_bar()) == []
