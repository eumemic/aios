"""``paired_eval``: the WaM gate run, registered as ``wam-gate``.

The bar (delta, alpha, the limits, the power target and planning values, the judge, the
control's requirements, the sample and wave sizes, the per-item budgets) is baked
into the registered version, so changing it makes a new version. A run's input names
only what is being compared::

    {"agent": {"agent_id", "version"}, "baseline_model",
     "candidate": {"workflow_id", "version"}, "seed",
     "window": {"start", "end"}, "candidate_created_at"}

A script has no clock and can't read workflow versions, so the launcher supplies the
window (opening no earlier than the candidate version, so the sample is a holdout)
and records ``candidate_created_at`` for the report to check.

1. Sample the agent's answered requests in the window. Items whose blobs are missing,
   or captured for a model other than the baseline (their params wouldn't carry to the
   baseline arm) unless captured for a workflow binding, can't be run.
2. Ask ``eval_analysis`` how many items the bar needs for its power. Too few eligible
   items is INCONCLUSIVE before anything is spent.
3. Run that many items in the sample's order, each an ``eval_item`` sub-run with its
   own budget: one at a time (the probe) until one has a record, then waves of at most
   ``wave_size``. An item the run cap refused, or one hit by load the eval caused, is
   retried in a later wave up to ``max_attempts``. An item left out for a reason the
   candidate can't reach (unavailable, screened out, retries spent) is replaced by the
   next one the sample holds; one the candidate could have caused (a judge failure on
   its reply, a failed item run) is not, and its cluster goes to the analysis as a
   loss in the worst case. A wave the cap refused entirely, or a budget too low for
   another item, stops the run. A judge-family violation stops it at once.
4. Hand the records to ``eval_analysis``. If that fails, the records are returned
   anyway, so the analysis can be run again without paying for the items again.

Per item, the candidate may spend ``(1 + limits.cost) * arm_usd * candidate_margin``:
its budget follows the cost limit the bar sets, so a recipe class that costs more by
design is gated by its own bar.
"""

from __future__ import annotations

from typing import Any

from evals.workflows import render

NAME = "wam-gate"
TOOLS: list[dict[str, str]] = [{"type": "sample_requests"}, {"type": "get_request"}]

SCRIPT = """
import hashlib
import json

CONFIG = json.loads(__CONFIG__)
BAR = CONFIG["bar"]


def budgets():
    '''Per-item spend ceilings. The candidate's follows the cost limit: it may spend
    up to ``candidate_margin`` times what the cost limit allows over the baseline's
    arm budget. The item's own budget holds the candidate's allowance twice, so a
    candidate overshooting within one step (its parallel calls all open before the
    budget is read) can't spend the judge's share.'''
    b = BAR["budget"]
    candidate = (1.0 + BAR["limits"]["cost"]) * b["arm_usd"] * b["candidate_margin"]
    arms = {"arm_usd": b["arm_usd"], "candidate_usd": candidate, "judge_usd": b["judge_usd"]}
    arms["item_usd"] = 2 * b["arm_usd"] + 2 * candidate + b["judge_usd"]
    return arms


def bump(counts, key):
    counts[key] = counts.get(key, 0) + 1


def cluster_of(item):
    return item["session_id"] + "|" + item["created_at"][:10]


async def run_item(item, spec, attempt):
    b = budgets()
    try:
        return await invoke_workflow(
            CONFIG["item"]["id"],
            {
                "request_ref": item["request_ref"],
                "item": {
                    "session_id": item["session_id"],
                    "created_at": item["created_at"],
                    "model": item["model"],
                },
                "agent": spec["agent"],
                "baseline_model": spec["baseline_model"],
                "candidate": spec["candidate"],
                "judge": BAR["judge"],
                "budget": {k: b[k] for k in ("arm_usd", "candidate_usd", "judge_usd")},
                "attempt": attempt,
                "final": attempt + 1 >= BAR["max_attempts"],
            },
            version=CONFIG["item"]["version"],
            request_ref=item["request_ref"],
            label="item:" + str(item["index"]) + ":" + str(attempt),
            budget_usd=b["item_usd"],
        )
    except AgentError as e:
        if e.kind == "invoke_workflow_refused":
            return {"status": "launch_refused"}
        if e.kind == "budget_exceeded":
            return {"status": "budget"}
        # The item run itself failed, after its candidate may have run.
        return {"status": "excluded", "reason": "item_failed", "attributable": True}


def screen(item, spec):
    '''Why a sampled item can't be run, or None. Its blobs are missing; or it was
    captured for a model other than the baseline, so its params wouldn't carry to the
    baseline arm (a capture for a workflow binding keeps them).'''
    if item["missing"]:
        return "missing"
    if item["model"] != spec["baseline_model"] and not item["model"].startswith("workflow:"):
        return "model_mismatch"
    return None


async def run_items(items, needed, spec, exclusions, attributed, flags):
    '''Run ``needed`` items, in the sample's seeded order. An item left out for a
    reason the candidate can't reach is replaced by the next one the sample holds;
    one the candidate could have caused is not (the analysis's worst case scores it
    as a loss instead). Items are screened as they are drawn, so the exclusion counts
    cover only the items the run considered.'''
    records = []
    pending = []
    stream = iter(items)

    def replace():
        for item in stream:
            reason = screen(item, spec)
            if reason is None:
                pending.append((item, 0))
                return
            bump(exclusions, reason)

    for _ in range(needed):
        replace()

    while pending:
        # The probe: one item at a time until a record exists, so a judge-family
        # violation stops the run before a full wave is paid for.
        size = BAR["wave_size"] if records else 1
        left = await budget()
        cap = size
        if left is not None:
            cap = min(cap, int(left["remaining_usd"] // budgets()["item_usd"]))
            if cap < 1:
                flags.append("budget_stop")
                break
        wave, pending = pending[:cap], pending[cap:]
        results = await parallel(
            [lambda w=w: run_item(w[0], spec, w[1]) for w in wave]
        )
        refused = 0
        stop = False
        for (item, attempt), result in zip(wave, results):
            status = result["status"]
            if status in ("launch_refused", "retry"):
                refused += status == "launch_refused"
                if attempt + 1 < BAR["max_attempts"]:
                    pending.append((item, attempt + 1))
                else:
                    bump(exclusions, "run_cap" if status == "launch_refused" else result["reason"])
                    replace()
            elif status == "budget":
                flags.append("budget_stop")
                stop = True
            elif status == "excluded":
                bump(exclusions, result["reason"])
                if result["attributable"]:
                    attributed.append(cluster_of(item))
                else:
                    replace()
            else:
                records.append(result["record"])
                stop = stop or result["record"]["family_violation"]
        if refused == len(wave):
            flags.append("run_cap")
            break
        if stop:
            break
    return records


def inconclusive(reason):
    return {"invalid": [], "inconclusive": [reason], "failed": []}


def resolved(records):
    out = {"workflows": set(), "agents": set(), "models": set()}
    for r in records:
        for key in out:
            out[key].update(r.get("candidate_resolved", {}).get(key, []))
    return {key: sorted(values) for key, values in out.items()}


async def main(input):
    spec = input
    seed = str(spec["seed"])
    alpha = BAR["alpha"]
    cand = spec["candidate"]
    report = {
        "candidate": "workflow:" + cand["workflow_id"] + "@" + str(cand["version"]),
        "agent": spec["agent"],
        "baseline_model": spec["baseline_model"],
        "window": spec["window"],
        "candidate_created_at": spec.get("candidate_created_at"),
        "seed": seed,
        "bar": BAR,
        "bar_digest": hashlib.sha256(
            json.dumps(BAR, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16],
    }

    phase("sample")
    sample = await tool(
        "sample_requests",
        {
            "agent_id": spec["agent"]["agent_id"],
            "start": spec["window"]["start"],
            "end": spec["window"]["end"],
            "n": BAR["sample_size"],
            "seed": seed,
            "cluster_cap": BAR["cluster_cap"],
        },
    )
    if "error" in sample:
        return dict(
            report,
            verdict="INCONCLUSIVE",
            reasons=inconclusive("sample_failed"),
            error=sample["error"],
        )

    power = await invoke_workflow(
        CONFIG["analysis"]["id"],
        {"mode": "power", "bar": BAR},
        version=CONFIG["analysis"]["version"],
        label="power",
    )
    needed = power["n"]
    items = [dict(item, index=i) for i, item in enumerate(sample["items"])]
    screened = {}
    for item in items:
        reason = screen(item, spec)
        if reason is not None:
            bump(screened, reason)
    eligible = len(items) - sum(screened.values())
    sampled = {
        "returned": len(items),
        "eligible": eligible,
        "n_required": needed,
        "power": power["power"],
    }
    if needed is None or eligible < needed:
        return dict(
            report,
            verdict="INCONCLUSIVE",
            reasons=inconclusive("underpowered"),
            sample=sampled,
            exclusions=dict(sorted(screened.items())),
        )

    phase("items")
    flags = []
    exclusions = {}
    attributed = []
    records = await run_items(items, needed, spec, exclusions, attributed, flags)
    considered = len(records) + sum(exclusions.values())
    report.update(
        sample=sampled,
        exclusions=dict(sorted(exclusions.items())),
        attributed=sorted(attributed),
        flags=sorted(set(flags)),
        candidate_resolved=resolved(records),
        records=records,
    )

    phase("analysis")
    try:
        analysis = await invoke_workflow(
            CONFIG["analysis"]["id"],
            {
                "mode": "gate",
                "bar": BAR,
                "alpha": alpha,
                "records": records,
                "exclusions": exclusions,
                "attributed": sorted(attributed),
                "considered": considered,
                "flags": flags,
                "seed": seed,
            },
            version=CONFIG["analysis"]["version"],
            label="analysis",
        )
    except AgentError as e:
        return dict(report, verdict=None, analysis_error=str(e))
    return dict(
        report,
        verdict=analysis["verdict"],
        reasons=analysis["reasons"],
        stats=analysis["stats"],
        diagnostics=analysis["diagnostics"],
    )
"""


def build(*, bar: dict[str, Any], item: dict[str, Any], analysis: dict[str, Any]) -> str:
    """``item`` and ``analysis`` are ``{"id", "version"}`` of the registered workflows."""
    return render(SCRIPT, config={"bar": bar, "item": item, "analysis": analysis})
