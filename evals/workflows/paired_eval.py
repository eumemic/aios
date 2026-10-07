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

1. Sample the agent's answered requests in the window. Skip and count items whose
   blobs are missing, and items captured for a model other than the baseline (their
   params wouldn't carry to the baseline arm), unless captured for a workflow binding.
2. Ask ``eval_analysis`` how many items the bar needs for its power. Too few eligible
   items is INCONCLUSIVE before anything is spent.
3. Run one item alone (a probe), then waves of at most ``wave_size`` items, each an
   ``eval_item`` sub-run with its own budget. An item the run cap refused, or one hit
   by a provider overload, is retried in a later wave up to ``max_attempts``. A wave
   the cap refused entirely, or a budget too low for another item, stops the run. A
   judge-family violation stops it at once.
4. Hand the records to ``eval_analysis``. If that fails, the records are returned
   anyway, so the analysis can be run again without paying for the items again.
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


def item_budget():
    b = BAR["budget"]
    return 2 * b["arm_usd"] + b["candidate_usd"] + b["judge_usd"]


def bump(counts, key):
    counts[key] = counts.get(key, 0) + 1


async def run_item(item, spec, attempt):
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
                "budget": BAR["budget"],
                "attempt": attempt,
            },
            version=CONFIG["item"]["version"],
            request_ref=item["request_ref"],
            label="item:" + str(item["index"]) + ":" + str(attempt),
            budget_usd=item_budget(),
        )
    except AgentError as e:
        if e.kind == "invoke_workflow_refused":
            return {"status": "launch_refused"}
        if e.kind == "budget_exceeded":
            return {"status": "budget"}
        return {"status": "excluded", "reason": "item_failed"}


async def run_items(queue, spec, exclusions, flags):
    records = []
    pending = [(item, 0) for item in queue]
    size = 1  # the probe: a judge-family violation stops the run after one item
    while pending:
        left = await budget()
        cap = size
        if left is not None:
            cap = min(cap, int(left["remaining_usd"] // item_budget()))
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
            elif status == "budget":
                flags.append("budget_stop")
                stop = True
            elif status == "excluded":
                bump(exclusions, result["reason"])
            else:
                records.append(result["record"])
                stop = stop or result["record"]["family_violation"]
        if refused == len(wave):
            flags.append("run_cap")
            break
        if stop:
            break
        size = BAR["wave_size"]
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
    exclusions = {}
    queue = []
    for i, item in enumerate(sample["items"]):
        if needed is not None and len(queue) >= needed:
            break
        if item["missing"]:
            bump(exclusions, "missing")
        elif item["model"] != spec["baseline_model"] and not item["model"].startswith("workflow:"):
            bump(exclusions, "model_mismatch")
        else:
            queue.append(dict(item, index=i))
    sampled = {
        "returned": len(sample["items"]),
        "eligible": len(queue),
        "n_required": needed,
        "power": power["power"],
    }
    if needed is None or len(queue) < needed:
        return dict(
            report,
            verdict="INCONCLUSIVE",
            reasons=inconclusive("underpowered"),
            sample=sampled,
            exclusions=exclusions,
        )

    phase("items")
    flags = []
    records = await run_items(queue, spec, exclusions, flags)
    considered = len(records) + sum(exclusions.values())
    report.update(
        sample=sampled,
        exclusions=dict(sorted(exclusions.items())),
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
