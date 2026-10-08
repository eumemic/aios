"""Check a WaM monitor from outside aios: its alarm, and that it is still watching.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.monitor_check NAME

Run it from an external cron (daily, say). A monitor runs on an operator trigger, so
no session hears when it fails or alarms; this pull is how anyone does. It exits:

* 0: the monitor is watching and shows nothing worse. The latest weekly run's
  verdict is PASS or FAIL (failing to re-prove non-inferiority isn't evidence of
  harm), or INCONCLUSIVE only because the week was too thin to decide (too few
  clusters, too few control items, no records). A thin week prints a warning, and
  two in a row a stronger one; a low-traffic agent is thin every week, which is no
  reason to page.
* 1: ALARM: the latest completed run shows the deployed workflow worse than the model
  it replaced (its win rate or a limit), at the monitor's false-alarm rate.
* 2: the monitor isn't watching, or this check couldn't tell:
  * the agent no longer runs the workflow the monitor tests (a rollback or a newer
    deploy: delete this monitor and create one for what is deployed);
  * it never fired, its last fire failed, or the run its last fire launched ended
    without completing;
  * it has no completed run in ``--max-age-days``;
  * its latest run was INVALID (the judge couldn't be trusted), or INCONCLUSIVE for
    an operational reason (a budget stop, the run cap, too many exclusions, cost
    that couldn't be priced, a failed sample or analysis);
  * the API couldn't be read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from evals.client import Api, Client

OK, ALARM, NOT_WATCHING = 0, 1, 2
FAILED_FIRES = frozenset({"error", "timeout"})
UNFINISHED = frozenset({"pending", "running", "suspended"})
# INCONCLUSIVE because the week was too thin to decide: not a fault of the monitor.
THIN_WEEK = frozenset({"too_few_clusters", "control_unpowered", "no_records"})


def _week(output: dict[str, Any]) -> str:
    window = output.get("window") or {}
    return f"{window.get('start', '?')[:10]}..{window.get('end', '?')[:10]}"


def _thin(output: dict[str, Any]) -> bool:
    reasons = output.get("reasons") or {}
    inconclusive = set(reasons.get("inconclusive") or [])
    return (
        output.get("verdict") == "INCONCLUSIVE"
        and not reasons.get("invalid")
        and bool(inconclusive)
        and inconclusive <= THIN_WEEK
    )


def check(
    api: Api, name: str, *, now: datetime, max_age: timedelta, out: Callable[[str], None]
) -> int:
    template = api.get(f"/v1/triggers/{name}")["action"]["input_template"]
    cand = template["candidate"]
    deployed = f"workflow:{cand['workflow_id']}@{cand['version']}"
    model = api.get(f"/v1/agents/{template['agent']['agent_id']}")["model"]
    if model != deployed:
        out(f"{name}: stale: the agent runs {model!r}, the monitor tests {deployed}")
        return NOT_WATCHING
    fires = api.get(f"/v1/triggers/{name}/runs", limit=20)["data"]
    if not fires:
        out(f"{name}: never fired")
        return NOT_WATCHING
    last = fires[0]
    if last["status"] in FAILED_FIRES:
        out(f"{name}: the last fire ({last['created_at']}) failed: {last['error_summary']}")
        return NOT_WATCHING
    completed: list[dict[str, Any]] = []
    for i, fire in enumerate(fires):
        if fire["result_id"] is None:
            continue
        run = api.get(f"/v1/runs/{fire['result_id']}")
        if run["status"] == "completed":
            completed.append(run)
            if len(completed) == 2:
                break
        elif i == 0 and run["status"] not in UNFINISHED:
            out(f"{name}: the last fire's run {run['id']} ended {run['status']}")
            return NOT_WATCHING
    if not completed:
        out(f"{name}: no completed run among the last {len(fires)} fires")
        return NOT_WATCHING
    run = completed[0]
    age = now - datetime.fromisoformat(run["created_at"])
    output = run["output"] or {}
    if age > max_age:
        out(f"{name}: the latest completed run {run['id']} is {age.days} days old")
        return NOT_WATCHING
    if output.get("alarm"):
        out(
            f"{name}: ALARM {run['id']} week {_week(output)}: {output['candidate']} is worse "
            f"on {', '.join(output['alarms'])}"
        )
        return ALARM
    verdict = output.get("verdict")
    if _thin(output):
        previous = completed[1]["output"] if len(completed) > 1 else None
        if previous is not None and _thin(previous):
            out(
                f"{name}: WARNING: two thin weeks in a row ({_week(previous)}, "
                f"{_week(output)}): the monitor can't decide on this traffic: "
                f"{output['reasons']['inconclusive']}"
            )
        else:
            out(
                f"{name}: warning: {run['id']} week {_week(output)} was too thin to decide: "
                f"{output['reasons']['inconclusive']}"
            )
        return OK
    if verdict in ("INVALID", "INCONCLUSIVE", None):
        out(f"{name}: {run['id']} week {_week(output)} is {verdict}: {output.get('reasons')}")
        return NOT_WATCHING
    out(f"{name}: ok, {run['id']} week {_week(output)} shows nothing worse ({verdict})")
    return OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the monitor's operator trigger")
    parser.add_argument("--max-age-days", type=float, default=8.0)
    args = parser.parse_args(argv)
    try:
        return check(
            Client(),
            args.name,
            now=datetime.now(UTC),
            max_age=timedelta(days=args.max_age_days),
            out=print,
        )
    except Exception as exc:
        # A crash must never read as ALARM (Python's own exit code for one is 1).
        print(f"{args.name}: the check failed: {type(exc).__name__}: {exc}")
        return NOT_WATCHING


if __name__ == "__main__":
    sys.exit(main())
