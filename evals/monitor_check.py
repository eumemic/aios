"""Check a WaM monitor from outside aios: its alarm, and that it is still running.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.monitor_check NAME

Run it from an external cron (daily, say). A monitor runs on an operator trigger, so
no session hears when it fails or alarms; this pull is how anyone does. It exits:

* 0: the latest completed weekly run is recent and shows nothing worse (its verdict
  is PASS, or FAIL: failing to re-prove non-inferiority isn't evidence of harm);
* 1: ALARM: the latest completed run shows the deployed workflow worse than the
  model it replaced (its win rate or a limit), at the monitor's false-alarm rate;
* 2: the monitor isn't watching: it never fired, its last fire failed, it has no
  completed run in ``--max-age-days``, or its latest run was INVALID (the judge
  couldn't tell replies apart) or INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from evals.client import Api, Client

OK, ALARM, NOT_WATCHING = 0, 1, 2
FAILED_FIRES = frozenset({"error", "timeout"})


def check(
    api: Api, name: str, *, now: datetime, max_age: timedelta, out: Callable[[str], None]
) -> int:
    fires = api.get(f"/v1/triggers/{name}/runs", limit=20)["data"]
    if not fires:
        out(f"{name}: never fired")
        return NOT_WATCHING
    last = fires[0]
    if last["status"] in FAILED_FIRES:
        out(f"{name}: the last fire ({last['created_at']}) failed: {last['error_summary']}")
        return NOT_WATCHING
    for fire in fires:
        if fire["result_id"] is None:
            continue
        run = api.get(f"/v1/runs/{fire['result_id']}")
        if run["status"] != "completed":
            continue
        age = now - datetime.fromisoformat(run["created_at"])
        output = run["output"] or {}
        window = output.get("window") or {}
        week = f"{window.get('start', '?')[:10]}..{window.get('end', '?')[:10]}"
        if age > max_age:
            out(f"{name}: the latest completed run {run['id']} is {age.days} days old")
            return NOT_WATCHING
        if output.get("alarm"):
            out(
                f"{name}: ALARM {run['id']} week {week}: {output['candidate']} is worse on "
                f"{', '.join(output['alarms'])}"
            )
            return ALARM
        verdict = output.get("verdict")
        if verdict in ("INVALID", "INCONCLUSIVE", None):
            out(f"{name}: {run['id']} week {week} is {verdict}: {output.get('reasons')}")
            return NOT_WATCHING
        out(f"{name}: ok, {run['id']} week {week} shows nothing worse ({verdict})")
        return OK
    out(f"{name}: no completed run among the last {len(fires)} fires")
    return NOT_WATCHING


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="the monitor's operator trigger")
    parser.add_argument("--max-age-days", type=float, default=8.0)
    args = parser.parse_args(argv)
    return check(
        Client(),
        args.name,
        now=datetime.now(UTC),
        max_age=timedelta(days=args.max_age_days),
        out=print,
    )


if __name__ == "__main__":
    sys.exit(main())
