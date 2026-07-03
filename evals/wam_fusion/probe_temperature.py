#!/usr/bin/env python3
"""Stage-0 mechanism-(b) probe: WHY was `temperature` rejected on opus-4-8?

The Claude API reference (checked 2026-07-03) says the sampling params
(`temperature`, `top_p`, `top_k`) were REMOVED from the Opus 4.7/4.8 request
surface — 400 on every provider (first-party, Bedrock, Vertex), independent of
thinking mode. This probe verifies that empirically through the exact treatment
path the eval uses (aios session -> workflow binding -> call_llm -> ant-proxy),
so the §10 record contains the provider's literal error rather than an inferred
one. Cost: one tiny turn (~cents) if it unexpectedly succeeds; $0 if rejected.

Verdict semantics for the redo:
  * REJECTED (expected): temperature is NOT a diversity lever for opus-4-8 —
    not via ant-proxy, not via OpenRouter (OpenRouter fronts the same providers,
    all of which 400), not with thinking off. Mechanism (a) archetype
    prompt-jitter is the lever.
  * ACCEPTED (unexpected): record and surface — would reopen mechanism (b).
"""

from __future__ import annotations

import sys
import time

from aios_client import AiosClient

PROBE_SCRIPT = """
async def main(input):
    resp = await call_llm({
        "model": "anthropic/claude-opus-4-8",
        "messages": input["messages"],
        "params": {"temperature": 1.0},
        "session_id": input.get("session_id"),
    })
    return resp
"""


def main() -> int:
    client = AiosClient()
    wf = client.ensure_workflow(
        "selgen-opus-tempprobe", PROBE_SCRIPT, "EVS §10 temperature probe (opus-4-8)"
    )
    wf = wf.get("data", wf) if isinstance(wf, dict) else wf
    bind = f"workflow:{wf['id']}@{wf['version']}" if wf.get("version") else f"workflow:{wf['id']}"
    try:
        agent_id = client.create_agent("selgen-opus-tempprobe-agent", bind, "Reply tersely.", {})["id"]
    except Exception:
        resp = client._request("GET", "/v1/agents?limit=200")
        items = resp.get("data", resp) if isinstance(resp, dict) else resp
        agent_id = next(a["id"] for a in items if a.get("name") == "selgen-opus-tempprobe-agent")
    env_id = client.ensure_environment("wam-eval-env")["id"]
    sess = client.create_session(agent_id, env_id, "Say OK.")
    sid = sess["id"]
    print(f"probe session {sid} bind {bind}")
    deadline = time.time() + 180
    while time.time() < deadline:
        asst = client.latest_assistant(sid)
        if asst and asst.get("content"):
            print("UNEXPECTED: turn succeeded with temperature=1.0 ->", repr(asst["content"][:120]))
            return 1
        events = client.session_events(sid)
        errs = []
        for ev in events:
            d = ev.get("data") if isinstance(ev.get("data"), dict) else ev
            blob = str(d)
            if "error" in blob.lower() and ("temperature" in blob or "Error" in blob):
                errs.append(blob[:500])
        run_id = client.park_run_id(sid)
        if run_id:
            run = client.get_run(run_id)
            if run.get("status") in ("errored", "failed") or run.get("error"):
                print(f"RUN {run_id} status={run.get('status')} error={str(run.get('error'))[:400]}")
                for e in errs[-2:]:
                    print("EVT:", e)
                return 0
        if errs:
            print("REJECTED (session error events):")
            for e in errs[-3:]:
                print("EVT:", e)
            return 0
        time.sleep(5)
    print("probe timed out with neither text nor error — inspect session", sid)
    return 2


if __name__ == "__main__":
    sys.exit(main())
