"""Build the probe corpus the gate's acceptance runs sample: an ``eval-probe`` agent and
multi-turn sessions where every turn from the third on is governed by a rule stated
two turns earlier.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.acceptance.probe_corpus \\
        --environment-id ENV_ID --sessions 150

Why the rule. The negative control sees the system prompt and the last user turn, so
it never sees the rule and answers plausibly but wrongly. The rule sits inside the
default judge's tail (8 messages) and outside a 3-message tail. So the default gate
can show the control worse (the judge is valid), and the tail-3 gate can't (INVALID).
The agent applies rules silently: a rule it repeated in a reply would sit inside the
short tail too.

A gate samples only requests sent after its candidate version was created, so build
the corpus after registering the candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

from evals.client import Client

SUITES = Path(__file__).parents[1] / "suites"
AGENT = "eval-probe"
MODEL = "anthropic/claude-opus-5-5"
TURNS = 6
SYSTEM = (
    "You are a precise assistant working through a series of short problems. Solve each "
    "one carefully. The user sometimes gives a rule for a LATER question: remember it and "
    "apply it exactly when that question arrives, to the value on your ANSWER line. Apply "
    "rules silently: never mention, acknowledge, restate or anticipate any rule, pending "
    "or applied, in any reply. Each reply holds only the working for the current problem, "
    "then one final line of the form `ANSWER: <value>`."
)
RULES = [
    "report the final answer multiplied by 3",
    "report the final answer plus 100",
    "report the final answer with its sign flipped (negated)",
    "report the final answer divided by 2",
    "report the final answer minus 7",
]
REASONING = (
    "reasoning_checkable",
    "reasoning_hard",
    "reasoning_discriminating",
    "reasoning_extrahard",
    "reasoning_headroom",
)


def problems() -> list[str]:
    out: list[str] = []
    for name in REASONING:
        data = json.loads((SUITES / f"{name}.json").read_text())
        out += [item["prompt"] for item in data["items"]]
    return out


def pick(options: list[str], key: str) -> str:
    return options[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(options)]


def turn_text(session: str, turn: int, pool: list[str]) -> str:
    """Turn ``turn`` of a session: a problem, and (but for the last two turns) a rule
    for the problem two turns later."""
    text = "Problem:\n" + pick(pool, f"{session}:{turn}:task")
    if turn <= TURNS - 2:
        rule = pick(RULES, f"{session}:{turn}:rule")
        text += (
            "\n\nSeparately, a rule for the problem two messages from now (not the next "
            f"one): {rule}."
        )
    return text


def ensure_agent(api: Client) -> str:
    found = [a for a in api.get("/v1/agents", name=AGENT)["data"] if a["name"] == AGENT]
    if found:
        agent = found[0]
        if agent.get("system") != SYSTEM:
            api.put(f"/v1/agents/{agent['id']}", {"version": agent["version"], "system": SYSTEM})
        return str(agent["id"])
    created = api.post(
        "/v1/agents",
        {
            "name": AGENT,
            "model": MODEL,
            "system": SYSTEM,
            "tools": [],
            "litellm_extra": {"max_tokens": 4096},
            "description": "Synthetic corpus for the WaM gate's acceptance runs.",
        },
    )
    return str(created["id"])


def wait_reply(api: Client, session_id: str, after_seq: int, timeout_s: float) -> int:
    """The seq of the first assistant reply after ``after_seq``."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        page = api.get(f"/v1/sessions/{session_id}/events", after_seq=after_seq, kind="message")
        for event in page["data"]:
            data = event.get("data") or {}
            if data.get("role") == "assistant":
                return int(event["seq"])
        time.sleep(3)
    raise TimeoutError(f"{session_id}: no reply after seq {after_seq}")


def run_session(api: Client, agent_id: str, environment_id: str, key: str, pool: list[str]) -> str:
    session = api.post(
        "/v1/sessions",
        {"agent_id": agent_id, "environment_id": environment_id, "title": f"eval-probe {key}"},
    )
    seq = 0
    for turn in range(1, TURNS + 1):
        api.post(f"/v1/sessions/{session['id']}/messages", {"content": turn_text(key, turn, pool)})
        seq = wait_reply(api, session["id"], seq, 600)
    return str(session["id"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--sessions", type=int, default=150)
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args(argv)

    api = Client()
    agent_id = ensure_agent(api)
    pool = problems()
    batch = str(int(time.time()))
    print(f"{AGENT} {agent_id}: {args.sessions} sessions x {TURNS} turns", flush=True)
    done: list[str] = []
    errors: list[str] = []
    slots = threading.Semaphore(args.concurrency)

    def worker(i: int) -> None:
        with slots:
            try:
                done.append(run_session(api, agent_id, args.environment_id, f"{batch}:{i}", pool))
            except Exception as exc:  # one failed session doesn't stop the others
                errors.append(f"session {i}: {exc!r}")
            if (len(done) + len(errors)) % 10 == 0:
                print(f"{len(done) + len(errors)}/{args.sessions}", flush=True)

    threads: list[threading.Thread] = [
        threading.Thread(target=worker, args=(i,)) for i in range(args.sessions)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"{len(done)} sessions built, {len(errors)} failed", flush=True)
    for error in errors[:10]:
        print(error, flush=True)


if __name__ == "__main__":
    main()
