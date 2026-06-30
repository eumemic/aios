"""WaM fusion RECIPE workflow bodies (Phase A: R1 verifier-gated chain).

Each recipe is a workflow script bound as ``model:"workflow:<id>"``. It receives the
binding input ``{messages, tools, params, session_id}`` and returns ONE assistant turn
in the ``call_llm`` result shape (``{content, tool_calls, finish_reason, ...}``), which
the binding boundary projects into the outer agent's turn.

**R1 — verifier-gated chain.** Thinker(A) drafts an approach → Worker(B) produces the
answer → Verifier(C) checks it. If the Verifier rejects, the Worker retries with the
Verifier's critique, up to ``RETRY_CAP`` times. The Verifier MUST sit on a different
substrate than the Worker (the substrate-different-verdict invariant — a checker that
shares the Worker's blind spots adds correlated noise, not independent detection).

Two builds from the SAME script, differing ONLY in the three role-model bindings:
  * **heterogeneous-R1** — A/B/C are different models, C ≠ B substrate (the recipe).
  * **self-fusion** — A=B=C = the best single model (the MATCHED-FAN-IN baseline:
    identical chain depth, retries, and prompt scaffold, homogeneous substrate).
A third condition, **best-single**, is a one-call passthrough (R0-shaped) on the best
single model — same model, no fan-in. The honest question is heterogeneous-R1 vs
self-fusion (equal fan-in), NOT vs best-single (confounded by fan-in).

Determinism: the retry loop branches only on the Verifier's (memoized) verdict and a
loop counter — both replay-stable — so the multi-call run is deterministic on re-wake.

Models are substituted at provision time via ``__A_MODEL__`` etc. (not f-strings: the
script body contains ``{`` / ``}``). A per-model params JSON is substituted too, so the
opus-4-8 "no temperature" gotcha is handled per role (Opus → ``{}``, others → temp=0).
"""

from __future__ import annotations

import json

RETRY_CAP = 2

# The verifier emits a verdict line we parse deterministically. Kept blunt so a
# capable model reliably complies and the parse is robust.
VERIFIER_SYSTEM = (
    "You are a strict verifier on a DIFFERENT model substrate than the worker. "
    "You are given a problem and a proposed answer. Check the answer's correctness by "
    "reasoning independently. Reply with a first line that is EXACTLY 'VERDICT: APPROVE' "
    "if the proposed final answer is correct, or 'VERDICT: REJECT' if it is wrong or "
    "unjustified. If you REJECT, add one short line explaining the specific error so the "
    "worker can fix it. Do not restate the whole solution."
)

# R1 verifier-gated chain. Placeholders substituted by build_r1_script().
R1_SCRIPT_TEMPLATE = """
async def main(input):
    msgs = input["messages"]
    A_MODEL = "__A_MODEL__"
    B_MODEL = "__B_MODEL__"
    C_MODEL = "__C_MODEL__"
    A_PARAMS = __A_PARAMS__
    B_PARAMS = __B_PARAMS__
    C_PARAMS = __C_PARAMS__
    VERIFIER_SYSTEM = __VERIFIER_SYSTEM__
    RETRY_CAP = __RETRY_CAP__

    # Extract the problem text (last user turn) for the worker/verifier sub-prompts.
    problem = ""
    for m in reversed(msgs):
        if m.get("role") == "user":
            problem = m.get("content") or ""
            break

    # ── Thinker(A): sketch an approach (not the final answer) ───────────────
    thinker = await call_llm({
        "model": A_MODEL,
        "messages": msgs + [{
            "role": "user",
            "content": "Briefly outline the approach to solve the problem above in 2-4 steps. Do NOT give the final answer yet.",
        }],
        "params": A_PARAMS,
    })
    if "error" in thinker:
        return thinker
    approach = thinker.get("content") or ""

    # ── Worker(B): produce the answer, guided by the approach ───────────────
    worker_msgs = msgs + [{
        "role": "user",
        "content": "Here is a suggested approach:\\n" + approach + "\\n\\nNow solve the problem and give the final answer.",
    }]
    worker = await call_llm({"model": B_MODEL, "messages": worker_msgs, "params": B_PARAMS})
    if "error" in worker:
        return worker

    # ── Verifier(C) gate, with Worker retry up to RETRY_CAP ─────────────────
    for attempt in range(RETRY_CAP + 1):
        answer = worker.get("content") or ""
        verifier = await call_llm({
            "model": C_MODEL,
            "messages": [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {"role": "user", "content": "PROBLEM:\\n" + problem + "\\n\\nPROPOSED ANSWER:\\n" + answer},
            ],
            "params": C_PARAMS,
        })
        if "error" in verifier:
            # Verifier unavailable: fail open to the worker's current answer (do not
            # let a checker outage manufacture a failure). Recorded by cost/markers.
            return worker
        verdict = (verifier.get("content") or "").strip()
        if verdict.upper().startswith("VERDICT: APPROVE") or "VERDICT: APPROVE" in verdict.upper()[:40]:
            return worker
        if attempt >= RETRY_CAP:
            return worker  # exhausted retries; return the last worker answer
        # Rejected with a critique → Worker retries incorporating the critique.
        critique = verdict
        worker_msgs = worker_msgs + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "A verifier on a different model rejected that answer:\\n" + critique + "\\n\\nReconsider carefully and give a corrected final answer."},
        ]
        worker = await call_llm({"model": B_MODEL, "messages": worker_msgs, "params": B_PARAMS})
        if "error" in worker:
            return worker

    return worker
"""

# best-single is the R0-shaped passthrough (one call, the best single model).
BEST_SINGLE_SCRIPT_TEMPLATE = """
async def main(input):
    resp = await call_llm({
        "model": "__B_MODEL__",
        "messages": input["messages"],
        "tools": input.get("tools"),
        "params": __B_PARAMS__,
        "session_id": input.get("session_id"),
    })
    return resp
"""


# OpenRouter rejects a request whose max_tokens exceeds the key's REMAINING credit
# affordance (HTTP 402: "requires more credits, or fewer max_tokens"). gpt-5.5's default
# max_tokens (65536) blows that on a low-credit key, erroring every coding call (which
# would silently zero out a condition's accuracy). Cap it to a value that (a) fits the
# affordance and (b) is large enough to regenerate the biggest corpus source file
# (loop.py ~ 18.5k tokens) with headroom. Anthropic (ant-proxy) has no such cap need.
_OPENROUTER_MAX_TOKENS = 24000


def _params_for(model: str) -> dict:
    """Per-model params: handle the opus-4-8 'temperature deprecated' gotcha AND the
    OpenRouter max_tokens affordance cap (see ``_OPENROUTER_MAX_TOKENS``).

    Opus-4-8 rejects ``temperature`` entirely (provider BadRequest); every other
    reachable pool model accepts ``temperature=0``. OpenRouter-routed models also need
    a bounded ``max_tokens`` so a low-credit key can afford the request.
    """
    params: dict = {} if "opus-4-8" in model else {"temperature": 0}
    if model.startswith("openrouter/"):
        params["max_tokens"] = _OPENROUTER_MAX_TOKENS
    return params


def build_r1_script(a_model: str, b_model: str, c_model: str) -> str:
    s = R1_SCRIPT_TEMPLATE
    s = s.replace("__A_MODEL__", a_model)
    s = s.replace("__B_MODEL__", b_model)
    s = s.replace("__C_MODEL__", c_model)
    s = s.replace("__A_PARAMS__", json.dumps(_params_for(a_model)))
    s = s.replace("__B_PARAMS__", json.dumps(_params_for(b_model)))
    s = s.replace("__C_PARAMS__", json.dumps(_params_for(c_model)))
    s = s.replace("__VERIFIER_SYSTEM__", json.dumps(VERIFIER_SYSTEM))
    s = s.replace("__RETRY_CAP__", str(RETRY_CAP))
    return s


def build_best_single_script(b_model: str) -> str:
    s = BEST_SINGLE_SCRIPT_TEMPLATE
    s = s.replace("__B_MODEL__", b_model)
    s = s.replace("__B_PARAMS__", json.dumps(_params_for(b_model)))
    return s
