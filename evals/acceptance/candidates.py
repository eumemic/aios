"""Register the acceptance runs' candidates: R0, a one-call workflow that forwards the
request it is bound to, pinned to a model.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.acceptance.candidates

``r0-opus-5-5`` is R0 on the probe agent's own model, so the gate should find it
non-inferior (PASS). ``r0-haiku-4-5`` is R0 on a much weaker model (FAIL). The weak
arm clamps ``max_tokens`` to what the model can emit, so it answers and loses on
quality rather than erroring on every call. Prints each ``id@version``.
"""

from __future__ import annotations

from evals.client import Client
from evals.register import ensure

R0 = """
async def main(input):
    params = dict(input.get("params") or {})
    if params.get("max_tokens", 0) > %(max_tokens)d:
        params["max_tokens"] = %(max_tokens)d
    return await call_llm({
        "model": %(model)r,
        "messages": input["messages"],
        "tools": input.get("tools"),
        "params": params,
        "session_id": input.get("session_id"),
    })
"""

CANDIDATES = {
    "r0-opus-5-5": ("anthropic/claude-opus-5-5", 128000, "R0 on the agent's own model."),
    "r0-haiku-4-5": ("anthropic/claude-haiku-4-5-20251001", 32000, "R0 on a much weaker model."),
}


def main() -> None:
    api = Client()
    for name, (model, max_tokens, description) in CANDIDATES.items():
        script = R0 % {"model": model, "max_tokens": max_tokens}
        ref = ensure(api, name, script, [], f"Acceptance candidate: {description}")
        print(f"{name:14} {ref['id']}@{ref['version']}")


if __name__ == "__main__":
    main()
