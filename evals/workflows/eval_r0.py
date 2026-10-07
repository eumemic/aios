"""``eval_r0``: one inference on a named model, the baseline and negative-control arm.

The baseline arm is invoked with the item's ref and ``{"model", "request_ref"}`` and
sends that request by reference, rendered for its model. The negative control is
invoked with an inline request (``{"model", "messages", "tools", "params"}``) holding
less context than the judge sees. Either way it returns the raw ``call_llm`` result,
the same assistant-turn shape a workflow bound as a model returns.
"""

from __future__ import annotations

NAME = "eval-r0"
TOOLS: list[dict[str, str]] = []

SCRIPT = """
async def main(input):
    if "request_ref" in input:
        return await call_llm(request_ref=input["request_ref"], model=input["model"])
    return await call_llm(
        {
            "model": input["model"],
            "messages": input["messages"],
            "tools": input.get("tools"),
            "params": input.get("params"),
        }
    )
"""


def build() -> str:
    return SCRIPT
