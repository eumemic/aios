"""``eval_judge``: compare replies to the baseline's, pairwise, in both orders.

Input::

    {"model", "params", "max_chars",
     "system": str, "conversation": str,      # already rendered by eval_item
     "base": reply, "pairs": {"cand": reply, "neg": reply}}

where a reply is ``{"content": str, "tool_calls": [...]}``. For each pair the judge
sees the system prompt, the tail of the conversation eval_item rendered, and the two
replies (text and the tool calls they request, unexecuted), once with the pair's
reply first and once with the baseline's first. Both orders agreeing is a win or a
loss; disagreeing, or either saying TIE, is a tie. Replies that are the same after
normalization are a tie without a judge call (``identical``). A verdict line that
can't be parsed is asked again once.

Returns ``{name: {"outcome": "win"|"loss"|"tie"|None, "identical": bool,
"orders": [...], "error"?: str}}``, the outcome from the pair's side.
"""

from __future__ import annotations

from evals.workflows import render

NAME = "eval-judge"
TOOLS: list[dict[str, str]] = []

RUBRIC = (
    "You compare two candidate replies an AI assistant could send next in a "
    "conversation. You see the assistant's system prompt, the most recent part of the "
    "conversation, and the two replies. A reply may request tool calls; they have not "
    "been run, so judge whether requesting them is the right next step. Decide which "
    "reply is the better next assistant turn: more correct, more helpful, better use "
    "of the context and tools, more faithful to the system prompt. Length is not "
    "quality. If they are equally good, say TIE. End your answer with one line that "
    "is exactly 'WINNER: 1', 'WINNER: 2' or 'WINNER: TIE'."
)

SCRIPT = '''
import json
import re

RUBRIC = json.loads(__RUBRIC__)
_VERDICT = re.compile(r"WINNER:\\s*(1|2|TIE)", re.IGNORECASE)


def canonical_args(raw):
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return raw
    return json.dumps(raw, sort_keys=True, separators=(",", ":"))


def call_parts(tool_call):
    fn = tool_call.get("function") or {}
    return fn.get("name") or "", canonical_args(fn.get("arguments"))


def normalized(reply):
    """A reply's comparable form: whitespace-collapsed text, and its tool calls by
    name and canonical arguments (call ids ignored)."""
    text = " ".join((reply.get("content") or "").split())
    calls = [list(call_parts(tc)) for tc in reply.get("tool_calls") or []]
    return [text, calls]


def clip(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\\n[... clipped ...]\\n" + text[-half:]


def render_reply(reply, limit):
    lines = [clip(reply.get("content") or "", limit)]
    for tc in reply.get("tool_calls") or []:
        name, args = call_parts(tc)
        lines.append("-> requests tool call " + name + "(" + clip(args, limit) + ")")
    return "\\n".join(line for line in lines if line) or "(empty reply)"


def parse(text):
    found = _VERDICT.findall(text or "")
    return found[-1].upper() if found else None


def combine(first, second):
    """``first`` is the verdict with the pair's reply as 1, ``second`` with it as 2."""
    if first == "1" and second == "2":
        return "win"
    if first == "2" and second == "1":
        return "loss"
    return "tie"


def prompt(input, reply_1, reply_2):
    limit = input["max_chars"]
    return (
        "SYSTEM PROMPT:\\n" + (input["system"] or "(none)")
        + "\\n\\nCONVERSATION (most recent part):\\n" + (input["conversation"] or "(none)")
        + "\\n\\nREPLY 1:\\n" + render_reply(reply_1, limit)
        + "\\n\\nREPLY 2:\\n" + render_reply(reply_2, limit)
        + "\\n\\nWhich reply is the better next assistant turn?"
    )


async def ask(input, reply_1, reply_2):
    request = {
        "model": input["model"],
        "messages": [
            {"role": "system", "content": RUBRIC},
            {"role": "user", "content": prompt(input, reply_1, reply_2)},
        ],
        "params": input.get("params"),
    }
    for _ in range(2):
        result = await call_llm(request)
        if "error" in result:
            return {"error": result["error"]}
        verdict = parse(result.get("content"))
        if verdict is not None:
            return {"verdict": verdict}
    return {"error": "judge_unparseable"}


async def main(input):
    base = input["base"]
    out = {}
    asks = []
    for name in sorted(input["pairs"]):
        reply = input["pairs"][name]
        if normalized(reply) == normalized(base):
            out[name] = {"outcome": "tie", "identical": True, "orders": []}
            continue
        asks.append((name, reply, base))
        asks.append((name, base, reply))
    answers = await parallel(
        [lambda a=a: ask(input, a[1], a[2]) for a in asks]
    )
    for i in range(0, len(asks), 2):
        name = asks[i][0]
        first, second = answers[i], answers[i + 1]
        errors = [a["error"] for a in (first, second) if "error" in a]
        if errors:
            out[name] = {"outcome": None, "identical": False, "orders": [], "error": errors[0]}
            continue
        orders = [first["verdict"], second["verdict"]]
        out[name] = {"outcome": combine(*orders), "identical": False, "orders": orders}
    return out
'''


def build() -> str:
    return render(SCRIPT, rubric=RUBRIC)
