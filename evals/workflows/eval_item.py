"""``eval_item``: one sampled request, end to end.

Invoked by the gate with the item's ref (its grant) and::

    {"request_ref", "item": {"session_id", "created_at", "model"},
     "agent": {"agent_id", "version"}, "baseline_model",
     "candidate": {"workflow_id", "version"},
     "judge": {"model", "params", "tail", "max_chars"},
     "budget": {"arm_usd", "candidate_usd", "judge_usd"}, "attempt"}

1. Read the request rendered for the baseline model (``get_request``).
2. Run the arms in parallel, each a sub-run acting for the agent (``as_agent``) with
   its own budget:

   * the baseline, ``eval_r0`` sending the ref for the baseline model;
   * the candidate, the pinned candidate workflow given the ref and no input, so it
     starts with the request as a workflow bound as the agent's model would;
   * the negative control, ``eval_r0`` given only the system prompt and the last user
     turn, only when that is strictly less than the judge's tail and the tail strictly
     less than the whole conversation.

3. Read the arms' facts (``sub_runs()``): the models each called, their cost at
   uncached rates, their duration, and the workflows and agents the candidate ran.
   If the judge's model family is one an arm used, or a model is unknown, stop: the
   record says ``family_violation``.
4. Judge the candidate and the control against the baseline, then check the judge's
   own models the same way.

Returns ``{"status": "ok", "record": {...}}``; ``{"status": "retry", "reason"}`` when
load the eval caused (a provider overload, a full run cap) hit any arm, so the gate
runs the whole item again later; or ``{"status": "excluded", "reason"}``. A candidate
error or an output that isn't an assistant turn is a loss and a degenerate turn. The
record holds no request or reply text.
"""

from __future__ import annotations

from evals.workflows import render

NAME = "eval-item"
TOOLS: list[dict[str, str]] = [{"type": "get_request"}]

SCRIPT = '''
import json
import re

CONFIG = json.loads(__CONFIG__)

# Errors the eval's own load causes. Any arm hitting one reruns the whole item, so
# load never turns into a loss for one arm only.
_OVERLOAD = re.compile(
    r"rate.?limit|overloaded|\\b429\\b|\\b503\\b|\\b529\\b|timed out|timeout|"
    r"service.?unavailable|apiconnectionerror",
    re.IGNORECASE,
)
_BUDGET = "budget exhausted"
_REFUSAL = frozenset({"content_filter", "refusal"})

# Router prefixes that say how a model is reached, not who made it.
_ROUTERS = frozenset(
    {"openrouter", "bedrock", "bedrock_converse", "vertex_ai", "vertex_ai_beta", "azure",
     "azure_ai", "together_ai", "fireworks_ai", "groq", "deepinfra", "litellm_proxy"}
)
_VENDORS = (
    ("anthropic", ("anthropic", "claude")),
    ("openai", ("openai", "gpt", "chatgpt", "o1", "o3", "o4", "codex")),
    ("google", ("google", "gemini", "gemma", "palm")),
    ("meta", ("meta", "meta-llama", "llama")),
    ("mistral", ("mistral", "mistralai", "mixtral", "codestral", "magistral")),
    ("moonshot", ("moonshot", "moonshotai", "kimi")),
    ("zhipu", ("zhipu", "z-ai", "zai", "glm")),
    ("xai", ("xai", "x-ai", "grok")),
    ("deepseek", ("deepseek",)),
    ("qwen", ("qwen", "alibaba")),
    ("cohere", ("cohere", "command")),
    ("amazon", ("amazon", "nova")),
    ("microsoft", ("microsoft", "phi")),
    ("nvidia", ("nvidia", "nemotron")),
)


def family(model):
    """The lab that made ``model``, or None when it can't be told (which the caller
    treats as an overlap: fail closed)."""
    if not isinstance(model, str) or not model or model.startswith("workflow:"):
        return None
    parts = model.lower().split("/")
    while len(parts) > 1 and parts[0] in _ROUTERS:
        parts = parts[1:]
    # The vendor segment (openai/gpt-..., or bedrock's anthropic.claude-...), then the
    # model name itself (gpt-..., claude-...).
    for name in (parts[0].split(".", 1)[0], parts[-1]):
        for vendor, aliases in _VENDORS:
            if any(name.startswith(alias) for alias in aliases):
                return vendor
    return None


# ── the request's windows ─────────────────────────────────────────────────────


def split_system(messages):
    i = 0
    while i < len(messages) and messages[i].get("role") == "system":
        i += 1
    return messages[:i], messages[i:]


def last_user(rest):
    for i in range(len(rest) - 1, -1, -1):
        if rest[i].get("role") == "user":
            return i
    return 0


def judge_tail(rest, size):
    """The last ``size`` messages, cut forward to start at a user message so no tool
    result is orphaned; never less than the last user turn."""
    start = max(len(rest) - size, 0)
    while start < len(rest) and rest[start].get("role") != "user":
        start += 1
    return rest[min(start, last_user(rest)):]


def text_of(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for part in content if isinstance(content, list) else [content]:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text") or "")
        elif isinstance(part, dict):
            parts.append("[" + str(part.get("type") or "part") + "]")
        else:
            parts.append(str(part))
    return "\\n".join(parts)


def clip(text, limit):
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\\n[... clipped ...]\\n" + text[-half:]


def render_messages(messages, limit):
    lines = []
    for m in messages:
        role = m.get("role") or "?"
        body = clip(text_of(m.get("content")), limit)
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            body += "\\n-> tool call " + str(fn.get("name")) + "(" + clip(
                str(fn.get("arguments") or ""), limit
            ) + ")"
        lines.append("[" + role + "] " + body)
    return "\\n\\n".join(lines)


# ── an arm's output, as the binding boundary reads it ─────────────────────────


def as_turn(output):
    """Read an arm's output the way a workflow bound as a model is read: an object
    with string-or-null ``content``, a list-of-objects ``tool_calls`` and an
    object-or-null ``message``. Returns ``{"content", "tool_calls", "finish_reason"}``,
    or None when the output isn't an assistant turn."""
    if not isinstance(output, dict):
        return None
    content = output.get("content")
    if content is not None and not isinstance(content, str):
        return None
    calls = output.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list) or not all(isinstance(tc, dict) for tc in calls):
        return None
    message = output.get("message")
    if message is not None and not isinstance(message, dict):
        return None
    content = content or ""
    inner = output.get("finish_reason")
    if inner in _REFUSAL or (not content and not calls):
        finish = "content_filter"
    elif inner == "length":
        finish = "length"
    else:
        finish = "tool_calls" if calls else "stop"
    return {"content": content, "tool_calls": calls, "finish_reason": finish}


def calls_valid(calls, offered):
    """Each requested call names an offered tool, its arguments parse to an object,
    and every argument the tool's schema requires is there."""
    schemas = {}
    for t in offered or []:
        fn = t.get("function") or {}
        if fn.get("name"):
            schemas[fn["name"]] = fn.get("parameters") or {}
    for tc in calls:
        fn = tc.get("function") or {}
        if fn.get("name") not in schemas:
            return False
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                return False
        if not isinstance(args, dict):
            return False
        for key in schemas[fn["name"]].get("required") or []:
            if key not in args:
                return False
    return True


def error_text(arm):
    if "raised" in arm:
        return arm["raised"]["message"]
    out = arm["output"]
    if isinstance(out, dict) and "error" in out:
        return str(out["error"])
    return None


# ── the arms' facts ───────────────────────────────────────────────────────────


def tree(facts):
    nodes = facts["nodes"]
    ids = {n["id"] for n in nodes}
    roots = {n["parent"]["id"] for n in nodes if n["parent"]["id"] not in ids}
    children = {}
    for n in nodes:
        children.setdefault(n["parent"]["id"], []).append(n)
    return nodes, children, roots


def labelled(facts, label):
    """The direct child of this run with ``label`` (a candidate's own sub-runs can't
    pose as an arm, since they aren't direct children)."""
    nodes, children, roots = tree(facts)
    for root in sorted(roots):
        for n in children.get(root, []):
            if n.get("label") == label:
                return n
    return None


def subtree(facts, node):
    nodes, children, _ = tree(facts)
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        out.append(n)
        stack.extend(children.get(n["id"], []))
    return out


def summarize(facts, label):
    node = labelled(facts, label)
    if node is None:
        return None
    nodes = subtree(facts, node)
    models, workflows, agents = set(), set(), set()
    cost, uncached = 0, 0
    for n in nodes:
        if n["kind"] == "session":
            models.add(n.get("model"))
            agents.add(str(n.get("agent_id")) + "@" + str(n.get("agent_version")))
        else:
            workflows.add(str(n.get("workflow_id")) + "@" + str(n.get("workflow_version")))
        for u in n["usage"]:
            models.add(u["model"])
            cost += u["cost_microusd"] or 0
            if uncached is not None:
                uncached = None if u.get("uncached_cost_microusd") is None else (
                    uncached + u["uncached_cost_microusd"]
                )
    return {
        # A model the ledger didn't record reads "<unknown>", which has no family.
        "models": sorted(m if m is not None else "<unknown>" for m in models),
        "workflows": sorted(workflows),
        "agents": sorted(agents),
        "cost_microusd": cost,
        "uncached_cost_microusd": uncached,
        "duration_ms": node.get("duration_ms"),
    }


def violation(facts, judge_families, summaries):
    """Whether a judge family overlaps an arm's, or a model can't be placed."""
    if facts.get("truncated") or None in judge_families:
        return True
    for s in summaries:
        if s is None:
            continue
        for m in s["models"]:
            f = family(m)
            if f is None or f in judge_families:
                return True
    return False


# ── the item ──────────────────────────────────────────────────────────────────


async def arm(workflow, input, *, ref, agent, label, budget, version):
    try:
        out = await invoke_workflow(
            workflow,
            input,
            version=version,
            request_ref=ref,
            as_agent=agent,
            label=label,
            budget_usd=budget,
        )
    except AgentError as e:
        return {"raised": {"kind": e.kind, "message": str(e)}}
    return {"output": out}


def arm_record(result, offered):
    err = error_text(result)
    turn = None if err is not None else as_turn(result["output"])
    rec = {
        "error_kind": None,
        "invalid_output": err is None and turn is None,
        "degenerate": True,
        "tool_calls_valid": False,
        "n_tool_calls": 0,
        "chars": 0,
    }
    if err is not None:
        kind = result["raised"]["kind"] if "raised" in result else None
        rec["error_kind"] = kind or ("budget" if _BUDGET in err else "error")
        return rec, None
    if turn is None:
        return rec, None
    rec["degenerate"] = turn["finish_reason"] == "content_filter" or (
        turn["finish_reason"] == "length" and not turn["tool_calls"]
    )
    rec["tool_calls_valid"] = calls_valid(turn["tool_calls"], offered)
    rec["n_tool_calls"] = len(turn["tool_calls"])
    rec["chars"] = len(turn["content"]) + sum(
        len(json.dumps(tc.get("function") or {}, sort_keys=True)) for tc in turn["tool_calls"]
    )
    return rec, {"content": turn["content"], "tool_calls": turn["tool_calls"]}


def eval_caused(result):
    """A launch the run cap refused, or a provider overload: the eval's own load."""
    if "raised" in result and result["raised"]["kind"] == "invoke_workflow_refused":
        return True
    err = error_text(result)
    return err is not None and _OVERLOAD.search(err) is not None


async def main(input):
    ref = input["request_ref"]
    agent = input["agent"]
    baseline = input["baseline_model"]
    cand = input["candidate"]
    judge = input["judge"]
    budget = input["budget"]
    r0 = CONFIG["r0"]

    request = await tool("get_request", {"request_ref": ref, "model": baseline})
    if "error" in request:
        return {"status": "excluded", "reason": "unavailable"}
    system, rest = split_system(request["messages"])
    if not rest:
        return {"status": "excluded", "reason": "empty"}
    neg_window = rest[last_user(rest):]
    tail = judge_tail(rest, judge["tail"])
    control = len(neg_window) < len(tail) < len(rest)

    arms = [
        lambda: arm(r0["id"], {"model": baseline, "request_ref": ref}, ref=ref, agent=agent,
                    label="arm:base", budget=budget["arm_usd"], version=r0["version"]),
        lambda: arm(cand["workflow_id"], None, ref=ref, agent=agent, label="arm:cand",
                    budget=budget["candidate_usd"], version=cand["version"]),
    ]
    if control:
        neg_input = {
            "model": baseline,
            "messages": system + neg_window,
            "tools": request["tools"],
            "params": request["params"],
        }
        arms.append(
            lambda: arm(r0["id"], neg_input, ref=None, agent=agent, label="arm:neg",
                        budget=budget["arm_usd"], version=r0["version"])
        )
    results = await parallel(arms)
    base, cand_result = results[0], results[1]
    neg = results[2] if control else None

    if any(eval_caused(r) for r in results):
        return {"status": "retry", "reason": "overload"}
    for r in results:
        if "raised" in r and r["raised"]["kind"] == "invoke_workflow_forbidden":
            return {"status": "excluded", "reason": "forbidden"}

    offered = request["tools"]
    base_rec, base_reply = arm_record(base, offered)
    if base_reply is None:
        reason = "baseline_budget" if base_rec["error_kind"] == "budget" else "baseline_error"
        return {"status": "excluded", "reason": reason}
    cand_rec, cand_reply = arm_record(cand_result, offered)
    neg_rec, neg_reply = (None, None)
    if neg is not None:
        neg_rec, neg_reply = arm_record(neg, offered)

    facts = await sub_runs()
    arms_facts = {
        name: summarize(facts, "arm:" + name)
        for name in (["base", "cand", "neg"] if control else ["base", "cand"])
    }
    judge_families = {family(judge["model"])}
    if violation(facts, judge_families, list(arms_facts.values())):
        return {"status": "ok", "record": {"family_violation": True, "ref": ref}}

    pairs = {}
    if cand_reply is not None:
        pairs["cand"] = cand_reply
    if neg_reply is not None:
        pairs["neg"] = neg_reply
    verdicts = {}
    if pairs:
        judge_input = {
            "model": judge["model"],
            "params": judge.get("params"),
            "max_chars": judge["max_chars"],
            "system": "\\n\\n".join(text_of(m.get("content")) for m in system),
            "conversation": render_messages(tail, judge["max_chars"]),
            "base": base_reply,
            "pairs": pairs,
        }
        try:
            verdicts = await invoke_workflow(
                CONFIG["judge"]["id"],
                judge_input,
                version=CONFIG["judge"]["version"],
                label="judge",
                budget_usd=budget["judge_usd"],
            )
        except AgentError as e:
            if _OVERLOAD.search(str(e)):
                return {"status": "retry", "reason": "overload"}
            return {"status": "excluded", "reason": "judge_error"}
        for v in verdicts.values():
            if v.get("error") is not None:
                if _OVERLOAD.search(v["error"]):
                    return {"status": "retry", "reason": "overload"}
                return {"status": "excluded", "reason": "judge_error"}
        facts = await sub_runs()
        judged = summarize(facts, "judge")
        judge_models = set(judged["models"]) if judged is not None else set()
        arm_families = {
            family(m) for s in arms_facts.values() if s is not None for m in s["models"]
        }
        if judged is None or facts.get("truncated") or any(
            family(m) is None or family(m) in arm_families for m in judge_models
        ):
            return {"status": "ok", "record": {"family_violation": True, "ref": ref}}

    cand_outcome = verdicts["cand"]["outcome"] if "cand" in verdicts else "loss"
    neg_outcome = verdicts["neg"]["outcome"] if "neg" in verdicts else None

    def with_facts(rec, name):
        if rec is None:
            return None
        s = arms_facts.get(name) or {}
        return dict(
            rec,
            cost_microusd=s.get("cost_microusd"),
            uncached_cost_microusd=s.get("uncached_cost_microusd"),
            duration_ms=s.get("duration_ms"),
            models=s.get("models", []),
        )

    cand_facts = arms_facts["cand"] or {}
    return {
        "status": "ok",
        "record": {
            "family_violation": False,
            "ref": ref,
            "cluster": input["item"]["session_id"] + "|" + input["item"]["created_at"][:10],
            "fidelity": request["fidelity"],
            "control_eligible": control,
            "outcomes": {"cand": cand_outcome, "neg": neg_outcome},
            "identical": {
                "cand": bool(verdicts.get("cand", {}).get("identical")),
                "neg": bool(verdicts.get("neg", {}).get("identical")),
            },
            "orders": {k: v.get("orders", []) for k, v in sorted(verdicts.items())},
            "arms": {
                "base": with_facts(base_rec, "base"),
                "cand": with_facts(cand_rec, "cand"),
                "neg": with_facts(neg_rec, "neg"),
            },
            "candidate_resolved": {
                "workflows": cand_facts.get("workflows", []),
                "agents": cand_facts.get("agents", []),
                "models": cand_facts.get("models", []),
            },
            "judge_models": sorted(judge_models) if pairs else [],
        },
    }
'''


def build(*, r0: dict[str, object], judge: dict[str, object]) -> str:
    """``r0`` and ``judge`` are ``{"id", "version"}`` of the registered workflows."""
    return render(SCRIPT, config={"r0": r0, "judge": judge})
