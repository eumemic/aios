#!/usr/bin/env python3
"""Filter aios sessions stream output to a one-line-per-message chat view."""

from __future__ import annotations

import json
import sys


def fmt_attachments(atts: list[dict]) -> list[str]:
    out = []
    for a in atts:
        out.append(
            "      ATT: {fn!r} {ct} {sz}B in_sandbox={p}".format(
                fn=a.get("filename"),
                ct=a.get("content_type"),
                sz=a.get("size"),
                p=a.get("in_sandbox_path"),
            )
        )
    return out


for raw in sys.stdin:
    raw = raw.strip()
    if not raw or not raw.startswith("{"):
        continue
    try:
        outer = json.loads(raw)
    except json.JSONDecodeError:
        continue
    # `aios sessions stream --raw` emits SSE-shape:
    # {"event":"event","data":"<JSON-string of the actual event>"}
    # for backfill + live frames.  Unwrap once; events from the static
    # `sessions events` array path are already at top level so fall back
    # to the outer dict if there's no nested data string.
    inner = outer.get("data")
    if isinstance(inner, str):
        try:
            e = json.loads(inner)
        except json.JSONDecodeError:
            continue
    else:
        e = outer
    if e.get("kind") != "message":
        continue
    data = e.get("data") or {}
    role = data.get("role")
    seq = e.get("seq")
    md = data.get("metadata") or {}

    if role == "user":
        content = data.get("content", "")
        sender = md.get("sender_name") or md.get("sender") or "?"
        channel = md.get("channel", "?")
        if isinstance(content, list):
            text = next((p.get("text", "") for p in content if p.get("type") == "text"), "")
            kinds = ",".join(p.get("type", "?") for p in content)
            print(f"[{seq}] USER<{sender}@{channel}> [{kinds}] {text[:200]}", flush=True)
        else:
            print(f"[{seq}] USER<{sender}@{channel}> {str(content)[:300]}", flush=True)
        for line in fmt_attachments(md.get("attachments") or []):
            print(line, flush=True)

    elif role == "assistant":
        content = data.get("content", "")
        if isinstance(content, str) and content.startswith("INTERNAL_MONOLOGUE"):
            mono = content.split(":", 1)[1].strip()[:140]
            content = f"[monologue: {mono}]"
        elif content:
            content = (content if isinstance(content, str) else str(content))[:200]
        for tc in data.get("tool_calls") or []:
            f = tc.get("function") or {}
            args_str = f.get("arguments", "")
            try:
                args = json.loads(args_str)
                args_summary = ", ".join(f"{k}={str(v)[:80]!r}" for k, v in list(args.items())[:3])
            except json.JSONDecodeError:
                args_summary = args_str[:200]
            print(
                f"[{seq}] ASSISTANT call {f.get('name', '?')}({args_summary})",
                flush=True,
            )
        if content:
            print(f"[{seq}] ASSISTANT {content}", flush=True)

    elif role == "tool":
        c = data.get("content", "")
        name = data.get("name", "?")
        if isinstance(c, list):
            kinds = ",".join(p.get("type", "?") for p in c)
            print(f"[{seq}] TOOL {name} -> [{kinds}]", flush=True)
        else:
            preview = (c if isinstance(c, str) else str(c))[:200]
            print(f"[{seq}] TOOL {name} -> {preview}", flush=True)
