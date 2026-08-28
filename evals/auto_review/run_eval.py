"""Operator-run eval for the MCP auto-review checker prompt (jarbot#229).

Runs the REAL prompt builder and verdict parser from
``aios.harness.auto_review`` against the REAL checker model over the
scenario corpus in ``cases.json``, and reports verdict agreement. This is
the behavioral half of the spec's eval list; the deterministic half
(timeout → ask, junk → ask, always-allow skips the checker, …) lives in
``tests/unit/test_auto_review.py`` and ``tests/e2e/test_auto_review_flow.py``.

NOT CI: it spends real tokens and its outcome depends on the live model.
Run it when tuning the prompt, swapping ``AIOS_AUTO_REVIEW_MODEL``, or
investigating a bad verdict from the field.

Usage (repo root; needs a provider key for the checker model in the env,
e.g. ``OPENAI_API_KEY`` — auth here is the litellm env fallback, not the
per-account ladder):

    uv run python evals/auto_review/run_eval.py [--model <litellm-model>] [--repeat N]

Exit code 0 iff every case matched on every repeat.
"""

from __future__ import annotations

import argparse
import asyncio
import json

# Placeholder env for aios.config import-time validation, matching
# scripts/regen-openapi.sh — nothing connects to these.
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("AIOS_API_KEY", "x")
os.environ.setdefault("AIOS_VAULT_KEY", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=")
os.environ.setdefault("AIOS_EGRESS_CA_KEY", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE=")
os.environ.setdefault("AIOS_DB_URL", "postgresql://x@localhost/x")

from aios.config import get_settings
from aios.harness.auto_review import _build_messages, _parse_verdict
from aios.harness.completion import LlmRequest, call_litellm

CASES_PATH = Path(__file__).parent / "cases.json"


def _call_dict(case: dict[str, Any]) -> dict[str, Any]:
    name = f"mcp__{case['server']}__{case['tool']}"
    return {"id": "eval", "function": {"name": name, "arguments": json.dumps(case["args"])}}


async def _run_case(case: dict[str, Any], model: str) -> tuple[str, str]:
    surface = SimpleNamespace(description=case["description"])
    messages = _build_messages(surface, _call_dict(case), case["user_lines"])  # type: ignore[arg-type]
    request = LlmRequest(
        messages=messages,
        tools=None,
        params={"timeout": 30.0, "max_tokens": 300},
        session_id=None,
    )
    response = await call_litellm(request, model=model, auth=None)
    parsed = _parse_verdict(response.content)
    if parsed is None:
        return "junk", repr(response.content)[:200]
    return parsed[0], parsed[1]


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="override the checker model")
    parser.add_argument("--repeat", type=int, default=1, help="runs per case")
    parser.add_argument("--only", default=None, help="run a single case by name")
    args = parser.parse_args()

    model = args.model or get_settings().auto_review_model
    if not model:
        print(
            "no checker model — pass --model or set AIOS_AUTO_REVIEW_MODEL",
            file=sys.stderr,
        )
        return 2
    cases = json.loads(CASES_PATH.read_text())
    if args.only:
        cases = [c for c in cases if c["name"] == args.only]
        if not cases:
            print(f"no case named {args.only!r}", file=sys.stderr)
            return 2

    print(f"model: {model}   cases: {len(cases)}   repeat: {args.repeat}\n")
    failures = 0
    for case in cases:
        for i in range(args.repeat):
            verdict, reason = await _run_case(case, model)
            ok = verdict == case["expected"]
            failures += 0 if ok else 1
            tag = "PASS" if ok else "FAIL"
            rep = f" [{i + 1}/{args.repeat}]" if args.repeat > 1 else ""
            print(f"{tag}  {case['name']}{rep}: expected {case['expected']}, got {verdict}")
            print(f"      reason: {reason}")
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURES'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
