"""Register the eval workflows in an aios account, in dependency order.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.register \\
        [--bar evals/bars/wam_gate.json]

Each workflow is created if no workflow has its name, updated (a new version) if its
script or declared tools differ, and otherwise left alone. The workflows that call
others have those ids and versions substituted in, and the gate has the bar, so the
printed ``id@version`` of ``wam-gate`` pins everything a gate run uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.client import Api, Client
from evals.workflows import eval_analysis, eval_item, eval_judge, eval_r0, paired_eval

DEFAULT_BAR = Path(__file__).parent / "bars" / "wam_gate.json"


def ensure(
    api: Api, name: str, script: str, tools: list[dict[str, str]], description: str
) -> dict[str, Any]:
    """The workflow called ``name`` with exactly this script and these tools, as
    ``{"id", "version"}``: created, updated to a new version, or as it was."""
    found = [w for w in api.get("/v1/workflows", name=name)["data"] if w["name"] == name]
    if not found:
        created = api.post(
            "/v1/workflows",
            {"name": name, "script": script, "tools": tools, "description": description},
        )
        return {"id": created["id"], "version": created["version"]}
    current = found[0]
    if current["script"] == script and [t["type"] for t in current["tools"]] == [
        t["type"] for t in tools
    ]:
        return {"id": current["id"], "version": current["version"]}
    updated = api.put(
        f"/v1/workflows/{current['id']}",
        {"version": current["version"], "script": script, "tools": tools},
    )
    return {"id": updated["id"], "version": updated["version"]}


def register(api: Api, bar: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r0 = ensure(api, eval_r0.NAME, eval_r0.build(), eval_r0.TOOLS, "Eval arm: one inference.")
    judge = ensure(
        api, eval_judge.NAME, eval_judge.build(), eval_judge.TOOLS, "Eval pairwise judge."
    )
    analysis = ensure(
        api,
        eval_analysis.NAME,
        eval_analysis.build(),
        eval_analysis.TOOLS,
        "Eval statistics and verdict.",
    )
    item = ensure(
        api,
        eval_item.NAME,
        eval_item.build(r0=r0, judge=judge),
        eval_item.TOOLS,
        "Eval of one sampled request: arms, then the judge.",
    )
    gate = ensure(
        api,
        paired_eval.NAME,
        paired_eval.build(bar=bar, item=item, analysis=analysis),
        paired_eval.TOOLS,
        "The workflow-as-model deploy gate.",
    )
    return {
        eval_r0.NAME: r0,
        eval_judge.NAME: judge,
        eval_analysis.NAME: analysis,
        eval_item.NAME: item,
        paired_eval.NAME: gate,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bar", type=Path, default=DEFAULT_BAR)
    args = parser.parse_args(argv)
    registered = register(Client(), json.loads(args.bar.read_text()))
    for name, ref in registered.items():
        print(f"{name:16} {ref['id']}@{ref['version']}")


if __name__ == "__main__":
    main()
