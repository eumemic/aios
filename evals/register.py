"""Register the eval workflows in an aios account, in dependency order.

    AIOS_URL=... AIOS_API_KEY=<operator key> uv run python -m evals.register \\
        [evals/bars/wam_gate.json ...]

Each workflow is created if no workflow has its name, updated (a new version) if its
script or declared tools differ, and otherwise left alone. The workflows that call
others have those ids and versions substituted in, and each gate has its bar, so the
printed ``id@version`` of a gate pins everything a run of it uses.

Every bar file is its own gate, named after the file: ``bars/wam_gate.json`` is
``wam-gate``, ``bars/wam_gate_fusion.json`` is ``wam-gate-fusion``. A recipe class
whose cost or latency differs by design (a fusion recipe's fan-out) is gated by its
own bar, so loosening a limit is a separate registered gate with its own history,
never an edit to the default. With no arguments, every bar in ``bars/`` is registered.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.client import Api, Client
from evals.workflows import eval_analysis, eval_item, eval_judge, eval_r0, paired_eval

BARS = Path(__file__).parent / "bars"


def gate_name(bar_path: Path) -> str:
    """The gate a bar file registers as: its file name, with dashes."""
    return bar_path.stem.replace("_", "-")


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


def register(api: Api, bars: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Register the shared workflows, then one gate per ``{name: bar}``."""
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
    registered = {
        eval_r0.NAME: r0,
        eval_judge.NAME: judge,
        eval_analysis.NAME: analysis,
        eval_item.NAME: item,
    }
    for name, bar in sorted(bars.items()):
        registered[name] = ensure(
            api,
            name,
            paired_eval.build(bar=bar, item=item, analysis=analysis),
            paired_eval.TOOLS,
            "A workflow-as-model deploy gate.",
        )
    return registered


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bars", type=Path, nargs="*", help="bar files (default: all in bars/)")
    args = parser.parse_args(argv)
    paths = args.bars or sorted(BARS.glob("*.json"))
    registered = register(Client(), {gate_name(p): json.loads(p.read_text()) for p in paths})
    for name, ref in registered.items():
        print(f"{name:16} {ref['id']}@{ref['version']}")


if __name__ == "__main__":
    main()
