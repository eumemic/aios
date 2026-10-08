"""The eval workflows, as script templates.

Each module holds one workflow's script text (``SCRIPT``), its registered name and
the run tools it declares. A workflow script can't import shared code, so a value
that differs per registration (the bar, the ids and versions of the workflows it
calls) is substituted into the text: a ``__NAME__`` token becomes a Python string
literal holding the value as JSON, which the script reads with ``json.loads``.
Registering a template with other values makes a new workflow version, so a change
to the bar shows up in the workflow's history.

The workflows:

* ``eval_r0``: one inference, the baseline and negative-control arm.
* ``eval_judge``: a pairwise judge, in both orders.
* ``eval_item``: one sampled request end to end: three arms, then the judge.
* ``eval_analysis``: the statistics, pure compute.
* ``paired_eval``: the gate run: sample, check power, run items in waves, analyse.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOKEN = re.compile(r"__[A-Z][A-Z0-9_]*__")


def render(template: str, **values: Any) -> str:
    """Substitute each ``__NAME__`` token with ``values[name]`` as a JSON string literal.

    Every token in the template must be given a value and every value must match a
    token, so a renamed placeholder can't silently leave a stale one behind."""
    out = template
    for name, value in values.items():
        token = f"__{name.upper()}__"
        if token not in out:
            raise KeyError(f"template has no {token}")
        out = out.replace(token, repr(json.dumps(value, sort_keys=True)))
    left = sorted(set(_TOKEN.findall(out)))
    if left:
        raise KeyError(f"template tokens left unfilled: {', '.join(left)}")
    return out


def load(script: str) -> dict[str, Any]:
    """Execute a rendered script's top level and return its namespace, so its pure
    helpers can be called outside a run (by tests and the launcher's estimates). The
    capabilities aren't bound, so ``main`` can't run this way.

    Only pass text built here from these templates. Script text read back from the
    API is whatever the workflow's last updater wrote, so callers compare it with a
    local build first (``evals.gate.trusted``) and never execute it."""
    namespace: dict[str, Any] = {"__name__": "<workflow>"}
    exec(compile(script, "<workflow>", "exec", dont_inherit=True), namespace)
    return namespace
