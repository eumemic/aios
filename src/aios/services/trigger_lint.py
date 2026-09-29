"""Conservative static lints for trigger writes.

Two warn-only checks (uncertainty never rejects a write):

- recurring triggers that wake the owner unconditionally;
- standing ``sandbox_command`` triggers that ``exit`` non-zero on purpose,
  which lets a monitor disable itself through the consecutive-failure
  breaker (#2402).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence

UNCONDITIONAL_WAKE_WARNING = (
    "This trigger wakes the owning session on every fire with no condition. "
    "If it is a watchdog, prefer `sandbox_command` that evaluates the condition "
    "and calls `wake_self` only on a real finding — an unconditional wake burns "
    "a model step per fire and trains the reader to ignore it. If it is a "
    "deliberate standing report (e.g. a morning digest), ignore this warning."
)

_GUARD_NODES = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)
_WAKE_RE = re.compile(r"\btool\s+wake_self\b")
OBSERVED_WAKE_WARNING = (
    "This recurring trigger has woken its owning session on nearly every recent fire. "
    "Check that its runtime guard is selective."
)


def observed_wake_is_noisy(outcomes: Sequence[bool]) -> bool:
    """Classify newest-first, 24-hour wake observations.

    Five consecutive wakes catches short noisy histories.  Independently, a
    greater-than-90% wake rate over any non-empty available history is the
    24-hour backstop.  Callers provide only observations from that window.
    """
    if len(outcomes) >= 5 and all(outcomes[:5]):
        return True
    return bool(outcomes) and sum(outcomes) / len(outcomes) > 0.9


def _workflow_has_unconditional_wake(script: str) -> bool:
    tree = ast.parse(script)
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_wake = (isinstance(node.func, ast.Name) and node.func.id == "wake_self") or (
            isinstance(node.func, ast.Name)
            and node.func.id == "tool"
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "wake_self"
        )
        if not is_wake:
            continue
        ancestor = parents.get(node)
        guarded = False
        while ancestor is not None and not isinstance(
            ancestor, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if isinstance(ancestor, _GUARD_NODES):
                guarded = True
                break
            ancestor = parents.get(ancestor)
        if not guarded:
            return True
    return False


def _sandbox_has_unconditional_wake(command: str) -> bool:
    for match in _WAKE_RE.finditer(command):
        prefix = command[: match.start()]
        # Deliberately conservative structural approximation. A preceding open
        # if/case or an && on this command segment dominates the invocation.
        segment = re.split(r"[;\n]", prefix)[-1]
        open_if = len(re.findall(r"\bif\b", prefix)) > len(re.findall(r"\bfi\b", prefix))
        open_case = len(re.findall(r"\bcase\b", prefix)) > len(re.findall(r"\besac\b", prefix))
        if not (open_if or open_case or "&&" in segment):
            return True
    return False


SELF_DISABLING_EXIT_WARNING = (
    "This sandbox_command exits non-zero on purpose (`exit N`, N != 0). Every "
    "non-zero exit counts as a trigger failure, and 5 consecutive failures "
    "auto-disable the trigger. If this is a monitor or heartbeat, the path that "
    "runs when the watched thing is failing will turn the monitor off exactly "
    "when it matters. Report the finding another way (ping a /fail URL, call "
    "`wake_self`, write a log) and `exit 0` on purpose, so only a broken "
    "command can trip the breaker."
)

# ``exit N`` as a shell command word: preceded by start/whitespace/a command
# separator, followed by whitespace/a separator/end. Excludes lookalikes such
# as ``on_exit 1``, ``exit_code=1`` and ``/exit/1``.
_EXIT_RE = re.compile(r"(?:^|(?<=[\s;&|({]))exit[ \t]+(\d+)(?=$|[\s;&|)}])", re.MULTILINE)

# Sources whose failures accumulate toward the auto-disable breaker. A one_shot
# fires once and deletes itself, so it has no breaker to trip.
_STANDING_SOURCES = frozenset({"cron", "run_completion", "external_event"})


def lint_self_disabling_exit(
    *,
    source_kind: str,
    action_kind: str,
    command: str | None = None,
) -> list[str]:
    """Warn when a standing ``sandbox_command`` has an explicit non-zero ``exit``.

    A syntactic probe: it catches the written-on-purpose case (#2402) and
    cannot see a command whose last statement just fails. Warnings only.
    """
    if source_kind not in _STANDING_SOURCES or action_kind != "sandbox_command":
        return []
    if command is None:
        return []
    for match in _EXIT_RE.finditer(command):
        if int(match.group(1)) != 0:
            return [SELF_DISABLING_EXIT_WARNING]
    return []


def lint_unconditional_wake(
    *,
    source_kind: str,
    action_kind: str,
    command: str | None = None,
    workflow_script: str | None = None,
) -> list[str]:
    """Return warnings only; uncertainty never rejects a trigger write."""
    if source_kind != "cron":
        return []
    unconditional = action_kind == "wake_owner"
    if action_kind == "sandbox_command" and command is not None:
        unconditional = _sandbox_has_unconditional_wake(command)
    if action_kind == "workflow" and workflow_script is not None:
        unconditional = _workflow_has_unconditional_wake(workflow_script)
    return [UNCONDITIONAL_WAKE_WARNING] if unconditional else []
