"""Conservative static lint for recurring triggers that wake unconditionally."""

from __future__ import annotations

import ast
import re

UNCONDITIONAL_WAKE_WARNING = (
    "This trigger wakes the owning session on every fire with no condition. "
    "If it is a watchdog, prefer `sandbox_command` that evaluates the condition "
    "and calls `wake_self` only on a real finding — an unconditional wake burns "
    "a model step per fire and trains the reader to ignore it. If it is a "
    "deliberate standing report (e.g. a morning digest), ignore this warning."
)

_GUARD_NODES = (ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match)
_WAKE_RE = re.compile(r"\btool\s+wake_self\b")


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
