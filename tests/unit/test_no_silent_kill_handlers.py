"""Audit guards against module-level signal manipulation in the server stack.

The silent-worker-exit incident traced to a top-level
``signal.signal(SIGPIPE, SIG_DFL)`` in a CLI module that the worker
transitively imported.  Two checks pin this class of bug:
boot-imports keep ``SIG_IGN`` at runtime, and no module under
``src/aios/`` carries a module-scope ``signal.signal(SIGPIPE,
SIG_DFL)`` in its AST.
"""

from __future__ import annotations

import ast
import importlib.util
import signal
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER_BOOT_IMPORTS: tuple[str, ...] = (
    "aios.__main__",
    "aios.cli.app",
    "aios.cli.runtime",
    "aios.cli.commands.ops",
    "aios.harness.worker",
    "aios.harness.procrastinate_app",
    "aios.harness.exit_diagnostics",
    "aios.harness.loop",
    "aios.harness.connector_supervisor",
    "aios.harness.runtime",
    "aios.mcp.pool",
    "aios.mcp.stdio_transport",
    "aios.config",
    "aios.logging",
)

# Cheap typo guard at collection time: confirms each name resolves on
# the import path without actually importing (which would run module
# bodies — including ``Settings()`` validation in procrastinate_app).
for _name in _WORKER_BOOT_IMPORTS:
    assert importlib.util.find_spec(_name) is not None, f"missing module: {_name}"


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="SIGPIPE not available")
def test_sigpipe_handler_stays_default_for_server_imports() -> None:
    """A clean subprocess that imports the worker boot path keeps SIGPIPE
    at Python's default ``SIG_IGN``.  Any future module-level handler
    install in a server-reachable file fails this test."""
    lines = ["import signal"]
    lines.extend(f"import {m}" for m in _WORKER_BOOT_IMPORTS)
    lines.append("print(int(signal.getsignal(signal.SIGPIPE)))")
    result = subprocess.run(
        [sys.executable, "-c", "\n".join(lines)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, f"boot-imports failed: stderr={result.stderr!r}"
    assert result.stdout.strip() == str(int(signal.SIG_IGN)), (
        f"a server-reachable module mutated the SIGPIPE handler. "
        f"got handler={result.stdout.strip()!r}"
    )


def _module_level_calls(tree: ast.Module) -> list[ast.Call]:
    """Yield ``ast.Call`` nodes whose enclosing scope is the module body.

    Descends through module-level ``If``/``Try``/``With`` chains at the
    statement level so ``if hasattr(...): signal.signal(...)`` is still
    caught — that's exactly the pattern the audit targets — but stops
    at function/class boundaries so calls inside nested ``def``/``class``
    bodies don't get falsely flagged.
    """
    calls: list[ast.Call] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            calls.append(node.value)
        elif isinstance(node, ast.If):
            for child in (*node.body, *node.orelse):
                visit(child)
        elif isinstance(node, ast.Try):
            for child in (*node.body, *node.orelse, *node.finalbody):
                visit(child)
            for handler in node.handlers:
                for child in handler.body:
                    visit(child)
        elif isinstance(node, ast.With):
            for child in node.body:
                visit(child)

    for node in tree.body:
        visit(node)
    return calls


def _is_signal_signal_sigpipe_sig_dfl(call: ast.Call) -> bool:
    """Match the canonical ``signal.signal(signal.SIGPIPE, signal.SIG_DFL)``."""
    func = call.func
    if not (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "signal"
        and func.attr == "signal"
    ):
        return False
    if len(call.args) != 2:
        return False
    sig, handler = call.args
    return (
        isinstance(sig, ast.Attribute)
        and isinstance(sig.value, ast.Name)
        and sig.value.id == "signal"
        and sig.attr == "SIGPIPE"
        and isinstance(handler, ast.Attribute)
        and isinstance(handler.value, ast.Name)
        and handler.value.id == "signal"
        and handler.attr == "SIG_DFL"
    )


def test_no_module_level_sigpipe_sig_dfl_in_aios() -> None:
    """No ``src/aios/**/*.py`` module may install SIGPIPE → SIG_DFL at
    module scope.  Even an ``if hasattr(signal, "SIGPIPE")`` guard at
    module level fires at import time and leaks the process-fatal
    handler into every entrypoint that imports the file.  Move installs
    into a function (e.g. ``aios.cli.runtime.run_or_die``) so only
    explicit client-CLI invocations get the kill-on-broken-pipe shape.
    """
    src_root = Path(__file__).resolve().parents[2] / "src" / "aios"
    assert src_root.is_dir(), f"expected source root at {src_root}"

    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        tree = ast.parse(py.read_text(), filename=str(py))
        for call in _module_level_calls(tree):
            if _is_signal_signal_sigpipe_sig_dfl(call):
                offenders.append(f"{py.relative_to(src_root.parent.parent)}:{call.lineno}")

    assert not offenders, "module-level signal.signal(SIGPIPE, SIG_DFL) found:\n" + "\n".join(
        f"  - {o}" for o in offenders
    )
