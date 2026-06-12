"""The ``executes`` execution-class on a registered tool (#988).

A built-in declares where its handler runs — ``"worker"`` (the default, for
owner-agnostic network/credential tools) or ``"sandbox"`` (the filesystem tools
that need a provisioned container). The workflow-run step routes a ``tool('bash')``
call to the run-sandbox executor iff its class is ``"sandbox"``; everything else
goes to the worker tool path.
"""

from __future__ import annotations

from aios.tools.registry import tool_executes_class


def test_sandbox_builtins_execute_in_sandbox() -> None:
    # The six filesystem built-ins run in the provisioned container.
    for name in ("bash", "read", "write", "edit", "glob", "grep"):
        assert tool_executes_class(name) == "sandbox", name


def test_network_builtins_execute_in_worker() -> None:
    # The owner-agnostic network/credential tools run on the worker.
    for name in ("web_search", "web_fetch", "http_request"):
        assert tool_executes_class(name) == "worker", name


def test_unknown_tool_defaults_to_worker() -> None:
    # An unregistered name is class-agnostic — the step's gate rejects it as a
    # value, so the default must not crash the router.
    assert tool_executes_class("nope_not_a_tool") == "worker"
