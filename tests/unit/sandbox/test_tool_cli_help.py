"""Pinned wording for ``bin/tool``'s per-tool ``--help`` INVOKE line.

Issue #675: the legacy ``mcp`` CLI's per-tool ``--help`` once printed
``mcp <server> <tool> --json '{...}'`` even though ``--json`` was never
a real flag — pasting the printed example failed with ``unknown option:
--json``. The flag was dropped (commit f8aefa3) and the binary later
renamed to ``tool`` (commit 57a0747), but no test pinned the wording,
so a future copy-edit could resurrect the same divergence between
documentation and parser. These tests lock in the positional form.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def tool_module() -> ModuleType:
    """Load ``bin/tool`` as a python module (same pattern as
    ``test_tool_cli_defaults.py`` — script has no ``.py`` extension)."""
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "bin" / "tool"
    loader = importlib.machinery.SourceFileLoader("tool_cli", str(script_path))
    spec = importlib.util.spec_from_file_location("tool_cli", script_path, loader=loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBuiltinHelpInvokeLine:
    def test_invoke_line_uses_positional_form(
        self, tool_module: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tool_module._print_help_builtin(
            "web_fetch", {"description": "Fetch a URL.", "input_schema": {"type": "object"}}
        )
        out = capsys.readouterr().out
        assert "INVOKE:" in out
        assert "tool web_fetch '{...}'" in out

    def test_invoke_line_does_not_advertise_json_flag(
        self, tool_module: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tool_module._print_help_builtin(
            "web_fetch", {"description": "Fetch a URL.", "input_schema": {"type": "object"}}
        )
        out = capsys.readouterr().out
        assert "--json" not in out


class TestMcpHelpInvokeLine:
    def test_invoke_line_uses_positional_form(
        self, tool_module: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tool_module._print_help_mcp(
            "github",
            "get_me",
            {"description": "Get the authenticated user.", "input_schema": {"type": "object"}},
        )
        out = capsys.readouterr().out
        assert "INVOKE:" in out
        assert "tool github get_me '{...}'" in out

    def test_invoke_line_does_not_advertise_json_flag(
        self, tool_module: ModuleType, capsys: pytest.CaptureFixture[str]
    ) -> None:
        tool_module._print_help_mcp(
            "github",
            "get_me",
            {"description": "Get the authenticated user.", "input_schema": {"type": "object"}},
        )
        out = capsys.readouterr().out
        assert "--json" not in out
