"""Shared fixtures for the in-sandbox ``bin/tool`` CLI tests.

The CLI script has no ``.py`` extension and is conventionally invoked
as an executable; ``importlib.util.spec_from_file_location`` is the
most direct way to import it for unit testing. The script's
``if __name__ == "__main__":`` guard means importing doesn't execute
``main()``. Centralised here so test files don't reproduce the loader
incantation.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def tool_module() -> ModuleType:
    repo_root = Path(__file__).parents[3]
    script_path = repo_root / "bin" / "tool"
    loader = importlib.machinery.SourceFileLoader("tool_cli", str(script_path))
    spec = importlib.util.spec_from_file_location("tool_cli", script_path, loader=loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
