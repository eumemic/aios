"""Structural contract for the core-to-connector import boundary."""

from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).parents[2] / "src" / "aios"
CONNECTOR_PACKAGE = "aios_connectors"


def _top_level_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """Return imports evaluated at module load, excluding definition bodies."""
    imports: list[ast.Import | ast.ImportFrom] = []

    def visit(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Import, ast.ImportFrom)):
                imports.append(statement)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            else:
                for child in ast.iter_child_nodes(statement):
                    if isinstance(child, ast.stmt):
                        visit([child])

    visit(tree.body)
    return imports


def test_core_has_no_top_level_connector_imports() -> None:
    violations: list[str] = []
    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _top_level_imports(tree):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            if any(
                module == CONNECTOR_PACKAGE or module.startswith(f"{CONNECTOR_PACKAGE}.")
                for module in modules
            ):
                violations.append(f"{path.relative_to(CORE_ROOT.parent)}:{node.lineno}")

    assert not violations, (
        "aios.* must not import aios_connectors.* at module top level; "
        "keep sanctioned crossings function-scoped:\n" + "\n".join(violations)
    )
