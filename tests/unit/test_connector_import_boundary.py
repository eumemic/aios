"""Guard the core-to-connector module import boundary."""

from __future__ import annotations

import ast
from pathlib import Path

CORE_ROOT = Path(__file__).parents[2] / "src" / "aios"
CONNECTOR_PACKAGE = "aios_connectors"


def _module_scope_imports(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    imports: list[ast.Import | ast.ImportFrom] = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.ClassDef):
            # A class body executes while its enclosing module imports. Traverse
            # that body (and the class declaration expressions), while the
            # function guard above still excludes method-local imports.
            for decorator in node.decorator_list:
                visit(decorator)
            for base in node.bases:
                visit(base)
            for keyword in node.keywords:
                visit(keyword.value)
            for statement in node.body:
                visit(statement)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return imports


def _connector_import(node: ast.Import | ast.ImportFrom) -> str | None:
    if isinstance(node, ast.Import):
        return next(
            (
                alias.name
                for alias in node.names
                if alias.name.split(".", 1)[0] == CONNECTOR_PACKAGE
            ),
            None,
        )
    if node.module and node.module.split(".", 1)[0] == CONNECTOR_PACKAGE:
        return node.module
    return None


def test_module_scope_import_finder_excludes_only_local_scopes() -> None:
    tree = ast.parse(
        """
if TYPE_CHECKING:
    import aios_connectors.types

class Adapter:
    import aios_connectors.class_body

    def method(self):
        import aios_connectors.method_local

def startup():
    import aios_connectors.function_local
"""
    )

    assert [node.lineno for node in _module_scope_imports(tree)] == [3, 6]


def test_core_modules_do_not_import_connectors_at_module_scope() -> None:
    """Connector crossings must remain function-local to keep core importable alone."""
    violations: list[str] = []

    for path in sorted(CORE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in _module_scope_imports(tree):
            imported = _connector_import(node)
            if imported:
                violations.append(f"{path.relative_to(CORE_ROOT.parent)}:{node.lineno}: {imported}")

    assert not violations, "Core modules import aios_connectors at module scope:\n" + "\n".join(
        violations
    )
