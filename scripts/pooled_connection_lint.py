#!/usr/bin/env python3
"""Reject non-database awaits while an asyncpg pool connection is held."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_PRAGMA = "pooled-connection-await: allow"
_ISSUE_REF = re.compile(r"(?<![A-Za-z0-9_.-])eumemic/aios#(\d+)(?!\d)")


@dataclass(frozen=True)
class Violation:
    filename: str
    line: int
    column: int
    message: str

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}: PCA001 {self.message}"


def _root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _Checker(ast.NodeVisitor):
    def __init__(self, filename: str, lines: list[str], local_db_helpers: set[str]) -> None:
        self.filename = filename
        self.local_db_helpers = local_db_helpers
        self.lines = lines
        self.held: list[set[str]] = []
        self.violations: list[Violation] = []

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        acquired: set[str] = set()
        transaction_connections: set[str] = set()
        for item in node.items:
            expression = item.context_expr
            if not isinstance(expression, ast.Call) or not isinstance(
                expression.func, ast.Attribute
            ):
                continue
            if expression.func.attr == "acquire" and isinstance(item.optional_vars, ast.Name):
                acquired.add(item.optional_vars.id)
            elif expression.func.attr == "transaction":
                root = _root_name(expression.func.value)
                if root is not None:
                    transaction_connections.add(root)

        names = acquired | transaction_connections
        if names:
            self.held.append(names)
            for statement in node.body:
                self.visit(statement)
            self.held.pop()
            for item in node.items:
                self.visit(item.context_expr)
            return
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        held = set().union(*self.held) if self.held else set()
        if held and not self._is_db_await(node.value, held) and not self._has_linked_pragma(node):
            connection = sorted(held)[0]
            self.violations.append(
                Violation(
                    self.filename,
                    node.lineno,
                    node.col_offset + 1,
                    f"pooled connection '{connection}' held across non-DB await",
                )
            )
        self.generic_visit(node)

    def _is_db_await(self, expression: ast.AST, held: set[str]) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        # Classification is by the CALLED OBJECT, never merely by seeing a
        # connection somewhere in the argument list.  In particular,
        # ``await arbitrary_network_call(conn)`` is foreign I/O, not DB I/O.
        if isinstance(expression.func, ast.Attribute) and _root_name(expression.func.value) in held:
            return True

        # The repository's DB helper surface lives below ``queries``.  Helpers
        # must receive the held connection as their first positional argument;
        # no second connection argument or ``conn=`` smuggling is accepted.
        root = _root_name(expression.func)
        called = (
            expression.func.attr
            if isinstance(expression.func, ast.Attribute)
            else (expression.func.id if isinstance(expression.func, ast.Name) else "")
        )
        recognized = (
            (
                isinstance(expression.func, ast.Name)
                and (
                    called in self.local_db_helpers
                    or called.startswith(
                        (
                            "get_",
                            "list_",
                            "find_",
                            "read_",
                            "resolve_",
                            "validate_",
                            "materialize_",
                            "session_",
                            "calibration_",
                            "prune",
                            "append_",
                            "agents_",
                            "write_",
                        )
                    )
                )
            )
            or root == "queries"
            or bool(
                root
                and (
                    root.endswith(("queries", "_queries", "_service", "_q"))
                    or root in {"queries", "trace_q", "wf_queries", "db_queries", "service"}
                )
            )
            or called.endswith("_conn")
        )
        if not recognized:
            return False
        values = [*expression.args, *(keyword.value for keyword in expression.keywords)]
        return sum(isinstance(value, ast.Name) and value.id in held for value in values) == 1

    def _has_linked_pragma(self, node: ast.Await) -> bool:
        line = self.lines[node.lineno - 1]
        return _PRAGMA in line and _ISSUE_REF.search(line) is not None


def check_source(source: str, *, filename: str = "<unknown>") -> list[Violation]:
    tree = ast.parse(source, filename=filename)
    local_db_helpers = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.args.args
        and node.args.args[0].arg in {"conn", "connection", "acquired"}
    }
    checker = _Checker(filename, source.splitlines(), local_db_helpers)
    checker.visit(tree)
    return checker.violations


def check_paths(paths: list[Path]) -> list[Violation]:
    violations: list[Violation] = []
    for root in paths:
        files = [root] if root.is_file() else sorted(root.rglob("*.py"))
        for path in files:
            violations.extend(check_source(path.read_text(), filename=str(path)))
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    violations = check_paths(args.paths)
    for violation in violations:
        print(violation)
    return bool(violations)


if __name__ == "__main__":
    sys.exit(main())
