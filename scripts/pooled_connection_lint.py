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


def _expression_name(node: ast.AST) -> str | None:
    """Return an exact dotted expression; never collapse ``self.conn`` to ``self``."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _expression_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


# Explicit repository DB surfaces.  Additions are reviewable: arbitrary names,
# suffixes, and service objects are deliberately not inferred as database I/O.
_DB_HELPER_QUALIFIERS = frozenset({"queries", "_queries", "wf_queries", "trace_q"})
_DB_HELPER_SYMBOLS = frozenset({
    '_advance_open_request_scan_floor_best_effort',
    '_allocate_version_seq',
    '_append_fire_event',
    '_append_transition',
    '_archive_binding_or_raise',
    '_assert_env_var_creds_contained',
    '_assert_no_residue',
    '_batch_list_all_echoes',
    '_cancel_run',
    '_classify_existing_tool_result',
    '_complete_run',
    '_current_alembic_version',
    '_dedup_skip',
    '_enrich_agent_result',
    '_enrich_session',
    '_errored_session_ids',
    '_fail_child_requests_for_terminal_error',
    '_insert_workflow_version',
    '_journal_agent_rejection',
    '_latest_cumulative_state',
    '_list_all_echoes',
    '_list_all_writable_store_ids',
    '_list_attached_resource_ids',
    '_load_for_session_conn',
    '_load_surfaces',
    '_open_agent_capability',
    '_open_invoke_workflow_capability',
    '_quiescence_owed_surfacing',
    '_record_timer_audit',
    '_rekey_column',
    '_resolve_agent_call',
    '_session_owned',
    '_walk',
    'accounts_service.resolve_effective_timezone_on',
    'agents_service.load_for_session',
    'agents_service.validate_pinned_agent_version',
    'append_tool_result',
    'archive_session_conn',
    'attach_to_session',
    'calibration_telemetry',
    'db_queries.insert_run_completion_fires',
    'db_queries.insert_session_cancel_marker',
    'db_queries.write_response_if_absent',
    'fail_open_child_requests_conn',
    'find_parked_servicer',
    'find_unharvested_model_dispatch_parks',
    'get_agent',
    'get_environment',
    'get_session_bare',
    'get_session_vault_ids',
    'get_session_workspace_path',
    'get_workflow',
    'github_repo_service.add_one',
    'github_repo_service.attach_to_session',
    'github_repo_service.detach_all_from_session',
    'github_repo_service.get_session_token',
    'github_repo_service.remove_one',
    'github_repo_service.set_session_resources',
    'list_session_ids_with_unharvested_cancel_marker',
    'materialize_store_to_host',
    'memory_service.add_one',
    'memory_service.attach_to_session',
    'memory_service.remove_one',
    'memory_service.set_session_resources',
    'prune',
    'read_request_response',
    'resolve_effective_spend_limit_usd_on',
    'resolve_effective_timezone_on',
    'resolve_run_env_var_credentials',
    'resolve_session_env_var_credentials',
    'respond_to_request_conn',
    'seed_outbound_cancel_conn',
    'service.append_tool_result',
    'session_has_pending_work',
    'sessions_service.append_tool_result',
    'sessions_service.archive_session_conn',
    'sessions_service.respond_to_request_conn',
    'triggers_service.validate_trigger_spec',
    'validate_trigger_spec',
    'write_gate_opened',
})


class _Checker(ast.NodeVisitor):
    def __init__(self, filename: str, lines: list[str]) -> None:
        self.filename = filename
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
                receiver = _expression_name(expression.func.value)
                if receiver is not None:
                    transaction_connections.add(receiver)

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
        if isinstance(expression.func, ast.Attribute):
            receiver = _expression_name(expression.func.value)
            if receiver in held:
                return True
        symbol = _expression_name(expression.func)
        if symbol is None:
            return False
        qualifier = symbol.rpartition(".")[0]
        if symbol not in _DB_HELPER_SYMBOLS and qualifier not in _DB_HELPER_QUALIFIERS:
            return False
        values = [*expression.args, *(keyword.value for keyword in expression.keywords)]
        return sum(_expression_name(value) in held for value in values) == 1

    def _has_linked_pragma(self, node: ast.Await) -> bool:
        line = self.lines[node.lineno - 1]
        return _PRAGMA in line and _ISSUE_REF.search(line) is not None


def check_source(source: str, *, filename: str = "<unknown>") -> list[Violation]:
    tree = ast.parse(source, filename=filename)
    checker = _Checker(filename, source.splitlines())
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
