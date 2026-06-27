"""Tests for the opt-in scheduled residue re-scan (#1580, Part B; epic #1572).

Part B is an ops-agent cron that runs the registry's residue scan against **live
prod RO** between deploys, to catch connector-sourced / thaw reintroduction of a
retired token faster than the next-boot scan. It is **opt-in / off by default**
to preserve the no-new-egress posture (the CI-to-prod-RO egress is a cost kept
off unless explicitly turned on); the next-boot scan remains the fail-closed
authority.

These tests pin two things:

* the scan plan is built **from the registry surfaces** — one query per
  ``(surface, token)``, using the surface's own ``predicate_sql`` with ``:token``
  bound — so a surface added to a descriptor is scanned by construction (no
  hand-listed table set to drift); and
* the cron is **OFF by default** and only runs when explicitly enabled, and when
  it runs it **alerts on nonzero** residue.
"""

from __future__ import annotations

import pytest

from aios.retirements import Retirement, Surface
from aios.retirements.residue_scan import (
    ResidueFinding,
    ResidueQuery,
    iter_residue_queries,
    rescan_enabled,
    run_residue_scan,
)


def _ret() -> Retirement:
    return Retirement(
        domain="tool_surface",
        action="rename",
        mappings=(("invoke", "call_session"), ("create_run", "call_workflow")),
        surfaces=(
            Surface(
                table="agents",
                jsonb_col="tools",
                predicate_sql="EXISTS(SELECT 1 FROM jsonb_array_elements(tools) e "
                "WHERE e->>'type' = :token)",
            ),
            Surface(
                table="connectors",
                jsonb_col="tools_schema",
                predicate_sql="EXISTS(SELECT 1 FROM jsonb_array_elements(tools_schema) e "
                "WHERE e->>'type' = :token)",
                nullable=True,
            ),
        ),
        introduced_rev="0100",
        contract_rev=None,
        sla_days=30,
    )


# ── scan-plan construction (from the registry surfaces) ──────────────────────


def test_one_query_per_surface_and_token() -> None:
    queries = list(iter_residue_queries((_ret(),)))
    # 2 surfaces x 2 tokens = 4 queries.
    assert len(queries) == 4
    pairs = {(q.table, q.column, q.token) for q in queries}
    assert pairs == {
        ("agents", "tools", "invoke"),
        ("agents", "tools", "create_run"),
        ("connectors", "tools_schema", "invoke"),
        ("connectors", "tools_schema", "create_run"),
    }


def test_query_uses_surface_predicate_and_binds_token() -> None:
    q = next(
        q for q in iter_residue_queries((_ret(),)) if q.table == "agents" and q.token == "invoke"
    )
    assert isinstance(q, ResidueQuery)
    # The COUNT(*) wrapper selects rows the surface predicate matches.
    assert "COUNT(*)" in q.sql
    assert "FROM agents" in q.sql
    assert "jsonb_array_elements(tools)" in q.sql
    # :token is a bound parameter, never string-interpolated.
    assert ":token" in q.sql
    assert q.params == {"token": "invoke"}


def test_nullable_surface_is_null_guarded() -> None:
    q = next(
        q
        for q in iter_residue_queries((_ret(),))
        if q.table == "connectors" and q.token == "invoke"
    )
    assert "tools_schema IS NOT NULL" in q.sql


def test_non_nullable_surface_is_not_null_guarded() -> None:
    q = next(
        q for q in iter_residue_queries((_ret(),)) if q.table == "agents" and q.token == "invoke"
    )
    assert "IS NOT NULL" not in q.sql


# ── opt-in gating (OFF by default) ───────────────────────────────────────────


def test_rescan_disabled_by_default() -> None:
    # No env set → opt-in is OFF. This is the no-new-egress default.
    assert rescan_enabled(env={}) is False


def test_rescan_disabled_for_falsey_values() -> None:
    for val in ("0", "false", "False", "no", "off", ""):
        assert rescan_enabled(env={"AIOS_RETIREMENT_RESCAN_ENABLED": val}) is False


def test_rescan_enabled_only_when_explicitly_turned_on() -> None:
    for val in ("1", "true", "True", "yes", "on"):
        assert rescan_enabled(env={"AIOS_RETIREMENT_RESCAN_ENABLED": val}) is True


# ── running the scan (alerts on nonzero) ─────────────────────────────────────


class _FakeConn:
    """Minimal stand-in for a RO DB connection.

    ``counts`` maps a ``(table, token)`` to the residue count the predicate
    returns; anything absent counts as zero.
    """

    def __init__(self, counts: dict[tuple[str, str], int]) -> None:
        self.counts = counts
        self.seen: list[tuple[str, dict[str, str]]] = []

    def scalar(self, sql: str, params: dict[str, str]) -> int:
        self.seen.append((sql, params))
        # Recover (table, token) from the query to look up the canned count.
        token = params["token"]
        table = sql.split("FROM ", 1)[1].split()[0]
        return self.counts.get((table, token), 0)


def test_clean_scan_returns_no_findings() -> None:
    conn = _FakeConn(counts={})
    findings = run_residue_scan(conn, registry=(_ret(),))
    assert findings == []
    # Every (surface, token) query was actually executed against the connection.
    assert len(conn.seen) == 4


def test_nonzero_residue_becomes_a_finding() -> None:
    conn = _FakeConn(counts={("connectors", "invoke"): 3})
    findings = run_residue_scan(conn, registry=(_ret(),))
    assert len(findings) == 1
    f = findings[0]
    assert isinstance(f, ResidueFinding)
    assert f.table == "connectors"
    assert f.column == "tools_schema"
    assert f.token == "invoke"
    assert f.count == 3


def test_multiple_residue_rows_all_reported() -> None:
    conn = _FakeConn(counts={("agents", "invoke"): 1, ("connectors", "create_run"): 2})
    findings = run_residue_scan(conn, registry=(_ret(),))
    assert {(f.table, f.token, f.count) for f in findings} == {
        ("agents", "invoke", 1),
        ("connectors", "create_run", 2),
    }


def test_real_registry_builds_queries() -> None:
    # The seeded registry produces a non-empty, well-formed scan plan.
    from aios.retirements.registry import REGISTRY

    queries = list(iter_residue_queries(REGISTRY))
    assert queries, "registry should yield residue queries"
    for q in queries:
        assert q.params == {"token": q.token}
        assert ":token" in q.sql
        assert "COUNT(*)" in q.sql


# ── CLI opt-in gating (off-by-default no-ops without connecting) ─────────────


def test_cli_off_by_default_noops(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import json as _json

    from aios.retirements import residue_scan as rs

    monkeypatch.delenv(rs.RESCAN_ENABLED_ENV, raising=False)
    rc = rs._main(["residue_scan"])
    assert rc == 0
    out = _json.loads(capsys.readouterr().out)
    assert out == {"enabled": False, "findings": []}


def test_cli_enabled_without_dsn_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aios.retirements import residue_scan as rs

    monkeypatch.setenv(rs.RESCAN_ENABLED_ENV, "1")
    monkeypatch.delenv(rs.RESCAN_DSN_ENV, raising=False)
    # Enabled but no DSN → refuse to report 'clean' without scanning.
    assert rs._main(["residue_scan"]) == 2


# ── async variant over a fake asyncpg connection ─────────────────────────────


class _FakeAsyncpgConn:
    """Fake asyncpg connection: ``fetchval($1-sql, token)`` → canned count."""

    def __init__(self, counts: dict[tuple[str, str], int]) -> None:
        self.counts = counts
        self.seen: list[tuple[str, str]] = []

    async def fetchval(self, sql: str, token: str) -> int:
        self.seen.append((sql, token))
        assert "$1" in sql and ":token" not in sql  # named bind translated
        table = sql.split("FROM ", 1)[1].split()[0]
        return self.counts.get((table, token), 0)


def test_async_scan_translates_bind_and_reports_nonzero() -> None:
    import anyio

    from aios.retirements.residue_scan import _run_residue_scan_async

    conn = _FakeAsyncpgConn(counts={("agents", "invoke"): 5})
    findings = anyio.run(lambda: _run_residue_scan_async(conn, registry=(_ret(),)))
    assert {(f.table, f.token, f.count) for f in findings} == {("agents", "invoke", 5)}
    assert len(conn.seen) == 4
