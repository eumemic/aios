"""Opt-in scheduled residue re-scan — between-deploy reproof (#1580, Part B).

Part B of the two ops-agent-owned anti-rot checks (epic #1572). The next-boot
residue scan is the **fail-closed authority**: a retired token reintroduced into
a persisted surface (e.g. a connector-catalog ``tools_schema`` write, or a thaw
that revives an old row) is caught at the next boot and blocks it. But between
boots there is a window: a connector-sourced reintroduction can sit unscanned
until the next deploy.

This module is the **opt-in** cron that narrows that window: an ops-agent job
that runs the *same* registry-derived residue scan against **live prod RO**
between deploys, and alerts on any nonzero residue — so a reintroduction is seen
in cron-period time rather than next-boot time. It is **OFF by default**
(:func:`rescan_enabled`) to preserve the no-new-egress posture: the
CI/ops-to-prod-RO egress is a real cost and stays off unless an operator
explicitly turns it on. Turning it on does NOT change the fail-closed authority —
the next-boot scan still blocks; this cron only *alerts faster*.

The scan plan is built **from the registry surfaces**, never a hand-listed table
set: one query per ``(surface, token)`` using the surface's own
``predicate_sql`` with ``:token`` bound. So a surface added to any descriptor is
scanned by construction — the same totality property the registry gives every
other consumer (validator / boot-gate / migration generator). Nullable surfaces
(e.g. ``sessions.tools``, populated only on frozen children) are ``IS NOT NULL``
guarded before the predicate, mirroring the migration generator's contract.

The DB seam is intentionally tiny: :func:`run_residue_scan` takes any object with
``scalar(sql, params) -> int`` (a thin RO-connection adapter the workflow wires
to prod RO), so the scan logic is unit-testable with a fake and carries no DB
import. ``:token`` is always a *bound parameter* — never string-interpolated —
so a token value can never be a SQL-injection vector.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Protocol

from aios.retirements import Retirement
from aios.retirements.registry import REGISTRY

#: Env var that opts the cron in. Off / absent → no scan (no-new-egress default).
RESCAN_ENABLED_ENV = "AIOS_RETIREMENT_RESCAN_ENABLED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def rescan_enabled(env: dict[str, str] | None = None) -> bool:
    """Whether the opt-in re-scan is enabled — **False unless explicitly on**.

    Reads :data:`RESCAN_ENABLED_ENV`; only an explicit truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) enables it. Absent, empty,
    or any other value keeps it OFF — the default that preserves no-new-egress.
    ``env`` defaults to ``os.environ`` and is injectable for tests.
    """

    source = os.environ if env is None else env
    return source.get(RESCAN_ENABLED_ENV, "").strip().lower() in _TRUTHY


@dataclass(frozen=True)
class ResidueQuery:
    """One ``(surface, token)`` residue probe.

    ``sql`` is a ``SELECT COUNT(*)`` over the surface's table, gated by the
    surface's own ``predicate_sql`` (and an ``IS NOT NULL`` guard for a nullable
    column). ``params`` binds ``:token`` — the token is never interpolated into
    ``sql``.
    """

    table: str
    column: str
    token: str
    sql: str
    params: dict[str, str]


@dataclass(frozen=True)
class ResidueFinding:
    """A surface/token pair with nonzero residue — i.e. an alert.

    ``count`` is the number of rows the surface predicate matched for ``token``.
    A nonzero finding means a retired token is sitting in a live persisted
    surface between deploys; the workflow alerts on any finding.
    """

    table: str
    column: str
    token: str
    count: int


class _ScalarConn(Protocol):
    """The minimal RO-connection seam: run a parameterised count, return the int."""

    def scalar(self, sql: str, params: dict[str, str]) -> int: ...


def iter_residue_queries(
    registry: tuple[Retirement, ...] = REGISTRY,
) -> Iterator[ResidueQuery]:
    """Yield a :class:`ResidueQuery` for every ``(surface, token)`` in ``registry``.

    Derived entirely from the registry's declared surfaces and tokens, so a
    surface added to any descriptor is scanned by construction. The surface's
    ``predicate_sql`` (which already references its own ``jsonb_col`` and binds
    ``:token``) is wrapped in a ``SELECT COUNT(*) ... WHERE <predicate>``; a
    nullable surface is additionally ``<col> IS NOT NULL`` guarded so the
    predicate never evaluates against a NULL JSONB column.
    """

    for retirement in registry:
        for surface in retirement.surfaces:
            guard = f"{surface.jsonb_col} IS NOT NULL AND " if surface.nullable else ""
            for token in retirement.tokens:
                sql = f"SELECT COUNT(*) FROM {surface.table} WHERE {guard}{surface.predicate_sql}"
                yield ResidueQuery(
                    table=surface.table,
                    column=surface.jsonb_col,
                    token=token,
                    sql=sql,
                    params={"token": token},
                )


def run_residue_scan(
    conn: _ScalarConn,
    *,
    registry: tuple[Retirement, ...] = REGISTRY,
) -> list[ResidueFinding]:
    """Run every residue query against ``conn`` and return the nonzero findings.

    ``conn`` is any object exposing ``scalar(sql, params) -> int`` (the workflow
    wires this to a prod **RO** connection). Each ``(surface, token)`` probe runs
    its parameterised ``COUNT(*)``; a count ``> 0`` becomes a
    :class:`ResidueFinding`. An empty result list means the live surfaces are
    clean; a non-empty list is the alert payload the cron raises on.

    This function does NOT consult :func:`rescan_enabled` — gating is the
    workflow's job (it skips the whole job when disabled). Once called, it always
    runs the full plan, so a caller that has decided to scan gets a complete
    verdict.
    """

    findings: list[ResidueFinding] = []
    for query in iter_residue_queries(registry):
        count = conn.scalar(query.sql, query.params)
        if count > 0:
            findings.append(
                ResidueFinding(
                    table=query.table,
                    column=query.column,
                    token=query.token,
                    count=count,
                )
            )
    return findings


# ---------------------------------------------------------------------------
# CLI driver for the OPT-IN scheduled re-scan workflow.
#
# Gated by :func:`rescan_enabled`: if the opt-in flag is OFF (the default), the
# CLI exits 0 immediately WITHOUT connecting anywhere — the no-new-egress posture
# is preserved by simply not running. When enabled, it connects to the prod RO
# DSN, runs the registry-derived scan, prints the findings as JSON, and exits
# nonzero if any residue is found (so the workflow alerts).
# ---------------------------------------------------------------------------

#: Env var carrying the prod **read-only** DSN to scan. Read only when the opt-in
#: flag is on; absent-while-enabled is a hard error (a misconfigured opt-in must
#: not silently scan nothing and report clean).
RESCAN_DSN_ENV = "AIOS_RETIREMENT_RESCAN_DSN"


async def _run_residue_scan_async(
    conn: object,
    *,
    registry: tuple[Retirement, ...] = REGISTRY,
) -> list[ResidueFinding]:
    """Async twin of :func:`run_residue_scan` over a raw asyncpg connection.

    The synchronous :func:`run_residue_scan` is the unit-tested seam (a fake
    ``scalar``); this async variant is what the live CLI uses so it can ``await``
    asyncpg directly without a thread-bridge. It walks the SAME
    :func:`iter_residue_queries` plan, translating the single ``:token`` named
    bind into asyncpg's positional ``$1``. Read-only by construction — every
    query is a ``SELECT COUNT(*)``.
    """

    findings: list[ResidueFinding] = []
    for query in iter_residue_queries(registry):
        pg_sql = query.sql.replace(":token", "$1")
        value = await conn.fetchval(pg_sql, query.token)  # type: ignore[attr-defined]
        count = int(value or 0)
        if count > 0:
            findings.append(
                ResidueFinding(
                    table=query.table,
                    column=query.column,
                    token=query.token,
                    count=count,
                )
            )
    return findings


def _main(argv: list[str]) -> int:
    """Run the opt-in re-scan when enabled; emit findings JSON; exit on residue.

    * OFF (default): print ``{"enabled": false, "findings": []}`` and exit 0 —
      no connection is opened, preserving no-new-egress.
    * ON: connect to the prod RO DSN, run the registry scan, print
      ``{"enabled": true, "findings": [...]}``, and exit 1 if any residue is
      found (the alert) or 0 if the live surfaces are clean.
    """

    if not rescan_enabled():
        json.dump({"enabled": False, "findings": []}, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    dsn = os.environ.get(RESCAN_DSN_ENV, "").strip()
    if not dsn:
        print(
            f"::error::{RESCAN_ENABLED_ENV} is on but {RESCAN_DSN_ENV} is unset; "
            "refusing to report 'clean' without scanning.",
            file=sys.stderr,
        )
        return 2

    import anyio
    import asyncpg

    from aios.db.pool import normalize_dsn

    async def _scan() -> list[ResidueFinding]:
        conn = await asyncpg.connect(normalize_dsn(dsn))
        try:
            return await _run_residue_scan_async(conn)
        finally:
            await conn.close()

    findings = anyio.run(_scan)
    payload = {"enabled": True, "findings": [asdict(f) for f in findings]}
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
