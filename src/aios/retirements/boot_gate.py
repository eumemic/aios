"""The fail-closed boot-admission gate (#1575, epic #1572).

This is the *keystone* provability invariant of the pattern-retirement
mechanism. It moves the "is this DB safe for this code to read?" proof OFF the
migration — which in prod runs **POST-deploy**, after the new container already
serves (``aios migrate`` is the api app's post-deployment command; the
eumemic-ops audit FAILS the build if it is configured pre-deploy) — and ONTO the
code itself, at startup, *before* readiness flips green.

:func:`assert_retirements_admissible` is called from BOTH the api and the worker
startup path, gating readiness. For every contracted descriptor (one whose
``contract_rev`` is set, i.e. whose data migration has been written) it proves:

1. **alembic head**: the live ``alembic_version`` is ``>= contract_rev``. If the
   DB is behind — the post-deploy migrate has not run yet, OR a restore pointed
   new code at an old DB — the proof FAILS. This is the startup alembic-head
   assertion that did not exist before (the only prior startup version check was
   the unrelated wf-version drift check at ``workflows/service.py``).
2. **live residue == 0**: once the rev is satisfied, the per-surface residue
   aggregate ``SUM over descriptor.surfaces of count(rows WHERE predicate(token))``
   must be zero. If any surface still carries a retired token the proof FAILS and
   an *algedonic* alert is emitted (a migration that was supposed to have
   rewritten those rows did not).
3. **fail-closed**: any DB / connection failure FAILS the proof. No connection
   ⇒ not admissible ⇒ not ready.

The caller (api lifespan / worker startup) loops on a raised
:class:`RetirementsNotAdmissible`, keeping the health/readiness signal RED and
*never serving*, while under Coolify's rolling deploy the old healthy container
keeps serving. This converts "new code meets old rows" from a per-row crash-loop
(the #1525 failure) into a single clean readiness-refusal BEFORE any request is
served. The proof is re-executed on **every** boot (including raw restores) and
never cached.

The gate reads the contracted descriptors and their surface predicates from the
registry (:mod:`aios.retirements.registry`) and nothing else; if a surface is
missing there it is missing here, by construction.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from aios.logging import get_logger
from aios.retirements import Retirement
from aios.retirements.registry import REGISTRY

log = get_logger("aios.retirements.boot_gate")


class RetirementsNotAdmissible(RuntimeError):
    """The boot-admission proof failed; the process must NOT flip ready.

    Raised by :func:`assert_retirements_admissible` for every refusal reason —
    behind-DB (alembic_version < contract_rev), nonzero live residue on a
    registered surface, or a fail-closed DB/connection error. The caller treats
    any instance identically: refuse readiness, hold, and retry on the next
    loop tick. The distinct subclasses exist only to make the *reason* legible
    in logs and tests; callers should catch the base class.
    """


class DatabaseBehindContract(RetirementsNotAdmissible):
    """``alembic_version`` is behind a descriptor's ``contract_rev``.

    The post-deploy migrate has not run yet, or a restore pointed new code at an
    old DB. Refuse readiness and loop; the old container keeps serving until the
    migrate lands.
    """


class LiveResidueDetected(RetirementsNotAdmissible):
    """A registered surface still carries a retired token after its contract.

    The contract migration was supposed to have rewritten these rows to
    canonical but did not (it never ran everywhere, or a stale row slipped in).
    Serving would re-introduce the #1525 per-row read crash-loop, so refuse
    readiness and emit an algedonic alert.
    """


class DatabaseUnavailable(RetirementsNotAdmissible):
    """No DB connection / a query raised: fail-closed ⇒ not admissible."""


def _contracted(registry: tuple[Retirement, ...]) -> list[Retirement]:
    """Descriptors whose data migration has been written (``contract_rev`` set).

    A descriptor with no ``contract_rev`` has no migration yet — there is
    nothing to assert "the rows have been rewritten by" — so it is not gated.
    """

    return [r for r in registry if r.contract_rev is not None]


def _bind_token_param(predicate_sql: str) -> str:
    """Rewrite the descriptor's ``:token`` named bind to asyncpg's ``$1``.

    Surface predicates are declared with a SQLAlchemy-style ``:token`` named
    parameter (one shared template per domain); asyncpg uses positional ``$N``
    binds. Every tool-surface predicate references ``:token`` exactly once, so a
    literal substitution to ``$1`` is exact and order-stable.
    """

    return predicate_sql.replace(":token", "$1")


async def _current_alembic_version(conn: asyncpg.Connection[Any]) -> str | None:
    """Return the live ``alembic_version.version_num`` (``None`` if unstamped).

    A fresh DB with no ``alembic_version`` table (never migrated) reads as
    ``None`` — which is unconditionally behind any ``contract_rev``, so the gate
    refuses, exactly as it must for an unmigrated DB.
    """

    table = await conn.fetchval("SELECT to_regclass('alembic_version')")
    if table is None:
        return None
    version: str | None = await conn.fetchval("SELECT version_num FROM alembic_version LIMIT 1")
    return version


async def assert_retirements_admissible(pool: asyncpg.Pool[Any]) -> None:
    """Prove the live DB is safe for this code, or refuse readiness.

    Raises :class:`RetirementsNotAdmissible` (one of its subclasses) on any
    refusal reason; returns ``None`` when every contracted descriptor's proof
    passes. The caller loops on the exception, keeping readiness RED.

    For each contracted descriptor:

    1. ``alembic_version >= contract_rev`` — else :class:`DatabaseBehindContract`.
       Revisions are 4-digit zero-padded strings on a linear chain, so a string
       comparison is the same as the migration order.
    2. ``SUM over surfaces of count(rows matching the token predicate) == 0`` —
       else :class:`LiveResidueDetected` plus an algedonic alert.

    Any DB/connection failure raises :class:`DatabaseUnavailable` (fail-closed).
    """

    contracted = _contracted(REGISTRY)
    if not contracted:
        # Nothing contracted yet ⇒ nothing to prove ⇒ admissible. (The gate is
        # still wired in, so the proof goes live the instant a descriptor gains
        # a contract_rev.)
        return

    try:
        async with pool.acquire() as conn:
            version = await _current_alembic_version(conn)

            # (1) alembic-head assertion. The DB must be at or past EVERY
            # contracted descriptor's contract_rev before any residue scan is
            # even meaningful — a behind DB hasn't run the rewrite yet, so its
            # rows are expected to still carry the token.
            for retirement in contracted:
                contract_rev = retirement.contract_rev
                assert contract_rev is not None  # _contracted filtered these
                if version is None or version < contract_rev:
                    log.info(
                        "boot_gate.behind_contract",
                        domain=retirement.domain,
                        alembic_version=version,
                        contract_rev=contract_rev,
                    )
                    raise DatabaseBehindContract(
                        f"DB at alembic_version={version!r} is behind "
                        f"contract_rev={contract_rev!r} for domain "
                        f"{retirement.domain!r}; refusing readiness until the "
                        "post-deploy migrate runs"
                    )

            # (2) per-surface live-residue aggregate. The rev is satisfied, so
            # the rewrite was supposed to have run; any surviving token is a
            # real, alert-worthy contract breach.
            for retirement in contracted:
                await _assert_no_residue(conn, retirement)
    except RetirementsNotAdmissible:
        raise
    except (asyncpg.PostgresError, OSError, ConnectionError, TimeoutError) as exc:
        # Fail-closed: no DB / a query that raised ⇒ not admissible. The caller
        # loops and retries; never serve on an unprovable DB.
        log.warning("boot_gate.db_unavailable", error=str(exc))
        raise DatabaseUnavailable(
            f"cannot prove retirement admissibility (DB unavailable): {exc}"
        ) from exc


async def _assert_no_residue(conn: asyncpg.Connection[Any], retirement: Retirement) -> None:
    """Assert every surface of ``retirement`` carries zero rows of any token.

    The aggregate is per-surface and per-token: for every ``(surface, token)``
    pair, count the rows matching the surface's predicate bound to that token,
    summed across the descriptor. A nonzero total on ANY surface fails the
    proof. ``nullable`` columns are guarded with ``IS NOT NULL`` so the scan
    never trips over an absent JSONB column.
    """

    tokens = retirement.tokens
    breaches: list[dict[str, Any]] = []
    for surface in retirement.surfaces:
        predicate = _bind_token_param(surface.predicate_sql)
        where = predicate
        if surface.nullable:
            where = f"{surface.jsonb_col} IS NOT NULL AND ({predicate})"
        sql = f"SELECT count(*) FROM {surface.table} WHERE {where}"
        for token in tokens:
            count = await conn.fetchval(sql, token)
            if count:
                breaches.append(
                    {
                        "table": surface.table,
                        "jsonb_col": surface.jsonb_col,
                        "token": token,
                        "count": int(count),
                    }
                )

    if breaches:
        # Algedonic alert: a contract migration that was supposed to have
        # rewritten these rows to canonical did not. ``algedonic=True`` marks
        # the highest-severity operator pain signal; serving here would
        # re-open the #1525 per-row read crash-loop.
        log.error(
            "boot_gate.live_residue",
            algedonic=True,
            domain=retirement.domain,
            contract_rev=retirement.contract_rev,
            breaches=breaches,
        )
        raise LiveResidueDetected(
            f"live residue on {len(breaches)} surface/token pair(s) for domain "
            f"{retirement.domain!r} (contract_rev={retirement.contract_rev!r}): "
            f"{breaches!r}; refusing readiness"
        )
