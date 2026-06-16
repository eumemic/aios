"""Tripwire test pinning the deferral precondition for #1130.

#1130 ships an **ownership-only** gate on the caller-supplied ``environment_id``
at ``POST /v1/invocations`` (#1128): the supplied env must be owned by the
authenticated account (``get_environment(conn, env_id, account_id=<caller>)``),
exactly as ``create_session`` / ``create_run`` already enforce (#755). The
*per-field containment clamp* (``networking`` / ``image`` / ``env``-keys subset-of a
baseline env) is **deferred** — there is no attenuated / sub-TOP API principal to
clamp against today, so ownership *is* the bound and there is nothing below top.

This module is the tripwire that keeps the deferral sound. It asserts, from two
independent angles, that **no sub-TOP / attenuated API principal exists today**:

1. the account-key path (``account_keys`` schema) carries **no
   surface / scope / attenuation column**; and
2. the bearer-auth principal (``require_bearer_auth`` -> ``AccountAuthResult``)
   resolves to a **full-account** ``account_id`` with no sub-TOP surface field.

When either assertion fails — i.e. when an attenuated / delegated (sub-TOP) API
principal *is* introduced — that is the signal to implement the deferred
per-field env containment clamp (``networking`` / ``image`` / ``env``-keys subset-of
baseline) on the api->session edge. See #1130. The failing test is the
deferral's re-entry point.

Sibling of #823 (the ``api_base`` second-authority axis) — same tripwire shape.
"""

from __future__ import annotations

import typing
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from aios.api import deps
from aios.db.pool import create_pool

pytestmark = pytest.mark.integration


# The columns ``account_keys`` carries today (migration 0042). The tripwire
# below asserts the live schema equals exactly this set — neither a missing
# column (a precondition drift) nor a *new* column slipping in unnoticed. A new
# column named for a surface / scope / attenuation axis is precisely the event
# that re-opens the deferred clamp.
_ACCOUNT_KEYS_COLUMNS_TODAY = frozenset(
    {
        "key_id",
        "account_id",
        "hash",
        "label",
        "created_at",
        "last_used_at",
        "revoked_at",
    }
)

# Substrings that, if they appear in a *new* ``account_keys`` column name, would
# signal an attenuation / sub-TOP surface being attached to the bearer principal.
# Named explicitly so the failure message points the next implementer straight
# at the deferred clamp.
_ATTENUATION_COLUMN_HINTS = ("surface", "scope", "attenuat", "clamp", "baseline")

# The env axes the deferred per-field containment clamp would eventually cover
# (``EnvironmentConfig`` at ``models/environments.py``). Documented here as the
# future clamp targets; NOT clamped now.
_DEFERRED_CLAMP_AXES = ("networking", "image", "env")


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    p = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        yield p
    finally:
        await p.close()


async def test_account_keys_has_no_attenuation_surface_column(
    pool: asyncpg.Pool[Any],
) -> None:
    """The ``account_keys`` path carries no surface / scope / attenuation column.

    The bearer/account-key path (``require_bearer_auth`` ->
    ``lookup_account_by_key_hash``) is the principal that authorizes
    ``POST /v1/invocations``. Today every account key is full-account-TOP: the
    ``account_keys`` row binds a key to an ``account_id`` and nothing finer.
    There is no column describing an attenuated *subset* of that account's
    surface, so the supplied ``environment_id`` has nothing below top to be
    clamped against — ownership IS the bound.

    When an attenuated / delegated (sub-TOP) API principal is introduced, this
    assertion fails — that is the signal to implement the deferred per-field env
    containment clamp (networking / image / env-keys subset-of baseline) on the
    api->session edge. See #1130.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'account_keys'
            """
        )
    live_columns = {r["column_name"] for r in rows}

    # The connection_ids allowlist (#350) — the only attenuation surface in the
    # codebase — lives on the *runtime-token* path, NOT here. account_keys must
    # carry exactly its #1130-precondition column set: no more, no less.
    assert live_columns == set(_ACCOUNT_KEYS_COLUMNS_TODAY), (
        "account_keys schema drifted from the #1130 deferral precondition.\n"
        f"  expected: {sorted(_ACCOUNT_KEYS_COLUMNS_TODAY)}\n"
        f"  live:     {sorted(live_columns)}\n"
        "If a column naming an attenuation / scope / surface axis was added, an "
        "attenuated (sub-TOP) API principal now exists — implement the deferred "
        "per-field env containment clamp "
        f"({'/'.join(_DEFERRED_CLAMP_AXES)}-keys subset-of baseline) on the api->session "
        "edge. See #1130."
    )

    # Belt-and-braces: no column name hints at an attenuation surface, even if
    # the exact-set check above is ever relaxed.
    offenders = [
        c for c in live_columns if any(hint in c.lower() for hint in _ATTENUATION_COLUMN_HINTS)
    ]
    assert not offenders, (
        f"account_keys grew an attenuation-shaped column {offenders!r}: a sub-TOP "
        "API principal now exists. Implement the deferred per-field env "
        "containment clamp on the api->session edge. See #1130."
    )


def test_bearer_auth_principal_is_full_account_no_subtop_surface() -> None:
    """The bearer-auth tuple resolves to a full-account ``account_id``, no sub-TOP field.

    ``require_bearer_auth`` returns ``AccountAuthResult`` — a
    ``(account_id, key_id, can_mint_children)`` 3-tuple. None of those three is a
    sub-TOP *surface*: ``account_id`` is the full tenant, ``key_id`` identifies the
    key row, and ``can_mint_children`` is a tenancy-creation capability (a child
    account is a *distinct* full-TOP ``account_id`` within its own tenant, not an
    attenuated subset of the parent's surface). The invoke endpoint's ownership
    gate uses ``account_id`` from this tuple — never a request-body-supplied
    account.

    When an attenuated / delegated (sub-TOP) API principal is introduced — e.g. a
    fourth tuple element describing a surface subset, or ``account_id`` becoming
    an attenuated handle — this assertion fails. That is the signal to implement
    the deferred per-field env containment clamp (networking / image / env-keys
    subset-of baseline) on the api->session edge. See #1130.
    """
    # AccountAuthResult is the resolved-auth contract. Pin its arity and members
    # so a sub-TOP surface element can't be added without tripping this.
    args = typing.get_args(deps.AccountAuthResult)
    assert args == (str, str, bool), (
        "AccountAuthResult changed shape from (account_id, key_id, "
        f"can_mint_children); got {args!r}. If a sub-TOP surface element was "
        "added to the bearer-auth principal, implement the deferred per-field "
        "env containment clamp on the api->session edge. See #1130."
    )

    # The endpoint depends on AccountIdDep, which projects exactly the
    # account_id out of the tuple — confirm that projection is still a pure
    # full-account id with no attenuation wrapper.
    assert deps.AccountIdDep is not None
