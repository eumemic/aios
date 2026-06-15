"""Tripwire: no attenuated / sub-top API principal exists today (#1130).

Part of #1122. The ``POST /v1/invocations`` invoke edge (#1128) lets an
external/operator caller supply an ``environment_id`` for the servicer session
it spawns. Issue #1130 ships an **ownership-only** gate on that supply: the env
must be owned by the authenticated account (``get_environment(.., account_id=)``,
the same check ``create_session`` / ``create_run`` already enforce). It
**defers** the per-field containment clamp (``networking`` / ``image`` /
``env``-keys ⊆ a baseline env) because there is **no sub-top API principal** to
clamp against — every account key is full-account-top, so ownership *is* the
bound and there is nothing below top to clamp to.

This test pins that deferral's precondition. It asserts, structurally, that the
API bearer-auth principal carries **no attenuation / surface / scope field**:

* the ``account_keys`` table (the bearer/account-key path) has **no**
  surface / scope / attenuation column — only ``key_id, account_id, hash,
  label, created_at, last_used_at, revoked_at`` (migration 0042);
* ``require_bearer_auth`` resolves a token to a full-account
  ``AccountAuthResult = (account_id, key_id, can_mint_children)`` — a plain
  ``account_id`` with no sub-top surface descriptor.

The ``connection_ids`` allowlist (#350) is a property of the **runtime-token**
path (``require_runtime_auth``), NOT the bearer/account-key path, so it is not
an attenuation of the API principal. Child accounts (``parent_account_id`` /
``can_mint_children``) are full-top *within their own tenant* — a distinct
``account_id``, not an attenuated subset of a parent's surface.

WHEN AN ATTENUATED / DELEGATED (SUB-top) API PRINCIPAL IS INTRODUCED, THE
ASSERTIONS BELOW FAIL — THAT IS THE SIGNAL TO IMPLEMENT THE DEFERRED PER-FIELD
ENV CONTAINMENT CLAMP (networking / image / env-keys ⊆ baseline) ON THE
api→session EDGE. The failing test is the deferral's re-entry point. The clamp
targets are the ``EnvironmentConfig`` axes (``image``, ``networking``, ``env``;
``models/environments.py``). See #1130. Sibling of #823 (the ``api_base``
second-authority axis — same tripwire shape).
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

# The exact column set ``account_keys`` carries (migration 0042). Anything
# beyond this on the bearer/account-key path is a candidate attenuation surface
# that re-opens the deferred clamp.
_EXPECTED_ACCOUNT_KEY_COLUMNS = frozenset(
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

# Substrings that would name a sub-top attenuation surface on the principal — a
# scope/surface/attenuation/grant restriction below full-account top.
_ATTENUATION_MARKERS = ("scope", "surface", "attenuat", "grant", "permission", "capabilit")


@pytest.fixture
async def pool(migrated_db_url: str, _reset_db_state: None) -> AsyncIterator[asyncpg.Pool[Any]]:
    pool = await create_pool(migrated_db_url, min_size=1, max_size=2)
    try:
        yield pool
    finally:
        await pool.close()


class TestNoSubTopApiPrincipal:
    async def test_account_keys_has_no_attenuation_column(self, pool: asyncpg.Pool[Any]) -> None:
        """The bearer/account-key path carries no scope/surface column.

        If this fails, an attenuation surface was added to ``account_keys`` —
        implement the deferred per-field env containment clamp on the
        api→session edge (networking / image / env-keys ⊆ baseline). See #1130.
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'account_keys'"
            )
        columns = {r["column_name"] for r in rows}
        # An exact-set match is the strongest pin: it fails both on a NEW column
        # (e.g. a scope/surface/attenuation surface — see _ATTENUATION_MARKERS for
        # the shapes that re-open the clamp) and on a dropped one. Any extra column
        # is a candidate sub-top attenuation surface that re-opens the deferred clamp.
        extra = columns - _EXPECTED_ACCOUNT_KEY_COLUMNS
        attenuation_like = sorted(
            c for c in extra if any(marker in c.lower() for marker in _ATTENUATION_MARKERS)
        )
        assert columns == _EXPECTED_ACCOUNT_KEY_COLUMNS, (
            "account_keys columns drifted from the full-account-top shape "
            f"{sorted(_EXPECTED_ACCOUNT_KEY_COLUMNS)} → {sorted(columns)} "
            f"(attenuation-like new columns: {attenuation_like}). "
            "If a scope/surface/attenuation column was added, a sub-top API "
            "principal now exists: implement the deferred per-field env "
            "containment clamp (networking/image/env-keys ⊆ baseline) on the "
            "api→session edge. See #1130."
        )

    def test_bearer_auth_result_is_full_account_with_no_surface(self) -> None:
        """``require_bearer_auth`` resolves to a full-account tuple, no sub-top surface.

        The bearer-auth principal is ``AccountAuthResult =
        (account_id, key_id, can_mint_children)`` — a plain ``account_id`` plus
        the minting bit, neither of which is a sub-top surface descriptor. If a
        surface/scope element is added to this tuple, a sub-top API principal now
        exists: implement the deferred per-field env containment clamp
        (networking/image/env-keys ⊆ baseline) on the api→session edge. See #1130.
        """
        # ``AccountAuthResult`` is a tuple alias; its element types pin the
        # principal's shape. account_id (str) + key_id (str) + can_mint (bool):
        # no surface/scope component.
        args = typing.get_args(deps.AccountAuthResult)
        assert args == (str, str, bool), (
            "AccountAuthResult drifted from (account_id: str, key_id: str, "
            f"can_mint_children: bool) → {args}. If a surface/scope element was "
            "added, a sub-top API principal now exists: implement the deferred "
            "per-field env containment clamp (networking/image/env-keys ⊆ "
            "baseline) on the api→session edge. See #1130."
        )

    def test_get_account_id_returns_bare_account_id(self) -> None:
        """``get_account_id`` (the dep the invoke endpoint uses) yields a bare account_id.

        ``AccountIdDep`` (the dependency ``POST /v1/invocations`` consumes for
        its ownership gate) destructures the auth tuple to a single
        ``account_id: str`` — the gate scopes ``get_environment`` by that
        full-account id, never a sub-top surface. Its return annotation is the
        structural pin.
        """
        hints = typing.get_type_hints(deps.get_account_id)
        assert hints.get("return") is str, (
            "get_account_id no longer returns a bare account_id (str): "
            f"{hints.get('return')}. The invoke ownership gate scopes on this "
            "value; if it became a sub-top surface, implement the deferred "
            "per-field env containment clamp. See #1130."
        )
