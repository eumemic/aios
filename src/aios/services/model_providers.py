"""Business logic for the model_providers resource.

Thin wrapper over :mod:`aios.db.queries`, plus the two functions the harness
calls on every model call: :func:`resolve_provider_auth` (nearest-ancestor
credential resolution) and :func:`check_provider_auth_conflict` (the guard).

**The guard is deliberately call-time-only, never agent-write-time.** A
``model_providers`` row can be created, rotated, or archived by an ancestor
account at any point AFTER an agent's ``litellm_extra.api_base`` was set (and
would have passed a write-time check at that moment) — mirroring why
spend-gate admission is re-evaluated live per-step rather than cached at
agent-creation time (``harness/loop.py``'s pre-model-call spend check). A
write-time-only check would be unsound: it cannot detect a conflict an
ancestor's later action introduces. No additional write-time check should be
added alongside this one without an explicit follow-up decision.
"""

from __future__ import annotations

from types import EllipsisType
from typing import Any

import asyncpg
import litellm

from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.models.model_providers import ModelProvider, ProviderAuth, provider_auth_conflict

# Surfaced as a session's stop_reason.message (session-visible) and, on the
# workflow call_llm path, as a journaled {"error": ...} value — both are read
# by principals who must not learn ancestor account ids or tree depth. Static
# and generic, matching every other stop_message constant in harness/loop.py
# (_SPEND_CAP_STOP_REASON_MESSAGE, _REFUSAL_STOP_REASON_MESSAGE): no dynamic
# data embedded.
PROVIDER_AUTH_CONFLICT_MESSAGE = (
    "This session's effective model-provider key is owned by an account above "
    "this one; supply your own api_key in litellm_extra or configure a "
    "model_providers row for this account."
)


def _encrypt_api_key(api_key: str, crypto_box: CryptoBox, *, account_id: str) -> Any:
    return crypto_box.derive_account_subkey(account_id).encrypt(api_key)


async def create_model_provider(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    provider: str,
    api_key: str,
    api_base: str | None,
) -> ModelProvider:
    async with pool.acquire() as conn:
        return await queries.insert_model_provider(
            conn,
            account_id=account_id,
            provider=provider,
            api_base=api_base,
            blob=_encrypt_api_key(api_key, crypto_box, account_id=account_id),
        )


async def get_model_provider(
    pool: asyncpg.Pool[Any], model_provider_id: str, *, account_id: str
) -> ModelProvider:
    async with pool.acquire() as conn:
        return await queries.get_model_provider(conn, model_provider_id, account_id=account_id)


async def list_model_providers(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    provider: str | None = None,
    limit: int = 50,
    after: str | None = None,
) -> list[ModelProvider]:
    async with pool.acquire() as conn:
        return await queries.list_model_providers(
            conn, account_id=account_id, provider=provider, limit=limit, after=after
        )


async def update_model_provider(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    model_provider_id: str,
    *,
    account_id: str,
    api_key: str | None,
    api_base: str | None | EllipsisType = ...,
) -> ModelProvider:
    """Rotate the key and/or edit ``api_base``.

    ``api_key=None`` means "keep the existing key" (rotation is opt-in via an
    explicit value). ``api_base`` follows the ``model_fields_set`` convention
    at the router — pass ``...`` (the default) to keep, an explicit value
    (including ``None``) to set/clear.
    """
    blob = _encrypt_api_key(api_key, crypto_box, account_id=account_id) if api_key else None
    async with pool.acquire() as conn:
        return await queries.update_model_provider(
            conn, model_provider_id, account_id=account_id, blob=blob, api_base=api_base
        )


async def archive_model_provider(
    pool: asyncpg.Pool[Any],
    model_provider_id: str,
    *,
    account_id: str,
    idempotent: bool = False,
) -> ModelProvider:
    async with pool.acquire() as conn:
        return await queries.archive_model_provider(
            conn, model_provider_id, account_id=account_id, idempotent=idempotent
        )


async def resolve_provider_auth(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    model: str,
    litellm_extra: dict[str, Any] | None,
) -> ProviderAuth | None:
    """Resolve the nearest-ancestor ``model_providers`` row for ``model``'s provider.

    Provider identity is derived via ``litellm.get_llm_provider``, passing
    through ``litellm_extra["custom_llm_provider"]`` when present — LiteLLM's
    own dispatch honors that override when selecting a provider (and, absent
    an explicit key, which env var it falls back to), so the row this
    function looks up must match what LiteLLM will actually call, not just
    the bare model string.

    An unresolvable ``model`` (``get_llm_provider`` raises — e.g. a
    ``workflow:`` binding or a garbage string) returns ``None`` without a DB
    round trip. This is intentionally NOT distinguished from a genuine
    no-row lookup at this boundary: :func:`check_provider_auth_conflict`
    treats every ``None`` identically (see its docstring for why a bare
    "nothing to check" pass here would reopen the guard's central bypass).
    """
    custom_llm_provider = (litellm_extra or {}).get("custom_llm_provider")
    try:
        provider = litellm.get_llm_provider(
            model, custom_llm_provider=custom_llm_provider if custom_llm_provider else None
        )[1]
    except Exception:
        return None
    async with pool.acquire() as conn:
        resolved = await queries.resolve_model_provider(
            conn, account_id=account_id, provider=provider
        )
    if resolved is None:
        return None
    subkey = crypto_box.derive_account_subkey(resolved.owner_account_id)
    return ProviderAuth(
        api_key=subkey.decrypt(resolved.blob),
        api_base=resolved.api_base,
        owner_account_id=resolved.owner_account_id,
    )


async def check_provider_auth_conflict(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    litellm_extra: dict[str, Any] | None,
    resolved: ProviderAuth | None,
) -> str | None:
    """The static conflict message if this call is inadmissible, else ``None``.

    Root lookup is skipped entirely unless ``resolved is None`` AND the call
    redirects — the common cases (no redirect, or a resolved row) never touch
    the ``accounts`` table beyond what resolution already did.
    """
    account_is_root = True
    if resolved is None and litellm_extra:
        # Only reachable when there's a redirect to evaluate (provider_auth_conflict
        # short-circuits on `redirect is None` before this matters) — but recompute
        # honestly rather than assume, since a future caller could pass this
        # differently. Cheap: one row lookup, only on the no-usable-row arm.
        async with pool.acquire() as conn:
            account = await queries.get_account(conn, account_id)
        assert account is not None, f"account {account_id} vanished mid-request"
        account_is_root = account.parent_account_id is None
    if provider_auth_conflict(
        litellm_extra=litellm_extra,
        resolved=resolved,
        account_id=account_id,
        account_is_root=account_is_root,
    ):
        return PROVIDER_AUTH_CONFLICT_MESSAGE
    return None
