"""Business logic for the model_providers resource.

Thin wrapper over :mod:`aios.db.queries`, plus :func:`resolve_provider_auth_or_conflict`
— the single entry point the harness calls on every model call. It fuses
nearest-ancestor credential resolution with the conflict guard: resolving
auth without also running the check (or vice versa) is exactly the security
hole this guard exists to close, so the two inner steps
(``_resolve_provider_auth`` / ``_check_provider_auth_conflict``) are private
and only reachable together.

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

import json
from functools import lru_cache
from types import EllipsisType
from typing import Any

import asyncpg
import litellm

from aios.config import get_settings
from aios.crypto.vault import CryptoBox
from aios.db import queries
from aios.errors import ValidationError
from aios.models.attenuation import api_base_of
from aios.models.model_providers import (
    CREDENTIAL_KWARGS,
    ENDPOINT_KWARGS,
    REASONING_KWARGS,
    ModelProvider,
    ProviderAuth,
    provider_auth_conflict,
    validate_row_content,
)
from aios.services.inference_credential_telemetry import observe_env_fallback

# Surfaced as a session's stop_reason.message (session-visible) and, on the
# workflow call_llm path, as a journaled {"error": ...} value — both are read
# by principals who must not learn ancestor account ids or tree depth. Static
# and generic, matching every other stop_message constant in harness/loop.py
# (_SPEND_CAP_STOP_REASON_MESSAGE, _REFUSAL_STOP_REASON_MESSAGE): no dynamic
# data embedded.
PROVIDER_NOT_CONFIGURED_MESSAGE = (
    "No account-scoped credentials are configured for this model provider."
)

PROVIDER_AUTH_CONFLICT_MESSAGE = (
    "This session's effective model-provider key is owned by an account above "
    "this one; configure a model_providers row for this account."
)


def inline_credentials_authoritative() -> bool:
    """Whether an agent's inline credential kwargs may stand as their own source.

    Under ``legacy_env``/``observe_legacy_env`` the issue #1816 all-or-nothing
    rule applies: an agent ``litellm_extra`` carrying any ``CREDENTIAL_KWARGS``
    key suppresses the matched row's credentials wholesale (today's oai-proxy
    fleet). Under ``account_only`` (#1969) account rows are authoritative and
    inline credential/endpoint kwargs are agent metadata that is stripped.
    """
    return get_settings().inference_credential_policy != "account_only"


def _owned(value: Any) -> Any:
    """Copy dict/list containers recursively; leave leaf objects shared.

    ``deepcopy`` would choke on live objects an agent may pass through
    ``litellm_extra`` (e.g. a ``client``), and copying the containers is all
    ownership needs — in-place mutation only ever targets dicts.
    """
    if isinstance(value, dict):
        return {k: _owned(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_owned(v) for v in value]
    return value


def merge_provider_config(
    *,
    harness_kwargs: dict[str, Any],
    resolved: ProviderAuth | None,
    agent_extra: dict[str, Any] | None,
    inline_credentials: bool | None = None,
) -> dict[str, Any]:
    """The ONE merge of call kwargs, applied only at the litellm call boundary.

    Precedence: harness kwargs < account ``litellm_defaults`` < account
    credentials < agent ``litellm_extra``. Whole-key shallow merge with two
    exceptions: an agent reasoning-family key drops the whole family from the
    account defaults, and ``extra_body`` unions per key (agent wins). Returns
    owned containers (deep copies) — callers mutate ``extra_body`` in place.

    Credentials never enter ``LlmRequest.params``: this runs downstream of it.
    ``inline_credentials`` defaults to :func:`inline_credentials_authoritative`.
    """
    agent = _owned(dict(agent_extra or {}))
    if "model" in agent:
        raise ValueError("model is forbidden in litellm_extra")
    if inline_credentials is None:
        inline_credentials = inline_credentials_authoritative()
    defaults: dict[str, Any] = _owned(resolved.litellm_defaults if resolved else {})
    if REASONING_KWARGS & agent.keys():
        for key in REASONING_KWARGS:
            defaults.pop(key, None)
    credentials: dict[str, Any] = {}
    if resolved is not None:
        credentials = _owned(resolved.credentials)
        if resolved.api_key:
            credentials["api_key"] = resolved.api_key
        if api_base_of(credentials) is None and resolved.api_base is not None:
            credentials["api_base"] = resolved.api_base
        if not inline_credentials:
            # Account row is authoritative: inline credential/endpoint kwargs
            # are agent metadata, never combined with the row's secret.
            for key in CREDENTIAL_KWARGS | ENDPOINT_KWARGS:
                agent.pop(key, None)
        elif CREDENTIAL_KWARGS & agent.keys():
            # All-or-nothing: the agent's endpoint+credential pair stands alone.
            credentials = {}
        if ENDPOINT_KWARGS & agent.keys():
            # One endpoint source only — LiteLLM's api_base-over-base_url
            # precedence would otherwise silently invert "agent wins".
            for key in ENDPOINT_KWARGS:
                credentials.pop(key, None)
    merged: dict[str, Any] = _owned(harness_kwargs)
    merged.update(defaults)
    merged.update(credentials)
    lower_extra_body = merged.get("extra_body")
    agent_extra_body = agent.pop("extra_body", None)
    merged.update(agent)
    if isinstance(lower_extra_body, dict) and isinstance(agent_extra_body, dict):
        merged["extra_body"] = {**lower_extra_body, **agent_extra_body}
    elif agent_extra_body is not None:
        merged["extra_body"] = agent_extra_body
    return merged


def _encrypt_credentials(
    credentials: dict[str, Any], crypto_box: CryptoBox, *, account_id: str
) -> Any:
    return crypto_box.derive_account_subkey(account_id).encrypt(
        json.dumps(credentials, separators=(",", ":"), sort_keys=True)
    )


async def create_model_provider(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    provider: str,
    api_key: str | None,
    api_base: str | None,
    credentials: dict[str, Any] | None = None,
    litellm_defaults: dict[str, Any] | None = None,
) -> ModelProvider:
    payload = dict(credentials or {})
    if api_key is not None:
        payload["api_key"] = api_key
    if api_base is not None:
        payload["api_base"] = api_base
    async with pool.acquire() as conn:
        return await queries.insert_model_provider(
            conn,
            account_id=account_id,
            provider=provider,
            api_base=api_base,
            blob=_encrypt_credentials(payload, crypto_box, account_id=account_id),
            litellm_defaults=litellm_defaults,
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


def _decode_credentials(plaintext: str) -> dict[str, Any]:
    """Parse a decrypted blob: JSON credentials, or a 0140-era bare API key."""
    try:
        credentials = json.loads(plaintext)
    except json.JSONDecodeError:
        return {"api_key": plaintext}
    if not isinstance(credentials, dict):
        return {"api_key": plaintext}
    return credentials


def _merge_credentials(
    existing: dict[str, Any],
    *,
    credentials: dict[str, Any] | None,
    api_key: str | None,
    api_base: str | EllipsisType | None,
) -> dict[str, Any]:
    """Apply a partial credentials update onto the stored (decrypted) set.

    PATCH semantics, matching how ``api_key``/``api_base`` already behave on
    this endpoint (omitted → keep): a key present in ``credentials`` replaces
    the stored value, an explicit ``null`` removes it, and an omitted key is
    kept. Credentials are write-only (never echoed), so replace-the-whole-blob
    semantics would force a caller to re-send a secret it cannot read back
    merely to add a header — and silently drop the key when it does not.
    """
    merged = dict(existing)
    for key, value in (credentials or {}).items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    if api_key is not None:
        merged["api_key"] = api_key
    if api_base is not ...:
        # One endpoint source: the explicit api_base replaces (or, on null,
        # clears) every alias the blob may carry, so the stored blob, the
        # plaintext column, and what resolution sees cannot disagree.
        merged.pop("api_base", None)
        merged.pop("base_url", None)
        if api_base is not None:
            merged["api_base"] = api_base
    return merged


async def update_model_provider(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    model_provider_id: str,
    *,
    account_id: str,
    api_key: str | None,
    api_base: str | EllipsisType | None = ...,
    credentials: dict[str, Any] | None = None,
    litellm_defaults: dict[str, Any] | EllipsisType | None = ...,
    expected_version: int | None = None,
) -> ModelProvider:
    """Rotate the key, patch credentials, and/or edit ``api_base``/defaults.

    ``api_key=None`` means "keep the existing key" (rotation is opt-in via an
    explicit value). ``api_base`` follows the ``model_fields_set`` convention
    at the router — pass ``...`` (the default) to keep, an explicit value
    (including ``None``) to set/clear. ``credentials`` is a partial update
    merged onto the stored set (see :func:`_merge_credentials`).

    Any credential-touching change is a read-modify-write of the encrypted
    blob under a row lock, so the stored key survives a partial update and two
    concurrent partial updates serialize rather than merge onto the same stale
    blob. The merged result is validated before it is written: a set that
    would leave an endpoint without ``api_key`` (which
    :func:`_resolve_provider_auth` refuses at call time) is rejected with a
    422 and nothing is written.

    ``expected_version``, when given, is checked (409 on mismatch) — see
    :func:`aios.db.queries.update_model_provider`.
    """
    touches_blob = credentials is not None or api_key is not None or api_base is not ...
    async with pool.acquire() as conn:
        if not touches_blob:
            return await queries.update_model_provider(
                conn,
                model_provider_id,
                account_id=account_id,
                blob=None,
                api_base=...,
                litellm_defaults=litellm_defaults,
                expected_version=expected_version,
            )
        async with conn.transaction():
            locked = await queries.lock_model_provider(
                conn, model_provider_id, account_id=account_id
            )
            # Owned-row edit: the blob is under THIS account's subkey. An
            # archived row was rejected by the lock (ciphertext is zeroed).
            existing = _decode_credentials(
                crypto_box.derive_account_subkey(account_id).decrypt(locked.blob)
            )
            merged = _merge_credentials(
                existing, credentials=credentials, api_key=api_key, api_base=api_base
            )
            retained_base = locked.api_base if api_base is ... else api_base
            effective = dict(merged)
            if api_base_of(effective) is None and retained_base is not None:
                effective["api_base"] = retained_base
            try:
                validate_row_content(effective, {})
            except ValueError as exc:
                raise ValidationError(
                    f"model provider config {model_provider_id}: {exc}",
                    detail={"id": model_provider_id},
                ) from exc
            return await queries.update_model_provider(
                conn,
                model_provider_id,
                account_id=account_id,
                blob=_encrypt_credentials(merged, crypto_box, account_id=account_id),
                api_base=api_base,
                litellm_defaults=litellm_defaults,
                expected_version=expected_version,
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


@lru_cache(maxsize=1024)
def _derive_provider(model: str, custom_llm_provider: str | None) -> str | None:
    """``litellm.get_llm_provider``'s provider sniff, cached.

    Same rationale as ``harness.completion.model_descriptor``: a pure
    function of its (hashable) args, called on every model call, over a
    distinct-``(model, custom_llm_provider)`` cardinality that's low in
    practice (agents typically reuse one or two models). ``None`` means
    unresolvable (``get_llm_provider`` raised — e.g. a ``workflow:`` binding
    or a garbage string) and is cached too, so a repeated bad string doesn't
    re-raise on every call.

    Bounded (``maxsize``) rather than unbounded ``@cache`` because on the
    workflow ``call_llm`` path the ``model`` arg is script-computed, so an
    adversarial script could otherwise grow this cache monotonically. Both
    args MUST be hashable — the caller guarantees ``str | None`` (an
    unhashable ``custom_llm_provider`` would raise a ``TypeError`` in the
    lru_cache key machinery, BEFORE this body's try/except, so the caller
    normalizes it, not us).
    """
    try:
        _, provider, _, _ = litellm.get_llm_provider(model, custom_llm_provider=custom_llm_provider)
    except Exception:
        return None
    return str(provider)


async def _resolve_provider_auth(
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
    no-row lookup at this boundary: :func:`_check_provider_auth_conflict`
    treats every ``None`` identically (see its docstring for why a bare
    "nothing to check" pass here would reopen the guard's central bypass).
    """
    # Only a non-empty *string* custom_llm_provider is an override; anything
    # else (None, a non-str the script smuggled into params — values are
    # unvalidated ``dict[str, Any]``) is normalized to None here so the
    # hashable-args contract of the lru_cache'd ``_derive_provider`` holds. A
    # truthy non-str would otherwise raise a ``TypeError`` in the cache-key
    # machinery, escaping this whole path as a raise (on the run_llm lane that
    # is an unbounded re-dispatch loop; see invoke_call_llm's guard-3 wrapping).
    raw_custom = (litellm_extra or {}).get("custom_llm_provider")
    custom_llm_provider = raw_custom if isinstance(raw_custom, str) and raw_custom else None
    provider = _derive_provider(model, custom_llm_provider)
    if provider is None:
        return None
    async with pool.acquire() as conn:
        resolved = await queries.resolve_model_provider(
            conn, account_id=account_id, provider=provider
        )
    if resolved is None:
        if get_settings().inference_credential_policy == "observe_legacy_env":
            observe_env_fallback(account_id=account_id, provider=provider)
        return None
    settings = get_settings()
    if resolved.owner_account_id != account_id and (
        settings.inference_credential_policy == "account_only"
        or settings.tenancy_posture == "external_byok"
    ):
        async with pool.acquire() as conn:
            owner = await queries.get_account(conn, resolved.owner_account_id)
        if owner is not None and owner.parent_account_id is None:
            # Defense in depth for legacy/manual rows: platform-root credentials
            # are never inherited by a descendant under a non-legacy policy.
            return None
    subkey = crypto_box.derive_account_subkey(resolved.owner_account_id)
    # Migration compatibility: rows written by 0140 hold a bare API key.
    credentials = _decode_credentials(subkey.decrypt(resolved.blob))
    if (
        (api_base_of(credentials) is not None or resolved.api_base is not None)
        and CREDENTIAL_KWARGS & credentials.keys()
        and not credentials.get("api_key")
    ):
        # Merge-time backstop for pre-validator rows: a credentialed row that
        # yields an endpoint without a key must never fall back to env keys.
        raise ValueError("model-provider credentials endpoint requires api_key")
    return ProviderAuth(
        api_key=str(credentials.get("api_key", "")),
        api_base=api_base_of(credentials) or resolved.api_base,
        owner_account_id=resolved.owner_account_id,
        credentials=credentials,
        litellm_defaults=resolved.litellm_defaults or {},
    )


async def _check_provider_auth_conflict(
    pool: asyncpg.Pool[Any],
    *,
    account_id: str,
    litellm_extra: dict[str, Any] | None,
    resolved: ProviderAuth | None,
) -> str | None:
    """The static conflict message if this call is inadmissible, else ``None``.

    Root lookup is skipped entirely unless ``resolved is None`` AND the call
    actually redirects — the common cases (no redirect at all, a redirect
    with no api_base/base_url key, or a resolved row) never touch the
    ``accounts`` table beyond what resolution already did.
    """
    account_is_root = True
    if resolved is None and api_base_of(litellm_extra) is not None:
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


async def resolve_provider_auth_or_conflict(
    pool: asyncpg.Pool[Any],
    crypto_box: CryptoBox,
    *,
    account_id: str,
    model: str,
    litellm_extra: dict[str, Any] | None,
) -> tuple[ProviderAuth | None, str | None]:
    """Resolve provider auth AND run the conflict guard, in one call.

    The single production entry point for the model-call sites (harness/loop.py,
    workflows/run_llm.py). Resolving auth without also checking it for conflict
    (or checking a conflict against auth resolved a different way) is exactly
    the cross-tenant key-exfiltration hole this feature exists to close, so the
    two steps are fused here rather than left as two functions a future call
    site could invoke out of order, partially, or with mismatched arguments —
    ``_resolve_provider_auth``/``_check_provider_auth_conflict`` are private.

    Returns ``(auth, conflict_message)``. ``conflict_message`` is ``None`` when
    the call is admissible — ``auth`` may still be ``None`` in that case (no
    row anywhere in the chain; LiteLLM falls back to the worker's env key).
    """
    auth = await _resolve_provider_auth(
        pool, crypto_box, account_id=account_id, model=model, litellm_extra=litellm_extra
    )
    # Only a credentialed row constrains endpoint identity; a defaults-only row
    # leaves the call on the env credential path, so it is checked as no-row.
    conflict = await _check_provider_auth_conflict(
        pool,
        account_id=account_id,
        litellm_extra=litellm_extra,
        resolved=auth if auth is not None and auth.has_credentials else None,
    )
    return auth, conflict
