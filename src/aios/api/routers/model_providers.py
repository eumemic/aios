"""Model-provider config endpoints — per-account API keys + proxy base URLs.

One active row per ``(account, provider)``. Resolved nearest-ancestor-wins
up the account tree at model-call time; a row's ``api_key`` and ``api_base``
always resolve together from the same account (row-atomicity) — see
``aios.models.model_providers`` for why this matters. ``api_key`` is
write-only: creation and rotation accept it, but no read ever returns it —
only the boolean ``api_key_set`` echo. Keys are not validated at write time;
a wrong key surfaces as a terminal authentication error on the next model
call that resolves to it.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from aios.api.deps import AccountIdDep, CryptoBoxDep, PoolDep
from aios.models.common import ListResponse
from aios.models.model_providers import ModelProvider, ModelProviderCreate, ModelProviderUpdate
from aios.models.pagination import PageLimit, page_cursor, resolve_page_limit
from aios.services import model_providers as service

router = APIRouter(prefix="/v1/model-providers", tags=["model-providers"])


@router.post("", operation_id="create_model_provider", status_code=status.HTTP_201_CREATED)
async def create(
    body: ModelProviderCreate, pool: PoolDep, crypto_box: CryptoBoxDep, account_id: AccountIdDep
) -> ModelProvider:
    """Create a model-provider config for this account.

    ``provider`` is a LiteLLM provider name (``anthropic``, ``openai``,
    ``openrouter``, …) — one active config per ``(account, provider)``; a
    second create for the same provider on this account 409s (archive the
    existing one first to replace it). ``api_key`` is encrypted at rest and
    never returned; ``api_base`` is stored plaintext for proxy/self-hosted
    routing.
    """
    return await service.create_model_provider(
        pool,
        crypto_box,
        account_id=account_id,
        provider=body.provider,
        api_key=body.api_key.get_secret_value(),
        api_base=body.api_base,
    )


@router.get("", operation_id="list_model_providers")
async def list_(
    pool: PoolDep,
    account_id: AccountIdDep,
    cursor: str | None = None,
    provider: str | None = None,
    limit: PageLimit = None,
) -> ListResponse[ModelProvider]:
    """List this account's own model-provider configs, newest first, excluding archived.

    Own rows only — does not reflect whether an ancestor account's config
    would win resolution for a provider this account has no row for.
    First page: optional ``provider`` filter + ``?limit=``. Subsequent
    pages: ``?cursor=<next_cursor>`` (carries the filter).
    """
    st = page_cursor(cursor, {"provider": provider, "limit": limit})
    after = str(st.cursor) if st is not None else None
    page_limit = resolve_page_limit(st, limit)
    if st is not None:
        provider = st.filters.get("provider")
    items = await service.list_model_providers(
        pool, provider=provider, limit=page_limit + 1, after=after, account_id=account_id
    )
    return ListResponse[ModelProvider].paginate(
        items, page_limit, cursor=lambda x: x.id, filters={"provider": provider}
    )


@router.get("/{model_provider_id}", operation_id="get_model_provider")
async def get(model_provider_id: str, pool: PoolDep, account_id: AccountIdDep) -> ModelProvider:
    """Fetch one model-provider config by id. ``api_key`` is never returned."""
    return await service.get_model_provider(pool, model_provider_id, account_id=account_id)


@router.put("/{model_provider_id}", operation_id="update_model_provider")
async def update(
    model_provider_id: str,
    body: ModelProviderUpdate,
    pool: PoolDep,
    crypto_box: CryptoBoxDep,
    account_id: AccountIdDep,
) -> ModelProvider:
    """Rotate the key and/or edit ``api_base``.

    ``api_key`` omitted → keep the existing key (there is no way to clear it
    back to unset — archive and recreate instead). ``api_base`` omitted →
    keep; explicit ``null`` → clear.
    """
    api_base = body.api_base if "api_base" in body.model_fields_set else ...
    return await service.update_model_provider(
        pool,
        crypto_box,
        model_provider_id,
        account_id=account_id,
        api_key=body.api_key.get_secret_value() if body.api_key is not None else None,
        api_base=api_base,
    )


@router.delete(
    "/{model_provider_id}",
    operation_id="archive_model_provider",
    status_code=status.HTTP_204_NO_CONTENT,
    openapi_extra={"x-codegen": {"mcp": {"destructiveHint": True}}},
)
async def archive(model_provider_id: str, pool: PoolDep, account_id: AccountIdDep) -> None:
    """Archive a config and **zero its encrypted key**.

    Hides it from default lists and from resolution; a future DB dump
    cannot leak the scrubbed key. Idempotent — archiving an already-archived
    config is a no-op, not a 404.
    """
    await service.archive_model_provider(
        pool, model_provider_id, account_id=account_id, idempotent=True
    )
