from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memory_store import MemoryStore
from ...models.memory_store_update import MemoryStoreUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    *,
    body: MemoryStoreUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/memory-stores/{store_id}".format(
            store_id=quote(str(store_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | MemoryStore | None:
    if response.status_code == 200:
        response_200 = MemoryStore.from_dict(response.json())

        return response_200

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | MemoryStore]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryStoreUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryStore]:
    """Update Store

     Update a memory store's ``name``, ``description``, or ``metadata``.

    Rejects with ``MemoryStoreArchivedError`` if the store is archived —
    archived stores are read-only. Treat as partial update on the listed
    fields.

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryStoreUpdate): Request body for ``POST /v1/memory-stores/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryStore]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryStoreUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryStore | None:
    """Update Store

     Update a memory store's ``name``, ``description``, or ``metadata``.

    Rejects with ``MemoryStoreArchivedError`` if the store is archived —
    archived stores are read-only. Treat as partial update on the listed
    fields.

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryStoreUpdate): Request body for ``POST /v1/memory-stores/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MemoryStore
    """

    return sync_detailed(
        store_id=store_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryStoreUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryStore]:
    """Update Store

     Update a memory store's ``name``, ``description``, or ``metadata``.

    Rejects with ``MemoryStoreArchivedError`` if the store is archived —
    archived stores are read-only. Treat as partial update on the listed
    fields.

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryStoreUpdate): Request body for ``POST /v1/memory-stores/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryStore]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryStoreUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryStore | None:
    """Update Store

     Update a memory store's ``name``, ``description``, or ``metadata``.

    Rejects with ``MemoryStoreArchivedError`` if the store is archived —
    archived stores are read-only. Treat as partial update on the listed
    fields.

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryStoreUpdate): Request body for ``POST /v1/memory-stores/{id}``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MemoryStore
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
