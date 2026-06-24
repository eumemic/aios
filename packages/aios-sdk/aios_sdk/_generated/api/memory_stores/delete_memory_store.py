from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memory_store import MemoryStore
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/memory-stores/{store_id}".format(
            store_id=quote(str(store_id), safe=""),
        ),
    }

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
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryStore]:
    """Delete Store

     Soft-archive a memory store (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the store from default lists, and makes it
    read-only (same behavior as ``archive_memory_store``). The store, its
    memories, and the host mirror all persist. Bare DELETE is never silently
    destructive; for the irreversible hard-delete (cascade + host-mirror rm)
    use ``POST /v1/memory-stores/{store_id}/purge``.

    Args:
        store_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryStore]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
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
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryStore | None:
    """Delete Store

     Soft-archive a memory store (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the store from default lists, and makes it
    read-only (same behavior as ``archive_memory_store``). The store, its
    memories, and the host mirror all persist. Bare DELETE is never silently
    destructive; for the irreversible hard-delete (cascade + host-mirror rm)
    use ``POST /v1/memory-stores/{store_id}/purge``.

    Args:
        store_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MemoryStore
    """

    return sync_detailed(
        store_id=store_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MemoryStore]:
    """Delete Store

     Soft-archive a memory store (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the store from default lists, and makes it
    read-only (same behavior as ``archive_memory_store``). The store, its
    memories, and the host mirror all persist. Bare DELETE is never silently
    destructive; for the irreversible hard-delete (cascade + host-mirror rm)
    use ``POST /v1/memory-stores/{store_id}/purge``.

    Args:
        store_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MemoryStore]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MemoryStore | None:
    """Delete Store

     Soft-archive a memory store (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at``, hides the store from default lists, and makes it
    read-only (same behavior as ``archive_memory_store``). The store, its
    memories, and the host mirror all persist. Bare DELETE is never silently
    destructive; for the irreversible hard-delete (cascade + host-mirror rm)
    use ``POST /v1/memory-stores/{store_id}/purge``.

    Args:
        store_id (str):
        authorization (None | str | Unset):

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
            authorization=authorization,
        )
    ).parsed
