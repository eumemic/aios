from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memory import Memory
from ...models.memory_create import MemoryCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    *,
    body: MemoryCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/memory-stores/{store_id}/memories".format(
            store_id=quote(str(store_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Memory | None:
    if response.status_code == 201:
        response_201 = Memory.from_dict(response.json())

        return response_201

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[HTTPValidationError | Memory]:
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
    body: MemoryCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Memory]:
    """Create Memory

     Create a memory at the given path within a store.

    The content is also mirrored to the store's host directory so that
    sandboxed sessions can read it through the filesystem. Creates an
    initial version (every memory mutation creates a version).

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryCreate): Request body for ``POST /v1/memory-stores/{store_id}/memories``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Memory]
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
    body: MemoryCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Memory | None:
    """Create Memory

     Create a memory at the given path within a store.

    The content is also mirrored to the store's host directory so that
    sandboxed sessions can read it through the filesystem. Creates an
    initial version (every memory mutation creates a version).

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryCreate): Request body for ``POST /v1/memory-stores/{store_id}/memories``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Memory
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
    body: MemoryCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Memory]:
    """Create Memory

     Create a memory at the given path within a store.

    The content is also mirrored to the store's host directory so that
    sandboxed sessions can read it through the filesystem. Creates an
    initial version (every memory mutation creates a version).

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryCreate): Request body for ``POST /v1/memory-stores/{store_id}/memories``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Memory]
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
    body: MemoryCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Memory | None:
    """Create Memory

     Create a memory at the given path within a store.

    The content is also mirrored to the store's host directory so that
    sandboxed sessions can read it through the filesystem. Creates an
    initial version (every memory mutation creates a version).

    Args:
        store_id (str):
        authorization (None | str | Unset):
        body (MemoryCreate): Request body for ``POST /v1/memory-stores/{store_id}/memories``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Memory
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
