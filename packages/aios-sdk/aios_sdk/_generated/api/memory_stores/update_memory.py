from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.memory import Memory
from ...models.memory_update import MemoryUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    memory_id: str,
    *,
    body: MemoryUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/memory-stores/{store_id}/memories/{memory_id}".format(
            store_id=quote(str(store_id), safe=""),
            memory_id=quote(str(memory_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Memory | None:
    if response.status_code == 200:
        response_200 = Memory.from_dict(response.json())

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
) -> Response[HTTPValidationError | Memory]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Memory]:
    """Update Memory

     Update a memory's content and/or path. Creates a new version.

    Optionally honors a ``precondition.content_sha256`` for optimistic
    concurrency — if the current content's SHA-256 differs, the update
    fails with a precondition error rather than overwriting. The host
    mirror is updated to match: renames delete the old path and write the
    new one.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):
        body (MemoryUpdate): Request body for ``POST /v1/memory-stores/{store_id}/memories/{id}``.

            Either ``content`` or ``path`` (or both) must be provided. Precondition
            only gates the content half — renames are unconditional, matching the
            Anthropic semantics confirmed by live probe.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Memory]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Memory | None:
    """Update Memory

     Update a memory's content and/or path. Creates a new version.

    Optionally honors a ``precondition.content_sha256`` for optimistic
    concurrency — if the current content's SHA-256 differs, the update
    fails with a precondition error rather than overwriting. The host
    mirror is updated to match: renames delete the old path and write the
    new one.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):
        body (MemoryUpdate): Request body for ``POST /v1/memory-stores/{store_id}/memories/{id}``.

            Either ``content`` or ``path`` (or both) must be provided. Precondition
            only gates the content half — renames are unconditional, matching the
            Anthropic semantics confirmed by live probe.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Memory
    """

    return sync_detailed(
        store_id=store_id,
        memory_id=memory_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | Memory]:
    """Update Memory

     Update a memory's content and/or path. Creates a new version.

    Optionally honors a ``precondition.content_sha256`` for optimistic
    concurrency — if the current content's SHA-256 differs, the update
    fails with a precondition error rather than overwriting. The host
    mirror is updated to match: renames delete the old path and write the
    new one.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):
        body (MemoryUpdate): Request body for ``POST /v1/memory-stores/{store_id}/memories/{id}``.

            Either ``content`` or ``path`` (or both) must be provided. Precondition
            only gates the content half — renames are unconditional, matching the
            Anthropic semantics confirmed by live probe.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Memory]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MemoryUpdate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | Memory | None:
    """Update Memory

     Update a memory's content and/or path. Creates a new version.

    Optionally honors a ``precondition.content_sha256`` for optimistic
    concurrency — if the current content's SHA-256 differs, the update
    fails with a precondition error rather than overwriting. The host
    mirror is updated to match: renames delete the old path and write the
    new one.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):
        body (MemoryUpdate): Request body for ``POST /v1/memory-stores/{store_id}/memories/{id}``.

            Either ``content`` or ``path`` (or both) must be provided. Precondition
            only gates the content half — renames are unconditional, matching the
            Anthropic semantics confirmed by live probe.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Memory
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            memory_id=memory_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
