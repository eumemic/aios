from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_memory_version import ListResponseMemoryVersion
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    *,
    memory_id: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_memory_id: None | str | Unset
    if isinstance(memory_id, Unset):
        json_memory_id = UNSET
    else:
        json_memory_id = memory_id
    params["memory_id"] = json_memory_id

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/memory-stores/{store_id}/memory-versions".format(
            store_id=quote(str(store_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseMemoryVersion | None:
    if response.status_code == 200:
        response_200 = ListResponseMemoryVersion.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseMemoryVersion]:
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
    memory_id: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseMemoryVersion]:
    """List Versions

     List memory versions in a store, newest first.

    Optional ``memory_id`` filters to a single memory's version history.
    Without the filter, returns versions across all memories in the store
    (useful for audit). No cursor pagination; bumps default limit to 100.

    Args:
        store_id (str):
        memory_id (None | str | Unset):
        limit (int | Unset):  Default: 100.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseMemoryVersion]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
        limit=limit,
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
    memory_id: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseMemoryVersion | None:
    """List Versions

     List memory versions in a store, newest first.

    Optional ``memory_id`` filters to a single memory's version history.
    Without the filter, returns versions across all memories in the store
    (useful for audit). No cursor pagination; bumps default limit to 100.

    Args:
        store_id (str):
        memory_id (None | str | Unset):
        limit (int | Unset):  Default: 100.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseMemoryVersion
    """

    return sync_detailed(
        store_id=store_id,
        client=client,
        memory_id=memory_id,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    memory_id: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseMemoryVersion]:
    """List Versions

     List memory versions in a store, newest first.

    Optional ``memory_id`` filters to a single memory's version history.
    Without the filter, returns versions across all memories in the store
    (useful for audit). No cursor pagination; bumps default limit to 100.

    Args:
        store_id (str):
        memory_id (None | str | Unset):
        limit (int | Unset):  Default: 100.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseMemoryVersion]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    *,
    client: AuthenticatedClient | Client,
    memory_id: None | str | Unset = UNSET,
    limit: int | Unset = 100,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseMemoryVersion | None:
    """List Versions

     List memory versions in a store, newest first.

    Optional ``memory_id`` filters to a single memory's version history.
    Without the filter, returns versions across all memories in the store
    (useful for audit). No cursor pagination; bumps default limit to 100.

    Args:
        store_id (str):
        memory_id (None | str | Unset):
        limit (int | Unset):  Default: 100.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseMemoryVersion
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            client=client,
            memory_id=memory_id,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
