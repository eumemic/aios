from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    store_id: str,
    memory_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/memory-stores/{store_id}/memories/{memory_id}".format(
            store_id=quote(str(store_id), safe=""),
            memory_id=quote(str(memory_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | HTTPValidationError]:
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
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Delete Memory

     Soft-delete a memory and remove it from the host mirror.

    Sets ``deleted_at`` on the memory and appends a ``deleted`` tombstone
    version. The row and version history persist; live reads filter on
    ``deleted_at IS NULL`` so the memory becomes invisible. Returns 204.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
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
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Delete Memory

     Soft-delete a memory and remove it from the host mirror.

    Sets ``deleted_at`` on the memory and appends a ``deleted`` tombstone
    version. The row and version history persist; live reads filter on
    ``deleted_at IS NULL`` so the memory becomes invisible. Returns 204.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        store_id=store_id,
        memory_id=memory_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Delete Memory

     Soft-delete a memory and remove it from the host mirror.

    Sets ``deleted_at`` on the memory and appends a ``deleted`` tombstone
    version. The row and version history persist; live reads filter on
    ``deleted_at IS NULL`` so the memory becomes invisible. Returns 204.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        store_id=store_id,
        memory_id=memory_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    store_id: str,
    memory_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Delete Memory

     Soft-delete a memory and remove it from the host mirror.

    Sets ``deleted_at`` on the memory and appends a ``deleted`` tombstone
    version. The row and version history persist; live reads filter on
    ``deleted_at IS NULL`` so the memory becomes invisible. Returns 204.

    Args:
        store_id (str):
        memory_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            store_id=store_id,
            memory_id=memory_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
