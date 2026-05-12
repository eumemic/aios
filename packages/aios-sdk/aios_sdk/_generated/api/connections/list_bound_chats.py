from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_bound_chat import ListResponseBoundChat
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connection_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connections/{connection_id}/bound-chats".format(
            connection_id=quote(str(connection_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseBoundChat | None:
    if response.status_code == 200:
        response_200 = ListResponseBoundChat.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseBoundChat]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseBoundChat]:
    """Bound Chats

     List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseBoundChat]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseBoundChat | None:
    """Bound Chats

     List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseBoundChat
    """

    return sync_detailed(
        connection_id=connection_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseBoundChat]:
    """Bound Chats

     List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseBoundChat]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseBoundChat | None:
    """Bound Chats

     List operator-curated chat → session bindings on this connection.

    Includes both manually-bound rows (via ``bind_chat``) and per_chat
    auto-spawned rows (via the supervisor on first inbound from a new
    chat partner).

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseBoundChat
    """

    return (
        await asyncio_detailed(
            connection_id=connection_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
