from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_connections_mode_type_0 import ListConnectionsModeType0
from ...models.list_response_connection import ListResponseConnection
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    connector: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    mode: ListConnectionsModeType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_connector: None | str | Unset
    if isinstance(connector, Unset):
        json_connector = UNSET
    else:
        json_connector = connector
    params["connector"] = json_connector

    json_session_id: None | str | Unset
    if isinstance(session_id, Unset):
        json_session_id = UNSET
    else:
        json_session_id = session_id
    params["session_id"] = json_session_id

    json_mode: None | str | Unset
    if isinstance(mode, Unset):
        json_mode = UNSET
    elif isinstance(mode, ListConnectionsModeType0):
        json_mode = mode.value
    else:
        json_mode = mode
    params["mode"] = json_mode

    params["limit"] = limit

    json_after: None | str | Unset
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connections",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseConnection | None:
    if response.status_code == 200:
        response_200 = ListResponseConnection.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseConnection]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    mode: ListConnectionsModeType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseConnection]:
    r"""List

     List connections, newest first, excluding archived. Cursor pagination via ``after``.

    Filters: ``connector`` (e.g. ``\"telegram\"``), ``session_id`` (only
    connections in single_session mode bound to that session), ``mode``
    (``detached`` / ``single_session`` / ``per_chat``). Filters compose.

    Args:
        connector (None | str | Unset):
        session_id (None | str | Unset):
        mode (ListConnectionsModeType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseConnection]
    """

    kwargs = _get_kwargs(
        connector=connector,
        session_id=session_id,
        mode=mode,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    connector: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    mode: ListConnectionsModeType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseConnection | None:
    r"""List

     List connections, newest first, excluding archived. Cursor pagination via ``after``.

    Filters: ``connector`` (e.g. ``\"telegram\"``), ``session_id`` (only
    connections in single_session mode bound to that session), ``mode``
    (``detached`` / ``single_session`` / ``per_chat``). Filters compose.

    Args:
        connector (None | str | Unset):
        session_id (None | str | Unset):
        mode (ListConnectionsModeType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseConnection
    """

    return sync_detailed(
        client=client,
        connector=connector,
        session_id=session_id,
        mode=mode,
        limit=limit,
        after=after,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    connector: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    mode: ListConnectionsModeType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseConnection]:
    r"""List

     List connections, newest first, excluding archived. Cursor pagination via ``after``.

    Filters: ``connector`` (e.g. ``\"telegram\"``), ``session_id`` (only
    connections in single_session mode bound to that session), ``mode``
    (``detached`` / ``single_session`` / ``per_chat``). Filters compose.

    Args:
        connector (None | str | Unset):
        session_id (None | str | Unset):
        mode (ListConnectionsModeType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseConnection]
    """

    kwargs = _get_kwargs(
        connector=connector,
        session_id=session_id,
        mode=mode,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    connector: None | str | Unset = UNSET,
    session_id: None | str | Unset = UNSET,
    mode: ListConnectionsModeType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseConnection | None:
    r"""List

     List connections, newest first, excluding archived. Cursor pagination via ``after``.

    Filters: ``connector`` (e.g. ``\"telegram\"``), ``session_id`` (only
    connections in single_session mode bound to that session), ``mode``
    (``detached`` / ``single_session`` / ``per_chat``). Filters compose.

    Args:
        connector (None | str | Unset):
        session_id (None | str | Unset):
        mode (ListConnectionsModeType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseConnection
    """

    return (
        await asyncio_detailed(
            client=client,
            connector=connector,
            session_id=session_id,
            mode=mode,
            limit=limit,
            after=after,
            authorization=authorization,
        )
    ).parsed
