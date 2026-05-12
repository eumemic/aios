from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.bind_chat_request import BindChatRequest
from ...models.bound_chat import BoundChat
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connection_id: str,
    *,
    body: BindChatRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connections/{connection_id}/bind-chat".format(
            connection_id=quote(str(connection_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> BoundChat | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = BoundChat.from_dict(response.json())

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
) -> Response[BoundChat | HTTPValidationError]:
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
    body: BindChatRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BoundChat | HTTPValidationError]:
    """Bind Chat

     Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (BindChatRequest): Request body for ``POST /v1/connections/{id}/bind-chat``.

            Pre-populates a ``connection_chat_sessions`` row so inbound on
            ``chat_id`` routes to ``session_id`` regardless of the connection's
            mode-default fallback (#215).  Operators use this to point
            different chats on a single account at different operator-curated
            existing sessions.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BoundChat | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
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
    body: BindChatRequest,
    authorization: None | str | Unset = UNSET,
) -> BoundChat | HTTPValidationError | None:
    """Bind Chat

     Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (BindChatRequest): Request body for ``POST /v1/connections/{id}/bind-chat``.

            Pre-populates a ``connection_chat_sessions`` row so inbound on
            ``chat_id`` routes to ``session_id`` regardless of the connection's
            mode-default fallback (#215).  Operators use this to point
            different chats on a single account at different operator-curated
            existing sessions.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BoundChat | HTTPValidationError
    """

    return sync_detailed(
        connection_id=connection_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BindChatRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[BoundChat | HTTPValidationError]:
    """Bind Chat

     Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (BindChatRequest): Request body for ``POST /v1/connections/{id}/bind-chat``.

            Pre-populates a ``connection_chat_sessions`` row so inbound on
            ``chat_id`` routes to ``session_id`` regardless of the connection's
            mode-default fallback (#215).  Operators use this to point
            different chats on a single account at different operator-curated
            existing sessions.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[BoundChat | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: BindChatRequest,
    authorization: None | str | Unset = UNSET,
) -> BoundChat | HTTPValidationError | None:
    """Bind Chat

     Operator-curate a chat → session mapping (#215).

    A row in ``connection_chat_sessions`` overrides the connection's
    mode-default fallback for that ``chat_id``.  Operators use this to
    point different chats on a single account at different existing
    sessions — the middle case the unified ``connections`` shape didn't
    cover after #205.

    Idempotent on ``(connection_id, chat_id)``: a second call with the
    same chat returns the existing row (its ``session_id`` may differ
    from the requested one if a concurrent writer landed first or the
    supervisor pre-populated it via per_chat spawn).

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (BindChatRequest): Request body for ``POST /v1/connections/{id}/bind-chat``.

            Pre-populates a ``connection_chat_sessions`` row so inbound on
            ``chat_id`` routes to ``session_id`` regardless of the connection's
            mode-default fallback (#215).  Operators use this to point
            different chats on a single account at different operator-curated
            existing sessions.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        BoundChat | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connection_id=connection_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
