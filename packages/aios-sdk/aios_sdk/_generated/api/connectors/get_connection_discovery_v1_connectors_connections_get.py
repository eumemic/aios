from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connectors/connections",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Any | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\", \"connection_id\": \"...\", \"account\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\", \"connection_id\": \"...\", \"account\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\", \"connection_id\": \"...\", \"account\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\", \"connection_id\": \"...\", \"account\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            authorization=authorization,
        )
    ).parsed
