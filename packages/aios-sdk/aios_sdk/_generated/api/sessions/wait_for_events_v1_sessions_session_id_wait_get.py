from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.wait_response import WaitResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    after: int | Unset = 0,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["after"] = after

    params["timeout"] = timeout

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/wait".format(
            session_id=quote(str(session_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WaitResponse | None:
    if response.status_code == 200:
        response_200 = WaitResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | WaitResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = 0,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WaitResponse]:
    """Wait For Events

     Long-poll for new events past sequence number ``after``.

    Blocks up to ``timeout`` seconds for events to arrive; returns an empty
    list if none land in time. Alternative to SSE for clients whose HTTP
    stack can't reliably consume server-sent events (notably Node's
    ``fetch`` — see issue #40).

    Pass the response's ``next_after`` as ``?after=`` on the next call to
    resume from where you left off. (The query param was previously named
    ``after_seq``; see issue #389.)

    Args:
        session_id (str):
        after (int | Unset):  Default: 0.
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WaitResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after=after,
        timeout=timeout,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = 0,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WaitResponse | None:
    """Wait For Events

     Long-poll for new events past sequence number ``after``.

    Blocks up to ``timeout`` seconds for events to arrive; returns an empty
    list if none land in time. Alternative to SSE for clients whose HTTP
    stack can't reliably consume server-sent events (notably Node's
    ``fetch`` — see issue #40).

    Pass the response's ``next_after`` as ``?after=`` on the next call to
    resume from where you left off. (The query param was previously named
    ``after_seq``; see issue #389.)

    Args:
        session_id (str):
        after (int | Unset):  Default: 0.
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WaitResponse
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        after=after,
        timeout=timeout,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = 0,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WaitResponse]:
    """Wait For Events

     Long-poll for new events past sequence number ``after``.

    Blocks up to ``timeout`` seconds for events to arrive; returns an empty
    list if none land in time. Alternative to SSE for clients whose HTTP
    stack can't reliably consume server-sent events (notably Node's
    ``fetch`` — see issue #40).

    Pass the response's ``next_after`` as ``?after=`` on the next call to
    resume from where you left off. (The query param was previously named
    ``after_seq``; see issue #389.)

    Args:
        session_id (str):
        after (int | Unset):  Default: 0.
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WaitResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        after=after,
        timeout=timeout,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    after: int | Unset = 0,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WaitResponse | None:
    """Wait For Events

     Long-poll for new events past sequence number ``after``.

    Blocks up to ``timeout`` seconds for events to arrive; returns an empty
    list if none land in time. Alternative to SSE for clients whose HTTP
    stack can't reliably consume server-sent events (notably Node's
    ``fetch`` — see issue #40).

    Pass the response's ``next_after`` as ``?after=`` on the next call to
    resume from where you left off. (The query param was previously named
    ``after_seq``; see issue #389.)

    Args:
        session_id (str):
        after (int | Unset):  Default: 0.
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WaitResponse
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            after=after,
            timeout=timeout,
            authorization=authorization,
        )
    ).parsed
