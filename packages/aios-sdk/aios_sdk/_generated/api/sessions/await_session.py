from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session_await_response import SessionAwaitResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    watermark: int | None | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_watermark: int | None | Unset
    if isinstance(watermark, Unset):
        json_watermark = UNSET
    else:
        json_watermark = watermark
    params["watermark"] = json_watermark

    params["timeout"] = timeout

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/await".format(
            session_id=quote(str(session_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SessionAwaitResponse | None:
    if response.status_code == 200:
        response_200 = SessionAwaitResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | SessionAwaitResponse]:
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
    watermark: int | None | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionAwaitResponse]:
    """Await Session

     Block until the session has fully reacted to a stimulus (``watermark``; defaults to the
    session's ``last_stimulus_seq`` at call time), or ``timeout`` seconds elapse — then
    ``done=false`` so the caller re-polls.

    The session **quiescence drive-and-join** alias: one JSON round-trip, MCP-usable so an agent
    can drive a session and join when it has fully reacted. Correlating a *request* response is
    the unified awaiter's job (``GET /v1/invocations/{task_id}/await?request_id=``). A
    cross-tenant session 404s before any subscription opens.

    Args:
        session_id (str):
        watermark (int | None | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionAwaitResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        watermark=watermark,
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
    watermark: int | None | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionAwaitResponse | None:
    """Await Session

     Block until the session has fully reacted to a stimulus (``watermark``; defaults to the
    session's ``last_stimulus_seq`` at call time), or ``timeout`` seconds elapse — then
    ``done=false`` so the caller re-polls.

    The session **quiescence drive-and-join** alias: one JSON round-trip, MCP-usable so an agent
    can drive a session and join when it has fully reacted. Correlating a *request* response is
    the unified awaiter's job (``GET /v1/invocations/{task_id}/await?request_id=``). A
    cross-tenant session 404s before any subscription opens.

    Args:
        session_id (str):
        watermark (int | None | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionAwaitResponse
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        watermark=watermark,
        timeout=timeout,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    watermark: int | None | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionAwaitResponse]:
    """Await Session

     Block until the session has fully reacted to a stimulus (``watermark``; defaults to the
    session's ``last_stimulus_seq`` at call time), or ``timeout`` seconds elapse — then
    ``done=false`` so the caller re-polls.

    The session **quiescence drive-and-join** alias: one JSON round-trip, MCP-usable so an agent
    can drive a session and join when it has fully reacted. Correlating a *request* response is
    the unified awaiter's job (``GET /v1/invocations/{task_id}/await?request_id=``). A
    cross-tenant session 404s before any subscription opens.

    Args:
        session_id (str):
        watermark (int | None | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionAwaitResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        watermark=watermark,
        timeout=timeout,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    watermark: int | None | Unset = UNSET,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionAwaitResponse | None:
    """Await Session

     Block until the session has fully reacted to a stimulus (``watermark``; defaults to the
    session's ``last_stimulus_seq`` at call time), or ``timeout`` seconds elapse — then
    ``done=false`` so the caller re-polls.

    The session **quiescence drive-and-join** alias: one JSON round-trip, MCP-usable so an agent
    can drive a session and join when it has fully reacted. Correlating a *request* response is
    the unified awaiter's job (``GET /v1/invocations/{task_id}/await?request_id=``). A
    cross-tenant session 404s before any subscription opens.

    Args:
        session_id (str):
        watermark (int | None | Unset):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionAwaitResponse
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            watermark=watermark,
            timeout=timeout,
            authorization=authorization,
        )
    ).parsed
