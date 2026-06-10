from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.wf_run_wait_response import WfRunWaitResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    run_id: str,
    *,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["timeout"] = timeout

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/runs/{run_id}/wait".format(
            run_id=quote(str(run_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WfRunWaitResponse | None:
    if response.status_code == 200:
        response_200 = WfRunWaitResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | WfRunWaitResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRunWaitResponse]:
    """Await Run

     Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The ``await``-a-completion primitive (runs backing): one JSON round-trip returning the
    completion record — ``done`` + ``output``, or ``is_error`` + ``error``. A run still running
    after ``timeout`` seconds returns ``done=false`` with its current status; call again to keep
    blocking. Unlike the SSE ``/stream`` this is a plain request/response, so it works as an MCP
    tool — an agent can await a sub-run and join. A cross-tenant run 404s.

    Args:
        run_id (str):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRunWaitResponse]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        timeout=timeout,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRunWaitResponse | None:
    """Await Run

     Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The ``await``-a-completion primitive (runs backing): one JSON round-trip returning the
    completion record — ``done`` + ``output``, or ``is_error`` + ``error``. A run still running
    after ``timeout`` seconds returns ``done=false`` with its current status; call again to keep
    blocking. Unlike the SSE ``/stream`` this is a plain request/response, so it works as an MCP
    tool — an agent can await a sub-run and join. A cross-tenant run 404s.

    Args:
        run_id (str):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRunWaitResponse
    """

    return sync_detailed(
        run_id=run_id,
        client=client,
        timeout=timeout,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WfRunWaitResponse]:
    """Await Run

     Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The ``await``-a-completion primitive (runs backing): one JSON round-trip returning the
    completion record — ``done`` + ``output``, or ``is_error`` + ``error``. A run still running
    after ``timeout`` seconds returns ``done=false`` with its current status; call again to keep
    blocking. Unlike the SSE ``/stream`` this is a plain request/response, so it works as an MCP
    tool — an agent can await a sub-run and join. A cross-tenant run 404s.

    Args:
        run_id (str):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WfRunWaitResponse]
    """

    kwargs = _get_kwargs(
        run_id=run_id,
        timeout=timeout,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    run_id: str,
    *,
    client: AuthenticatedClient | Client,
    timeout: int | Unset = 30,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WfRunWaitResponse | None:
    """Await Run

     Block until the run reaches a terminal status (completed/errored/cancelled), or timeout.

    The ``await``-a-completion primitive (runs backing): one JSON round-trip returning the
    completion record — ``done`` + ``output``, or ``is_error`` + ``error``. A run still running
    after ``timeout`` seconds returns ``done=false`` with its current status; call again to keep
    blocking. Unlike the SSE ``/stream`` this is a plain request/response, so it works as an MCP
    tool — an agent can await a sub-run and join. A cross-tenant run 404s.

    Args:
        run_id (str):
        timeout (int | Unset):  Default: 30.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WfRunWaitResponse
    """

    return (
        await asyncio_detailed(
            run_id=run_id,
            client=client,
            timeout=timeout,
            authorization=authorization,
        )
    ).parsed
