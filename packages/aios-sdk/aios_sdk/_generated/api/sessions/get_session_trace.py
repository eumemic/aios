from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trace_response import TraceResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["verbose"] = verbose

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/trace".format(
            session_id=quote(str(session_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TraceResponse | None:
    if response.status_code == 200:
        response_200 = TraceResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | TraceResponse]:
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
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TraceResponse]:
    """Get Session Trace

     One-call linear trace rooted at a session + all nested sub-runs/sessions (#1149).

    The session-root counterpart of ``GET /v1/runs/{id}/trace``: walks the
    parent→child invocation-edge tree from this session (its ``agent()`` peer
    sessions and any runs it launched via the still-live ``launcher_session_id``
    FK), normalizes each node to ``terminal_state`` + raw ``error_kind``, and
    interleaves journals into a flat DFS-pre-order list. See the run-trace
    endpoint for the verbosity / ordering / scope caveats. A cross-tenant session
    404s.

    Args:
        session_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TraceResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        verbose=verbose,
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
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TraceResponse | None:
    """Get Session Trace

     One-call linear trace rooted at a session + all nested sub-runs/sessions (#1149).

    The session-root counterpart of ``GET /v1/runs/{id}/trace``: walks the
    parent→child invocation-edge tree from this session (its ``agent()`` peer
    sessions and any runs it launched via the still-live ``launcher_session_id``
    FK), normalizes each node to ``terminal_state`` + raw ``error_kind``, and
    interleaves journals into a flat DFS-pre-order list. See the run-trace
    endpoint for the verbosity / ordering / scope caveats. A cross-tenant session
    404s.

    Args:
        session_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TraceResponse
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        verbose=verbose,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TraceResponse]:
    """Get Session Trace

     One-call linear trace rooted at a session + all nested sub-runs/sessions (#1149).

    The session-root counterpart of ``GET /v1/runs/{id}/trace``: walks the
    parent→child invocation-edge tree from this session (its ``agent()`` peer
    sessions and any runs it launched via the still-live ``launcher_session_id``
    FK), normalizes each node to ``terminal_state`` + raw ``error_kind``, and
    interleaves journals into a flat DFS-pre-order list. See the run-trace
    endpoint for the verbosity / ordering / scope caveats. A cross-tenant session
    404s.

    Args:
        session_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TraceResponse]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        verbose=verbose,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    verbose: bool | Unset = False,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TraceResponse | None:
    """Get Session Trace

     One-call linear trace rooted at a session + all nested sub-runs/sessions (#1149).

    The session-root counterpart of ``GET /v1/runs/{id}/trace``: walks the
    parent→child invocation-edge tree from this session (its ``agent()`` peer
    sessions and any runs it launched via the still-live ``launcher_session_id``
    FK), normalizes each node to ``terminal_state`` + raw ``error_kind``, and
    interleaves journals into a flat DFS-pre-order list. See the run-trace
    endpoint for the verbosity / ordering / scope caveats. A cross-tenant session
    404s.

    Args:
        session_id (str):
        verbose (bool | Unset):  Default: False.
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TraceResponse
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            verbose=verbose,
            authorization=authorization,
        )
    ).parsed
