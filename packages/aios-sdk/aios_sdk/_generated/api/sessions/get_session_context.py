from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.context_response import ContextResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions/{session_id}/context".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ContextResponse | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ContextResponse.from_dict(response.json())

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
) -> Response[ContextResponse | HTTPValidationError]:
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
    authorization: None | str | Unset = UNSET,
) -> Response[ContextResponse | HTTPValidationError]:
    """Get Context

     Return the chat-completions payload the worker would send next.

    Dry-run preview for debugging prompt construction. Reuses the exact
    composer the worker's step function uses (:func:`compose_step_context`).
    Side effects (skill provisioning, session-status bumps, event
    appends) are omitted; the endpoint is read-only.

    One known divergence from the worker's output: unresolved tool_calls
    that the worker is currently executing render as ``_PENDING_EXTERNAL``
    here (the API process has no view into the worker's inflight_tool_registry).
    The worker would render them as ``_PENDING_BACKGROUND``. Custom and
    awaiting-confirm calls render identically on both sides.

    Image attachments — including those under ``/workspace/...`` — render
    identically to the worker: ``compose_step_context`` resolves the
    bind-mount source from the session row, not a worker-only sandbox
    handle.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
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
    authorization: None | str | Unset = UNSET,
) -> ContextResponse | HTTPValidationError | None:
    """Get Context

     Return the chat-completions payload the worker would send next.

    Dry-run preview for debugging prompt construction. Reuses the exact
    composer the worker's step function uses (:func:`compose_step_context`).
    Side effects (skill provisioning, session-status bumps, event
    appends) are omitted; the endpoint is read-only.

    One known divergence from the worker's output: unresolved tool_calls
    that the worker is currently executing render as ``_PENDING_EXTERNAL``
    here (the API process has no view into the worker's inflight_tool_registry).
    The worker would render them as ``_PENDING_BACKGROUND``. Custom and
    awaiting-confirm calls render identically on both sides.

    Image attachments — including those under ``/workspace/...`` — render
    identically to the worker: ``compose_step_context`` resolves the
    bind-mount source from the session row, not a worker-only sandbox
    handle.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextResponse | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[ContextResponse | HTTPValidationError]:
    """Get Context

     Return the chat-completions payload the worker would send next.

    Dry-run preview for debugging prompt construction. Reuses the exact
    composer the worker's step function uses (:func:`compose_step_context`).
    Side effects (skill provisioning, session-status bumps, event
    appends) are omitted; the endpoint is read-only.

    One known divergence from the worker's output: unresolved tool_calls
    that the worker is currently executing render as ``_PENDING_EXTERNAL``
    here (the API process has no view into the worker's inflight_tool_registry).
    The worker would render them as ``_PENDING_BACKGROUND``. Custom and
    awaiting-confirm calls render identically on both sides.

    Image attachments — including those under ``/workspace/...`` — render
    identically to the worker: ``compose_step_context`` resolves the
    bind-mount source from the session row, not a worker-only sandbox
    handle.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ContextResponse | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> ContextResponse | HTTPValidationError | None:
    """Get Context

     Return the chat-completions payload the worker would send next.

    Dry-run preview for debugging prompt construction. Reuses the exact
    composer the worker's step function uses (:func:`compose_step_context`).
    Side effects (skill provisioning, session-status bumps, event
    appends) are omitted; the endpoint is read-only.

    One known divergence from the worker's output: unresolved tool_calls
    that the worker is currently executing render as ``_PENDING_EXTERNAL``
    here (the API process has no view into the worker's inflight_tool_registry).
    The worker would render them as ``_PENDING_BACKGROUND``. Custom and
    awaiting-confirm calls render identically on both sides.

    Image attachments — including those under ``/workspace/...`` — render
    identically to the worker: ``compose_step_context`` resolves the
    bind-mount source from the session row, not a worker-only sandbox
    handle.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ContextResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
