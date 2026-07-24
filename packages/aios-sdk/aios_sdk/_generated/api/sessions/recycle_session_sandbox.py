from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.event import Event
from ...models.http_validation_error import HTTPValidationError
from ...models.sandbox_recycle_request import SandboxRecycleRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: SandboxRecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/sandbox/recycle".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Event | HTTPValidationError | None:
    if response.status_code == 202:
        response_202 = Event.from_dict(response.json())

        return response_202

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Event | HTTPValidationError]:
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
    body: SandboxRecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Recycle Sandbox

     Discard the sandbox writable layer and provision current config fresh.

    Durable workspace, repositories, memory stores, and uploads are bind mounts
    outside the container and survive. Packages, caches, and dotfiles do not.
    In-flight sandbox calls are cancelled with their normal retryable result.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (SandboxRecycleRequest): Explicit acknowledgement for discarding sandbox-local
            writable state.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Event | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
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
    body: SandboxRecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Recycle Sandbox

     Discard the sandbox writable layer and provision current config fresh.

    Durable workspace, repositories, memory stores, and uploads are bind mounts
    outside the container and survive. Packages, caches, and dotfiles do not.
    In-flight sandbox calls are cancelled with their normal retryable result.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (SandboxRecycleRequest): Explicit acknowledgement for discarding sandbox-local
            writable state.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Event | HTTPValidationError
    """

    return sync_detailed(
        session_id=session_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SandboxRecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Event | HTTPValidationError]:
    """Recycle Sandbox

     Discard the sandbox writable layer and provision current config fresh.

    Durable workspace, repositories, memory stores, and uploads are bind mounts
    outside the container and survive. Packages, caches, and dotfiles do not.
    In-flight sandbox calls are cancelled with their normal retryable result.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (SandboxRecycleRequest): Explicit acknowledgement for discarding sandbox-local
            writable state.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Event | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        session_id=session_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    session_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: SandboxRecycleRequest,
    authorization: None | str | Unset = UNSET,
) -> Event | HTTPValidationError | None:
    """Recycle Sandbox

     Discard the sandbox writable layer and provision current config fresh.

    Durable workspace, repositories, memory stores, and uploads are bind mounts
    outside the container and survive. Packages, caches, and dotfiles do not.
    In-flight sandbox calls are cancelled with their normal retryable result.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (SandboxRecycleRequest): Explicit acknowledgement for discarding sandbox-local
            writable state.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Event | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
