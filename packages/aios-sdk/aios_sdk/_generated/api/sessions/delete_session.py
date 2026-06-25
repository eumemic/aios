from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session import Session
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
        "method": "delete",
        "url": "/v1/sessions/{session_id}".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | Session | None:
    if response.status_code == 200:
        response_200 = Session.from_dict(response.json())

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
) -> Response[HTTPValidationError | Session]:
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
) -> Response[HTTPValidationError | Session]:
    """Delete

     Soft-archive a session (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the session from default lists (same
    behavior as ``archive_session``); events, vaults, and bindings are
    retained. Bare DELETE is never silently destructive; for the
    irreversible hard-delete (cascade of events / vaults / bindings) use
    ``POST /v1/sessions/{session_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
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
) -> HTTPValidationError | Session | None:
    """Delete

     Soft-archive a session (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the session from default lists (same
    behavior as ``archive_session``); events, vaults, and bindings are
    retained. Bare DELETE is never silently destructive; for the
    irreversible hard-delete (cascade of events / vaults / bindings) use
    ``POST /v1/sessions/{session_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
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
) -> Response[HTTPValidationError | Session]:
    """Delete

     Soft-archive a session (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the session from default lists (same
    behavior as ``archive_session``); events, vaults, and bindings are
    retained. Bare DELETE is never silently destructive; for the
    irreversible hard-delete (cascade of events / vaults / bindings) use
    ``POST /v1/sessions/{session_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | Session]
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
) -> HTTPValidationError | Session | None:
    """Delete

     Soft-archive a session (bare DELETE = soft-archive; T2 convention).

    Sets ``archived_at`` and hides the session from default lists (same
    behavior as ``archive_session``); events, vaults, and bindings are
    retained. Bare DELETE is never silently destructive; for the
    irreversible hard-delete (cascade of events / vaults / bindings) use
    ``POST /v1/sessions/{session_id}/purge``.

    Idempotent: a repeat bare DELETE (or a DELETE after ``/archive``) returns
    the existing archived row with 200, not 404.

    Args:
        session_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | Session
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            authorization=authorization,
        )
    ).parsed
