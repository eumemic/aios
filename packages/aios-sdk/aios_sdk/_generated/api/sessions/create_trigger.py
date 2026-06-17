from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.trigger_create import TriggerCreate
from ...models.trigger_created import TriggerCreated
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: TriggerCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/triggers".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TriggerCreated | None:
    if response.status_code == 201:
        response_201 = TriggerCreated.from_dict(response.json())

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
) -> Response[HTTPValidationError | TriggerCreated]:
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
    body: TriggerCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TriggerCreated]:
    """Create Trigger

     Add a trigger. Granular operation per #270 — there is no whole-list
    ``set`` surface on ``SessionUpdate``.

    For an ``external_event`` source the response carries ``ingest_token`` —
    the plaintext ingest secret, surfaced EXACTLY ONCE (mirrors
    ``RuntimeTokenIssued``). The full ingress URL
    (``POST /v1/triggers/ingest/{ingest_token}``) is derivable client-side;
    it is never stored and cannot be re-read.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (TriggerCreate): Request body for adding a trigger to a session.

            Carries a ``source`` (cron / one_shot) and an ``action``
            (sandbox_command / wake_owner). Also accepted in
            :class:`SessionCreate.triggers` for initial attachment at session
            creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TriggerCreated]
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
    body: TriggerCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TriggerCreated | None:
    """Create Trigger

     Add a trigger. Granular operation per #270 — there is no whole-list
    ``set`` surface on ``SessionUpdate``.

    For an ``external_event`` source the response carries ``ingest_token`` —
    the plaintext ingest secret, surfaced EXACTLY ONCE (mirrors
    ``RuntimeTokenIssued``). The full ingress URL
    (``POST /v1/triggers/ingest/{ingest_token}``) is derivable client-side;
    it is never stored and cannot be re-read.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (TriggerCreate): Request body for adding a trigger to a session.

            Carries a ``source`` (cron / one_shot) and an ``action``
            (sandbox_command / wake_owner). Also accepted in
            :class:`SessionCreate.triggers` for initial attachment at session
            creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TriggerCreated
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
    body: TriggerCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TriggerCreated]:
    """Create Trigger

     Add a trigger. Granular operation per #270 — there is no whole-list
    ``set`` surface on ``SessionUpdate``.

    For an ``external_event`` source the response carries ``ingest_token`` —
    the plaintext ingest secret, surfaced EXACTLY ONCE (mirrors
    ``RuntimeTokenIssued``). The full ingress URL
    (``POST /v1/triggers/ingest/{ingest_token}``) is derivable client-side;
    it is never stored and cannot be re-read.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (TriggerCreate): Request body for adding a trigger to a session.

            Carries a ``source`` (cron / one_shot) and an ``action``
            (sandbox_command / wake_owner). Also accepted in
            :class:`SessionCreate.triggers` for initial attachment at session
            creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TriggerCreated]
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
    body: TriggerCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TriggerCreated | None:
    """Create Trigger

     Add a trigger. Granular operation per #270 — there is no whole-list
    ``set`` surface on ``SessionUpdate``.

    For an ``external_event`` source the response carries ``ingest_token`` —
    the plaintext ingest secret, surfaced EXACTLY ONCE (mirrors
    ``RuntimeTokenIssued``). The full ingress URL
    (``POST /v1/triggers/ingest/{ingest_token}``) is derivable client-side;
    it is never stored and cannot be re-read.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (TriggerCreate): Request body for adding a trigger to a session.

            Carries a ``source`` (cron / one_shot) and an ``action``
            (sandbox_command / wake_owner). Also accepted in
            :class:`SessionCreate.triggers` for initial attachment at session
            creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TriggerCreated
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
