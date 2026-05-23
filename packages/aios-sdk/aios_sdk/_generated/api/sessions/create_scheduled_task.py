from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.scheduled_task_create import ScheduledTaskCreate
from ...models.scheduled_task_echo import ScheduledTaskEcho
from ...types import UNSET, Response, Unset


def _get_kwargs(
    session_id: str,
    *,
    body: ScheduledTaskCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/sessions/{session_id}/scheduled-tasks".format(
            session_id=quote(str(session_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ScheduledTaskEcho | None:
    if response.status_code == 201:
        response_201 = ScheduledTaskEcho.from_dict(response.json())

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
) -> Response[HTTPValidationError | ScheduledTaskEcho]:
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
    body: ScheduledTaskCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ScheduledTaskEcho]:
    """Create Scheduled Task

     Add a scheduled task. Granular operation per #270 — there is no
    whole-list ``set`` surface on ``SessionUpdate``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ScheduledTaskCreate): Request body for adding a scheduled task to a session.

            Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
            attachment at session creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ScheduledTaskEcho]
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
    body: ScheduledTaskCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ScheduledTaskEcho | None:
    """Create Scheduled Task

     Add a scheduled task. Granular operation per #270 — there is no
    whole-list ``set`` surface on ``SessionUpdate``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ScheduledTaskCreate): Request body for adding a scheduled task to a session.

            Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
            attachment at session creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ScheduledTaskEcho
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
    body: ScheduledTaskCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ScheduledTaskEcho]:
    """Create Scheduled Task

     Add a scheduled task. Granular operation per #270 — there is no
    whole-list ``set`` surface on ``SessionUpdate``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ScheduledTaskCreate): Request body for adding a scheduled task to a session.

            Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
            attachment at session creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ScheduledTaskEcho]
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
    body: ScheduledTaskCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ScheduledTaskEcho | None:
    """Create Scheduled Task

     Add a scheduled task. Granular operation per #270 — there is no
    whole-list ``set`` surface on ``SessionUpdate``.

    Args:
        session_id (str):
        authorization (None | str | Unset):
        body (ScheduledTaskCreate): Request body for adding a scheduled task to a session.

            Also accepted in :class:`SessionCreate.scheduled_tasks` for initial
            attachment at session creation.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ScheduledTaskEcho
    """

    return (
        await asyncio_detailed(
            session_id=session_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
