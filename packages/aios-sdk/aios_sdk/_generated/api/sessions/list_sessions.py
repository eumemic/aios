from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_session import ListResponseSession
from ...models.list_sessions_status_type_0 import ListSessionsStatusType0
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    cursor: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    status: ListSessionsStatusType0 | None | Unset = UNSET,
    parent_run_id: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_cursor: None | str | Unset
    if isinstance(cursor, Unset):
        json_cursor = UNSET
    else:
        json_cursor = cursor
    params["cursor"] = json_cursor

    json_agent_id: None | str | Unset
    if isinstance(agent_id, Unset):
        json_agent_id = UNSET
    else:
        json_agent_id = agent_id
    params["agent_id"] = json_agent_id

    json_status: None | str | Unset
    if isinstance(status, Unset):
        json_status = UNSET
    elif isinstance(status, ListSessionsStatusType0):
        json_status = status.value
    else:
        json_status = status
    params["status"] = json_status

    json_parent_run_id: None | str | Unset
    if isinstance(parent_run_id, Unset):
        json_parent_run_id = UNSET
    else:
        json_parent_run_id = parent_run_id
    params["parent_run_id"] = json_parent_run_id

    json_limit: int | None | Unset
    if isinstance(limit, Unset):
        json_limit = UNSET
    else:
        json_limit = limit
    params["limit"] = json_limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/sessions",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseSession | None:
    if response.status_code == 200:
        response_200 = ListResponseSession.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseSession]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    status: ListSessionsStatusType0 | None | Unset = UNSET,
    parent_run_id: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseSession]:
    """List

     List sessions, newest first, keyset-paginated.

    Soft-archived sessions are hidden by default. Two filters surface them so a
    workflow run's spent ``agent()`` children stay enumerable with their terminal
    status and token usage (#831): ``?parent_run_id=`` lists a run's children
    (alive or archived), and ``?status=archived`` lists the terminal ones. Each
    row carries the derived ``status`` ({active, idle, archived}) and cumulative
    ``usage``.

    Args:
        cursor (None | str | Unset):
        agent_id (None | str | Unset):
        status (ListSessionsStatusType0 | None | Unset):
        parent_run_id (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseSession]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        agent_id=agent_id,
        status=status,
        parent_run_id=parent_run_id,
        limit=limit,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    status: ListSessionsStatusType0 | None | Unset = UNSET,
    parent_run_id: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseSession | None:
    """List

     List sessions, newest first, keyset-paginated.

    Soft-archived sessions are hidden by default. Two filters surface them so a
    workflow run's spent ``agent()`` children stay enumerable with their terminal
    status and token usage (#831): ``?parent_run_id=`` lists a run's children
    (alive or archived), and ``?status=archived`` lists the terminal ones. Each
    row carries the derived ``status`` ({active, idle, archived}) and cumulative
    ``usage``.

    Args:
        cursor (None | str | Unset):
        agent_id (None | str | Unset):
        status (ListSessionsStatusType0 | None | Unset):
        parent_run_id (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseSession
    """

    return sync_detailed(
        client=client,
        cursor=cursor,
        agent_id=agent_id,
        status=status,
        parent_run_id=parent_run_id,
        limit=limit,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    status: ListSessionsStatusType0 | None | Unset = UNSET,
    parent_run_id: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseSession]:
    """List

     List sessions, newest first, keyset-paginated.

    Soft-archived sessions are hidden by default. Two filters surface them so a
    workflow run's spent ``agent()`` children stay enumerable with their terminal
    status and token usage (#831): ``?parent_run_id=`` lists a run's children
    (alive or archived), and ``?status=archived`` lists the terminal ones. Each
    row carries the derived ``status`` ({active, idle, archived}) and cumulative
    ``usage``.

    Args:
        cursor (None | str | Unset):
        agent_id (None | str | Unset):
        status (ListSessionsStatusType0 | None | Unset):
        parent_run_id (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseSession]
    """

    kwargs = _get_kwargs(
        cursor=cursor,
        agent_id=agent_id,
        status=status,
        parent_run_id=parent_run_id,
        limit=limit,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    cursor: None | str | Unset = UNSET,
    agent_id: None | str | Unset = UNSET,
    status: ListSessionsStatusType0 | None | Unset = UNSET,
    parent_run_id: None | str | Unset = UNSET,
    limit: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseSession | None:
    """List

     List sessions, newest first, keyset-paginated.

    Soft-archived sessions are hidden by default. Two filters surface them so a
    workflow run's spent ``agent()`` children stay enumerable with their terminal
    status and token usage (#831): ``?parent_run_id=`` lists a run's children
    (alive or archived), and ``?status=archived`` lists the terminal ones. Each
    row carries the derived ``status`` ({active, idle, archived}) and cumulative
    ``usage``.

    Args:
        cursor (None | str | Unset):
        agent_id (None | str | Unset):
        status (ListSessionsStatusType0 | None | Unset):
        parent_run_id (None | str | Unset):
        limit (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseSession
    """

    return (
        await asyncio_detailed(
            client=client,
            cursor=cursor,
            agent_id=agent_id,
            status=status,
            parent_run_id=parent_run_id,
            limit=limit,
            authorization=authorization,
        )
    ).parsed
