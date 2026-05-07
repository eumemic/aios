from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_session import ListResponseSession
from ...models.list_v1_sessions_get_status_type_0 import ListV1SessionsGetStatusType0
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    agent_id: None | str | Unset = UNSET,
    status: ListV1SessionsGetStatusType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_agent_id: None | str | Unset
    if isinstance(agent_id, Unset):
        json_agent_id = UNSET
    else:
        json_agent_id = agent_id
    params["agent_id"] = json_agent_id

    json_status: None | str | Unset
    if isinstance(status, Unset):
        json_status = UNSET
    elif isinstance(status, ListV1SessionsGetStatusType0):
        json_status = status.value
    else:
        json_status = status
    params["status"] = json_status

    params["limit"] = limit

    json_after: None | str | Unset
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

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
    agent_id: None | str | Unset = UNSET,
    status: ListV1SessionsGetStatusType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseSession]:
    """List

    Args:
        agent_id (None | str | Unset):
        status (ListV1SessionsGetStatusType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseSession]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        status=status,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    agent_id: None | str | Unset = UNSET,
    status: ListV1SessionsGetStatusType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseSession | None:
    """List

    Args:
        agent_id (None | str | Unset):
        status (ListV1SessionsGetStatusType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseSession
    """

    return sync_detailed(
        client=client,
        agent_id=agent_id,
        status=status,
        limit=limit,
        after=after,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    agent_id: None | str | Unset = UNSET,
    status: ListV1SessionsGetStatusType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseSession]:
    """List

    Args:
        agent_id (None | str | Unset):
        status (ListV1SessionsGetStatusType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseSession]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        status=status,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    agent_id: None | str | Unset = UNSET,
    status: ListV1SessionsGetStatusType0 | None | Unset = UNSET,
    limit: int | Unset = 50,
    after: None | str | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseSession | None:
    """List

    Args:
        agent_id (None | str | Unset):
        status (ListV1SessionsGetStatusType0 | None | Unset):
        limit (int | Unset):  Default: 50.
        after (None | str | Unset):
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
            agent_id=agent_id,
            status=status,
            limit=limit,
            after=after,
            authorization=authorization,
        )
    ).parsed
