from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_agent_version import ListResponseAgentVersion
from ...types import UNSET, Response, Unset


def _get_kwargs(
    agent_id: str,
    *,
    limit: int | Unset = 50,
    after: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["limit"] = limit

    json_after: int | None | Unset
    if isinstance(after, Unset):
        json_after = UNSET
    else:
        json_after = after
    params["after"] = json_after

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/{agent_id}/versions".format(
            agent_id=quote(str(agent_id), safe=""),
        ),
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseAgentVersion | None:
    if response.status_code == 200:
        response_200 = ListResponseAgentVersion.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseAgentVersion]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    after: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseAgentVersion]:
    """List Versions

     List historical versions of an agent, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete snapshot of the agent's config at the time it was created.

    Args:
        agent_id (str):
        limit (int | Unset):  Default: 50.
        after (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseAgentVersion]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    after: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseAgentVersion | None:
    """List Versions

     List historical versions of an agent, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete snapshot of the agent's config at the time it was created.

    Args:
        agent_id (str):
        limit (int | Unset):  Default: 50.
        after (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseAgentVersion
    """

    return sync_detailed(
        agent_id=agent_id,
        client=client,
        limit=limit,
        after=after,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    after: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseAgentVersion]:
    """List Versions

     List historical versions of an agent, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete snapshot of the agent's config at the time it was created.

    Args:
        agent_id (str):
        limit (int | Unset):  Default: 50.
        after (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseAgentVersion]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        limit=limit,
        after=after,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = 50,
    after: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseAgentVersion | None:
    """List Versions

     List historical versions of an agent, newest first.

    Cursor pagination by version number: pass ``after`` from a previous
    response's ``next_after`` to get the next page. Each version is a
    complete snapshot of the agent's config at the time it was created.

    Args:
        agent_id (str):
        limit (int | Unset):  Default: 50.
        after (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseAgentVersion
    """

    return (
        await asyncio_detailed(
            agent_id=agent_id,
            client=client,
            limit=limit,
            after=after,
            authorization=authorization,
        )
    ).parsed
