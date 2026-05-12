from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_version import AgentVersion
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    agent_id: str,
    version: int,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/{agent_id}/versions/{version}".format(
            agent_id=quote(str(agent_id), safe=""),
            version=quote(str(version), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentVersion | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = AgentVersion.from_dict(response.json())

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
) -> Response[AgentVersion | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    agent_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[AgentVersion | HTTPValidationError]:
    """Get Version

     Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        agent_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentVersion | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        version=version,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    agent_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> AgentVersion | HTTPValidationError | None:
    """Get Version

     Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        agent_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentVersion | HTTPValidationError
    """

    return sync_detailed(
        agent_id=agent_id,
        version=version,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    agent_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[AgentVersion | HTTPValidationError]:
    """Get Version

     Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        agent_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentVersion | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        version=version,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    agent_id: str,
    version: int,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> AgentVersion | HTTPValidationError | None:
    """Get Version

     Fetch one historical version's config snapshot.

    The snapshot reflects the agent's config at the time the version was
    written and is unaffected by subsequent updates or archival.

    Args:
        agent_id (str):
        version (int):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentVersion | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            agent_id=agent_id,
            version=version,
            client=client,
            authorization=authorization,
        )
    ).parsed
