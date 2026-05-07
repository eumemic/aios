from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent import Agent
from ...models.agent_update import AgentUpdate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    agent_id: str,
    *,
    body: AgentUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/{agent_id}".format(
            agent_id=quote(str(agent_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Agent | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Agent.from_dict(response.json())

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
) -> Response[Agent | HTTPValidationError]:
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
    body: AgentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Agent | HTTPValidationError]:
    """Update

     Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).

    Args:
        agent_id (str):
        authorization (None | str | Unset):
        body (AgentUpdate): Request body for ``PUT /v1/agents/{id}``.

            All config fields are optional; omitted fields are preserved. The
            ``version`` field is required for optimistic concurrency — it must match
            the current version. If the update produces a change, a new version is
            created; otherwise the existing version is returned unchanged.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Agent | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        body=body,
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
    body: AgentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Agent | HTTPValidationError | None:
    """Update

     Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).

    Args:
        agent_id (str):
        authorization (None | str | Unset):
        body (AgentUpdate): Request body for ``PUT /v1/agents/{id}``.

            All config fields are optional; omitted fields are preserved. The
            ``version`` field is required for optimistic concurrency — it must match
            the current version. If the update produces a change, a new version is
            created; otherwise the existing version is returned unchanged.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Agent | HTTPValidationError
    """

    return sync_detailed(
        agent_id=agent_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AgentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Agent | HTTPValidationError]:
    """Update

     Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).

    Args:
        agent_id (str):
        authorization (None | str | Unset):
        body (AgentUpdate): Request body for ``PUT /v1/agents/{id}``.

            All config fields are optional; omitted fields are preserved. The
            ``version`` field is required for optimistic concurrency — it must match
            the current version. If the update produces a change, a new version is
            created; otherwise the existing version is returned unchanged.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Agent | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        agent_id=agent_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    agent_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: AgentUpdate,
    authorization: None | str | Unset = UNSET,
) -> Agent | HTTPValidationError | None:
    """Update

     Update an agent, creating a new immutable version.

    The ``version`` field on the body is required for optimistic concurrency
    and must match the agent's current version. Omitted config fields are
    preserved from the previous version. If the merged config is identical
    to the current version, no new version is created and the existing one
    is returned unchanged (no-op).

    Args:
        agent_id (str):
        authorization (None | str | Unset):
        body (AgentUpdate): Request body for ``PUT /v1/agents/{id}``.

            All config fields are optional; omitted fields are preserved. The
            ``version`` field is required for optimistic concurrency — it must match
            the current version. If the update produces a change, a new version is
            created; otherwise the existing version is returned unchanged.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Agent | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            agent_id=agent_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
