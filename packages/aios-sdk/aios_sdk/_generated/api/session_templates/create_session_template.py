from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.session_template import SessionTemplate
from ...models.session_template_create import SessionTemplateCreate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SessionTemplateCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/session-templates",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SessionTemplate | None:
    if response.status_code == 201:
        response_201 = SessionTemplate.from_dict(response.json())

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
) -> Response[HTTPValidationError | SessionTemplate]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionTemplate]:
    """Create

     Create a session template — a frozen recipe for per_chat session spawn.

    Captures the agent + environment + vaults + memory stores that a
    per_chat connection will use when spawning a session for a new chat
    partner. Pin ``agent_version`` to a specific version for deterministic
    spawning, or leave unset to track the agent's latest.

    Args:
        authorization (None | str | Unset):
        body (SessionTemplateCreate): Request body for ``POST /v1/session-templates``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionTemplate]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionTemplate | None:
    """Create

     Create a session template — a frozen recipe for per_chat session spawn.

    Captures the agent + environment + vaults + memory stores that a
    per_chat connection will use when spawning a session for a new chat
    partner. Pin ``agent_version`` to a specific version for deterministic
    spawning, or leave unset to track the agent's latest.

    Args:
        authorization (None | str | Unset):
        body (SessionTemplateCreate): Request body for ``POST /v1/session-templates``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionTemplate
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SessionTemplate]:
    """Create

     Create a session template — a frozen recipe for per_chat session spawn.

    Captures the agent + environment + vaults + memory stores that a
    per_chat connection will use when spawning a session for a new chat
    partner. Pin ``agent_version`` to a specific version for deterministic
    spawning, or leave unset to track the agent's latest.

    Args:
        authorization (None | str | Unset):
        body (SessionTemplateCreate): Request body for ``POST /v1/session-templates``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SessionTemplate]
    """

    kwargs = _get_kwargs(
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SessionTemplateCreate,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SessionTemplate | None:
    """Create

     Create a session template — a frozen recipe for per_chat session spawn.

    Captures the agent + environment + vaults + memory stores that a
    per_chat connection will use when spawning a session for a new chat
    partner. Pin ``agent_version`` to a specific version for deterministic
    spawning, or leave unset to track the agent's latest.

    Args:
        authorization (None | str | Unset):
        body (SessionTemplateCreate): Request body for ``POST /v1/session-templates``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SessionTemplate
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
