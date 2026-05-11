from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_inbound_request import ConnectorInboundRequest
from ...models.connector_inbound_response import ConnectorInboundResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectorInboundRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/inbound",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorInboundResponse | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = ConnectorInboundResponse.from_dict(response.json())

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
) -> Response[ConnectorInboundResponse | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorInboundRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorInboundResponse | HTTPValidationError]:
    """Post Inbound

     Append an inbound user message to the session bound to the caller's connection.

    Idempotent on ``body.event_id`` — replays return the original
    event id with ``deduped=True``.  Drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (ConnectorInboundRequest): Body for ``POST /v1/connectors/inbound``.

            Authenticated via ``ConnectorAuthDep`` so the connection_id is
            server-resolved from the bearer token — clients don't pick which
            connection their inbound lands on.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorInboundResponse | HTTPValidationError]
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
    body: ConnectorInboundRequest,
    authorization: None | str | Unset = UNSET,
) -> ConnectorInboundResponse | HTTPValidationError | None:
    """Post Inbound

     Append an inbound user message to the session bound to the caller's connection.

    Idempotent on ``body.event_id`` — replays return the original
    event id with ``deduped=True``.  Drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (ConnectorInboundRequest): Body for ``POST /v1/connectors/inbound``.

            Authenticated via ``ConnectorAuthDep`` so the connection_id is
            server-resolved from the bearer token — clients don't pick which
            connection their inbound lands on.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorInboundResponse | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorInboundRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorInboundResponse | HTTPValidationError]:
    """Post Inbound

     Append an inbound user message to the session bound to the caller's connection.

    Idempotent on ``body.event_id`` — replays return the original
    event id with ``deduped=True``.  Drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (ConnectorInboundRequest): Body for ``POST /v1/connectors/inbound``.

            Authenticated via ``ConnectorAuthDep`` so the connection_id is
            server-resolved from the bearer token — clients don't pick which
            connection their inbound lands on.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorInboundResponse | HTTPValidationError]
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
    body: ConnectorInboundRequest,
    authorization: None | str | Unset = UNSET,
) -> ConnectorInboundResponse | HTTPValidationError | None:
    """Post Inbound

     Append an inbound user message to the session bound to the caller's connection.

    Idempotent on ``body.event_id`` — replays return the original
    event id with ``deduped=True``.  Drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (ConnectorInboundRequest): Body for ``POST /v1/connectors/inbound``.

            Authenticated via ``ConnectorAuthDep`` so the connection_id is
            server-resolved from the bearer token — clients don't pick which
            connection their inbound lands on.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorInboundResponse | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
