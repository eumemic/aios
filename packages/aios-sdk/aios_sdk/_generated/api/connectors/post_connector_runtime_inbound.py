from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.body_post_connector_runtime_inbound import (
    BodyPostConnectorRuntimeInbound,
)
from ...models.connector_inbound_response import ConnectorInboundResponse
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: BodyPostConnectorRuntimeInbound,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/runtime/inbound",
    }

    _kwargs["files"] = body.to_multipart()

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
    body: BodyPostConnectorRuntimeInbound,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorInboundResponse | HTTPValidationError]:
    """Post Runtime Inbound

     Append an inbound user message to ``connection_id``'s session.

    The bearer authenticates the caller as one connector *type*;
    ``connection_id`` rides as a form field and must belong to that
    type.  When the bearer carries a ``connection_ids`` allowlist
    (#350), the form field must also be on the list — otherwise 403.
    Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (BodyPostConnectorRuntimeInbound):

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
    body: BodyPostConnectorRuntimeInbound,
    authorization: None | str | Unset = UNSET,
) -> ConnectorInboundResponse | HTTPValidationError | None:
    """Post Runtime Inbound

     Append an inbound user message to ``connection_id``'s session.

    The bearer authenticates the caller as one connector *type*;
    ``connection_id`` rides as a form field and must belong to that
    type.  When the bearer carries a ``connection_ids`` allowlist
    (#350), the form field must also be on the list — otherwise 403.
    Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (BodyPostConnectorRuntimeInbound):

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
    body: BodyPostConnectorRuntimeInbound,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorInboundResponse | HTTPValidationError]:
    """Post Runtime Inbound

     Append an inbound user message to ``connection_id``'s session.

    The bearer authenticates the caller as one connector *type*;
    ``connection_id`` rides as a form field and must belong to that
    type.  When the bearer carries a ``connection_ids`` allowlist
    (#350), the form field must also be on the list — otherwise 403.
    Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (BodyPostConnectorRuntimeInbound):

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
    body: BodyPostConnectorRuntimeInbound,
    authorization: None | str | Unset = UNSET,
) -> ConnectorInboundResponse | HTTPValidationError | None:
    """Post Runtime Inbound

     Append an inbound user message to ``connection_id``'s session.

    The bearer authenticates the caller as one connector *type*;
    ``connection_id`` rides as a form field and must belong to that
    type.  When the bearer carries a ``connection_ids`` allowlist
    (#350), the form field must also be on the list — otherwise 403.
    Idempotent on ``event_id``; drops surface as 4xx/5xx with
    a body explaining the reason (operator-config issue vs server
    error vs payload).

    Args:
        authorization (None | str | Unset):
        body (BodyPostConnectorRuntimeInbound):

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
