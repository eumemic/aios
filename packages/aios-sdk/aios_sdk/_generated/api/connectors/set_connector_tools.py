from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_set_tools import ConnectionSetTools
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectionSetTools,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/connectors/tools",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Connection.from_dict(response.json())

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
) -> Response[Connection | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionSetTools,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    """Set Tools

     Publish the connector's tool schemas onto its own connection.

    The connector container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and POSTs them here, replacing whatever was on the
    connection wholesale.  Operators don't hand-write ``tools.json``.

    Authorization: the bearer token resolves to one ``connection_id``;
    a connector can only publish tools for its own connection.  This
    is the connector-scoped twin of operator-scoped
    ``PUT /v1/connections/{id}/tools``.

    Args:
        authorization (None | str | Unset):
        body (ConnectionSetTools): Request body for ``PUT /v1/connections/{id}/tools`` (#301).

            Replaces the connection's tools array wholesale.  Each entry must
            be ``type="custom"`` — see :func:`_validate_connection_tools`.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
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
    body: ConnectionSetTools,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    """Set Tools

     Publish the connector's tool schemas onto its own connection.

    The connector container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and POSTs them here, replacing whatever was on the
    connection wholesale.  Operators don't hand-write ``tools.json``.

    Authorization: the bearer token resolves to one ``connection_id``;
    a connector can only publish tools for its own connection.  This
    is the connector-scoped twin of operator-scoped
    ``PUT /v1/connections/{id}/tools``.

    Args:
        authorization (None | str | Unset):
        body (ConnectionSetTools): Request body for ``PUT /v1/connections/{id}/tools`` (#301).

            Replaces the connection's tools array wholesale.  Each entry must
            be ``type="custom"`` — see :func:`_validate_connection_tools`.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionSetTools,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    """Set Tools

     Publish the connector's tool schemas onto its own connection.

    The connector container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and POSTs them here, replacing whatever was on the
    connection wholesale.  Operators don't hand-write ``tools.json``.

    Authorization: the bearer token resolves to one ``connection_id``;
    a connector can only publish tools for its own connection.  This
    is the connector-scoped twin of operator-scoped
    ``PUT /v1/connections/{id}/tools``.

    Args:
        authorization (None | str | Unset):
        body (ConnectionSetTools): Request body for ``PUT /v1/connections/{id}/tools`` (#301).

            Replaces the connection's tools array wholesale.  Each entry must
            be ``type="custom"`` — see :func:`_validate_connection_tools`.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
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
    body: ConnectionSetTools,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    """Set Tools

     Publish the connector's tool schemas onto its own connection.

    The connector container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and POSTs them here, replacing whatever was on the
    connection wholesale.  Operators don't hand-write ``tools.json``.

    Authorization: the bearer token resolves to one ``connection_id``;
    a connector can only publish tools for its own connection.  This
    is the connector-scoped twin of operator-scoped
    ``PUT /v1/connections/{id}/tools``.

    Args:
        authorization (None | str | Unset):
        body (ConnectionSetTools): Request body for ``PUT /v1/connections/{id}/tools`` (#301).

            Replaces the connection's tools array wholesale.  Each entry must
            be ``type="custom"`` — see :func:`_validate_connection_tools`.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
