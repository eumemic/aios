from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.tools_schema_update import ToolsSchemaUpdate
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connector: str,
    *,
    body: ToolsSchemaUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/connectors/{connector}/tools_schema".format(
            connector=quote(str(connector), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = response.json()
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
) -> Response[Any | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connector: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolsSchemaUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Put Tools Schema

     Publish the runtime container's tool catalog for a connector type.

    The runtime container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and calls this once, replacing whatever was on the
    ``connectors.tools_schema`` row wholesale.  Operators don't
    hand-write the schema.

    Authorization: the runtime bearer's ``connector`` must match the
    path's ``connector``.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (ToolsSchemaUpdate): Body for ``PUT /v1/connectors/{connector}/tools_schema``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connector=connector,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connector: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolsSchemaUpdate,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Put Tools Schema

     Publish the runtime container's tool catalog for a connector type.

    The runtime container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and calls this once, replacing whatever was on the
    ``connectors.tools_schema`` row wholesale.  Operators don't
    hand-write the schema.

    Authorization: the runtime bearer's ``connector`` must match the
    path's ``connector``.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (ToolsSchemaUpdate): Body for ``PUT /v1/connectors/{connector}/tools_schema``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        connector=connector,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connector: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolsSchemaUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Put Tools Schema

     Publish the runtime container's tool catalog for a connector type.

    The runtime container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and calls this once, replacing whatever was on the
    ``connectors.tools_schema`` row wholesale.  Operators don't
    hand-write the schema.

    Authorization: the runtime bearer's ``connector`` must match the
    path's ``connector``.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (ToolsSchemaUpdate): Body for ``PUT /v1/connectors/{connector}/tools_schema``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connector=connector,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connector: str,
    *,
    client: AuthenticatedClient | Client,
    body: ToolsSchemaUpdate,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Put Tools Schema

     Publish the runtime container's tool catalog for a connector type.

    The runtime container is the source of truth for what tools it
    serves — it knows their names, parameter shapes, and docstrings.
    The SDK derives JSON Schemas from ``@tool``-decorated methods at
    startup and calls this once, replacing whatever was on the
    ``connectors.tools_schema`` row wholesale.  Operators don't
    hand-write the schema.

    Authorization: the runtime bearer's ``connector`` must match the
    path's ``connector``.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (ToolsSchemaUpdate): Body for ``PUT /v1/connectors/{connector}/tools_schema``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connector=connector,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
