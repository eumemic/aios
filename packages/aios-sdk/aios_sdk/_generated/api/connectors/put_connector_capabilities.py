from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.capabilities_update import CapabilitiesUpdate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connector: str,
    *,
    body: CapabilitiesUpdate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/connectors/{connector}/capabilities".format(
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
    body: CapabilitiesUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Put Capabilities

     Publish the runtime container's typed capability descriptor for a
    connector type.

    A sibling to :func:`put_tools_schema` (same root-only publication path),
    kept on its own route so capability churn is decoupled from a full
    ``tools_schema`` republish.  The runtime container is the source of truth
    for what richer renderings it supports; it publishes the descriptor at
    startup, replacing whatever was on the ``connectors.capabilities`` row
    wholesale.

    Authorization: the runtime bearer's ``connector`` must match the path's
    ``connector``; publication itself is root-only (enforced in the service
    layer — connectors are root-owned, the same cross-tenant rationale as
    ``tools_schema``).  Capabilities declare NO authority: they constrain
    RENDERING, never what any principal may invoke.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (CapabilitiesUpdate): Body for ``PUT /v1/connectors/{connector}/capabilities``.

            A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
            is decoupled from a full ``tools_schema`` republish and the shipped
            ``tools_schema`` body contract stays untouched.

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
    body: CapabilitiesUpdate,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Put Capabilities

     Publish the runtime container's typed capability descriptor for a
    connector type.

    A sibling to :func:`put_tools_schema` (same root-only publication path),
    kept on its own route so capability churn is decoupled from a full
    ``tools_schema`` republish.  The runtime container is the source of truth
    for what richer renderings it supports; it publishes the descriptor at
    startup, replacing whatever was on the ``connectors.capabilities`` row
    wholesale.

    Authorization: the runtime bearer's ``connector`` must match the path's
    ``connector``; publication itself is root-only (enforced in the service
    layer — connectors are root-owned, the same cross-tenant rationale as
    ``tools_schema``).  Capabilities declare NO authority: they constrain
    RENDERING, never what any principal may invoke.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (CapabilitiesUpdate): Body for ``PUT /v1/connectors/{connector}/capabilities``.

            A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
            is decoupled from a full ``tools_schema`` republish and the shipped
            ``tools_schema`` body contract stays untouched.

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
    body: CapabilitiesUpdate,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Put Capabilities

     Publish the runtime container's typed capability descriptor for a
    connector type.

    A sibling to :func:`put_tools_schema` (same root-only publication path),
    kept on its own route so capability churn is decoupled from a full
    ``tools_schema`` republish.  The runtime container is the source of truth
    for what richer renderings it supports; it publishes the descriptor at
    startup, replacing whatever was on the ``connectors.capabilities`` row
    wholesale.

    Authorization: the runtime bearer's ``connector`` must match the path's
    ``connector``; publication itself is root-only (enforced in the service
    layer — connectors are root-owned, the same cross-tenant rationale as
    ``tools_schema``).  Capabilities declare NO authority: they constrain
    RENDERING, never what any principal may invoke.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (CapabilitiesUpdate): Body for ``PUT /v1/connectors/{connector}/capabilities``.

            A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
            is decoupled from a full ``tools_schema`` republish and the shipped
            ``tools_schema`` body contract stays untouched.

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
    body: CapabilitiesUpdate,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Put Capabilities

     Publish the runtime container's typed capability descriptor for a
    connector type.

    A sibling to :func:`put_tools_schema` (same root-only publication path),
    kept on its own route so capability churn is decoupled from a full
    ``tools_schema`` republish.  The runtime container is the source of truth
    for what richer renderings it supports; it publishes the descriptor at
    startup, replacing whatever was on the ``connectors.capabilities`` row
    wholesale.

    Authorization: the runtime bearer's ``connector`` must match the path's
    ``connector``; publication itself is root-only (enforced in the service
    layer — connectors are root-owned, the same cross-tenant rationale as
    ``tools_schema``).  Capabilities declare NO authority: they constrain
    RENDERING, never what any principal may invoke.

    Args:
        connector (str):
        authorization (None | str | Unset):
        body (CapabilitiesUpdate): Body for ``PUT /v1/connectors/{connector}/capabilities``.

            A sibling to :class:`ToolsSchemaUpdate` — kept separate so capability churn
            is decoupled from a full ``tools_schema`` republish and the shipped
            ``tools_schema`` body contract stays untouched.

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
