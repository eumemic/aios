from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_tools_v1_connectors_connector_instance_tools_get_response_list_tools_v1_connectors_connector_instance_tools_get import (
    ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet,
)
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connector: str,
    instance: str,
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connectors/{connector}/{instance}/tools".format(
            connector=quote(str(connector), safe=""),
            instance=quote(str(instance), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
    | None
):
    if response.status_code == 200:
        response_200 = ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet.from_dict(
            response.json()
        )

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
) -> Response[
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
]:
    """List Tools

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet]
    """

    kwargs = _get_kwargs(
        connector=connector,
        instance=instance,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
    | None
):
    """List Tools

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
    """

    return sync_detailed(
        connector=connector,
        instance=instance,
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
]:
    """List Tools

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet]
    """

    kwargs = _get_kwargs(
        connector=connector,
        instance=instance,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> (
    HTTPValidationError
    | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
    | None
):
    """List Tools

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListToolsV1ConnectorsConnectorInstanceToolsGetResponseListToolsV1ConnectorsConnectorInstanceToolsGet
    """

    return (
        await asyncio_detailed(
            connector=connector,
            instance=instance,
            client=client,
            authorization=authorization,
        )
    ).parsed
