from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_instance_v1_connectors_connector_instance_get_response_get_instance_v1_connectors_connector_instance_get import (
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet,
)
from ...models.http_validation_error import HTTPValidationError
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
        "url": "/v1/connectors/{connector}/{instance}".format(
            connector=quote(str(connector), safe=""),
            instance=quote(str(instance), safe=""),
        ),
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
    | None
):
    if response.status_code == 200:
        response_200 = GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet.from_dict(
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
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
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
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
]:
    """Get Instance

     Snapshot a single ``(connector, instance)`` pair.

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet | HTTPValidationError]
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
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
    | None
):
    """Get Instance

     Snapshot a single ``(connector, instance)`` pair.

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet | HTTPValidationError
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
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
]:
    """Get Instance

     Snapshot a single ``(connector, instance)`` pair.

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet | HTTPValidationError]
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
    GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet
    | HTTPValidationError
    | None
):
    """Get Instance

     Snapshot a single ``(connector, instance)`` pair.

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        GetInstanceV1ConnectorsConnectorInstanceGetResponseGetInstanceV1ConnectorsConnectorInstanceGet | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connector=connector,
            instance=instance,
            client=client,
            authorization=authorization,
        )
    ).parsed
