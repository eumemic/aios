from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.call_v1_connectors_connector_instance_call_post_response_call_v1_connectors_connector_instance_call_post import (
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost,
)
from ...models.connector_call_body import ConnectorCallBody
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connector: str,
    instance: str,
    *,
    body: ConnectorCallBody,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/{connector}/{instance}/call".format(
            connector=quote(str(connector), safe=""),
            instance=quote(str(instance), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> (
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
    | HTTPValidationError
    | None
):
    if response.status_code == 200:
        response_200 = CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost.from_dict(
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
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
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
    body: ConnectorCallBody,
    authorization: None | str | Unset = UNSET,
) -> Response[
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
    | HTTPValidationError
]:
    """Call

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):
        body (ConnectorCallBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connector=connector,
        instance=instance,
        body=body,
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
    body: ConnectorCallBody,
    authorization: None | str | Unset = UNSET,
) -> (
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
    | HTTPValidationError
    | None
):
    """Call

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):
        body (ConnectorCallBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost | HTTPValidationError
    """

    return sync_detailed(
        connector=connector,
        instance=instance,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorCallBody,
    authorization: None | str | Unset = UNSET,
) -> Response[
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
    | HTTPValidationError
]:
    """Call

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):
        body (ConnectorCallBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connector=connector,
        instance=instance,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connector: str,
    instance: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorCallBody,
    authorization: None | str | Unset = UNSET,
) -> (
    CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost
    | HTTPValidationError
    | None
):
    """Call

    Args:
        connector (str):
        instance (str):
        authorization (None | str | Unset):
        body (ConnectorCallBody):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CallV1ConnectorsConnectorInstanceCallPostResponseCallV1ConnectorsConnectorInstanceCallPost | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connector=connector,
            instance=instance,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
