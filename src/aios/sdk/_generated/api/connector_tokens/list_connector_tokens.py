from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.list_response_connector_token import ListResponseConnectorToken
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    connection_id: str,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    params["connection_id"] = connection_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connector-tokens",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | ListResponseConnectorToken | None:
    if response.status_code == 200:
        response_200 = ListResponseConnectorToken.from_dict(response.json())

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
) -> Response[HTTPValidationError | ListResponseConnectorToken]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    connection_id: str,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseConnectorToken]:
    """List

     All tokens (revoked included) for ``connection_id``, newest first.

    Revoked tokens stay in the listing for audit; clients filter by
    ``revoked_at IS NULL`` if they only want live tokens.

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseConnectorToken]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    connection_id: str,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseConnectorToken | None:
    """List

     All tokens (revoked included) for ``connection_id``, newest first.

    Revoked tokens stay in the listing for audit; clients filter by
    ``revoked_at IS NULL`` if they only want live tokens.

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseConnectorToken
    """

    return sync_detailed(
        client=client,
        connection_id=connection_id,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    connection_id: str,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | ListResponseConnectorToken]:
    """List

     All tokens (revoked included) for ``connection_id``, newest first.

    Revoked tokens stay in the listing for audit; clients filter by
    ``revoked_at IS NULL`` if they only want live tokens.

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | ListResponseConnectorToken]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    connection_id: str,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | ListResponseConnectorToken | None:
    """List

     All tokens (revoked included) for ``connection_id``, newest first.

    Revoked tokens stay in the listing for audit; clients filter by
    ``revoked_at IS NULL`` if they only want live tokens.

    Args:
        connection_id (str):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | ListResponseConnectorToken
    """

    return (
        await asyncio_detailed(
            client=client,
            connection_id=connection_id,
            authorization=authorization,
        )
    ).parsed
