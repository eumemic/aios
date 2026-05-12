from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_token_issue import ConnectorTokenIssue
from ...models.connector_token_issued import ConnectorTokenIssued
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectorTokenIssue,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connector-tokens",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorTokenIssued | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = ConnectorTokenIssued.from_dict(response.json())

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
) -> Response[ConnectorTokenIssued | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorTokenIssue,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorTokenIssued | HTTPValidationError]:
    """Issue

     Mint a new bearer token for ``body.connection_id``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.  Subsequent ``GET``
    on this resource returns the read view without plaintext.

    Args:
        authorization (None | str | Unset):
        body (ConnectorTokenIssue): Request body for ``POST /v1/connector-tokens``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorTokenIssued | HTTPValidationError]
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
    body: ConnectorTokenIssue,
    authorization: None | str | Unset = UNSET,
) -> ConnectorTokenIssued | HTTPValidationError | None:
    """Issue

     Mint a new bearer token for ``body.connection_id``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.  Subsequent ``GET``
    on this resource returns the read view without plaintext.

    Args:
        authorization (None | str | Unset):
        body (ConnectorTokenIssue): Request body for ``POST /v1/connector-tokens``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorTokenIssued | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorTokenIssue,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorTokenIssued | HTTPValidationError]:
    """Issue

     Mint a new bearer token for ``body.connection_id``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.  Subsequent ``GET``
    on this resource returns the read view without plaintext.

    Args:
        authorization (None | str | Unset):
        body (ConnectorTokenIssue): Request body for ``POST /v1/connector-tokens``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorTokenIssued | HTTPValidationError]
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
    body: ConnectorTokenIssue,
    authorization: None | str | Unset = UNSET,
) -> ConnectorTokenIssued | HTTPValidationError | None:
    """Issue

     Mint a new bearer token for ``body.connection_id``.

    The plaintext is included in the response and CANNOT be recovered
    later — operators must save it at issue time.  Subsequent ``GET``
    on this resource returns the read view without plaintext.

    Args:
        authorization (None | str | Unset):
        body (ConnectorTokenIssue): Request body for ``POST /v1/connector-tokens``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorTokenIssued | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
