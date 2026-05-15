from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.mint_key_request import MintKeyRequest
from ...models.mint_key_response import MintKeyResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    target_id: str,
    *,
    body: MintKeyRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/accounts/{target_id}/keys".format(
            target_id=quote(str(target_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | MintKeyResponse | None:
    if response.status_code == 201:
        response_201 = MintKeyResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | MintKeyResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MintKeyRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MintKeyResponse]:
    """Mint Account Key

     Mint an additional API key on a caller-or-child account.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (MintKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MintKeyResponse]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MintKeyRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MintKeyResponse | None:
    """Mint Account Key

     Mint an additional API key on a caller-or-child account.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (MintKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MintKeyResponse
    """

    return sync_detailed(
        target_id=target_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MintKeyRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MintKeyResponse]:
    """Mint Account Key

     Mint an additional API key on a caller-or-child account.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (MintKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MintKeyResponse]
    """

    kwargs = _get_kwargs(
        target_id=target_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    target_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: MintKeyRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MintKeyResponse | None:
    """Mint Account Key

     Mint an additional API key on a caller-or-child account.

    Args:
        target_id (str):
        authorization (None | str | Unset):
        body (MintKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MintKeyResponse
    """

    return (
        await asyncio_detailed(
            target_id=target_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
