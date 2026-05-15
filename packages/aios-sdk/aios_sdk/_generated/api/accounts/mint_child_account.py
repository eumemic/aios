from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.mint_account_request import MintAccountRequest
from ...models.mint_account_response import MintAccountResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: MintAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/accounts/children",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | MintAccountResponse | None:
    if response.status_code == 201:
        response_201 = MintAccountResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | MintAccountResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: MintAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MintAccountResponse]:
    """Mint Child

     Mint a direct child account under the caller and its first API key.

    Requires the caller's ``can_mint_children`` to be true. Returns the
    new account id, the first key's id, and the plaintext bearer (the
    only time that plaintext is recoverable).

    Args:
        authorization (None | str | Unset):
        body (MintAccountRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MintAccountResponse]
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
    body: MintAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MintAccountResponse | None:
    """Mint Child

     Mint a direct child account under the caller and its first API key.

    Requires the caller's ``can_mint_children`` to be true. Returns the
    new account id, the first key's id, and the plaintext bearer (the
    only time that plaintext is recoverable).

    Args:
        authorization (None | str | Unset):
        body (MintAccountRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MintAccountResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: MintAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | MintAccountResponse]:
    """Mint Child

     Mint a direct child account under the caller and its first API key.

    Requires the caller's ``can_mint_children`` to be true. Returns the
    new account id, the first key's id, and the plaintext bearer (the
    only time that plaintext is recoverable).

    Args:
        authorization (None | str | Unset):
        body (MintAccountRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | MintAccountResponse]
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
    body: MintAccountRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | MintAccountResponse | None:
    """Mint Child

     Mint a direct child account under the caller and its first API key.

    Requires the caller's ``can_mint_children`` to be true. Returns the
    new account id, the first key's id, and the plaintext bearer (the
    only time that plaintext is recoverable).

    Args:
        authorization (None | str | Unset):
        body (MintAccountRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | MintAccountResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
