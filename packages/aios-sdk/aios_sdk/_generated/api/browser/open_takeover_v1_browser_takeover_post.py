from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.takeover_open_request import TakeoverOpenRequest
from ...models.takeover_open_response import TakeoverOpenResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: TakeoverOpenRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/browser/takeover",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | TakeoverOpenResponse | None:
    if response.status_code == 200:
        response_200 = TakeoverOpenResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | TakeoverOpenResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverOpenRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TakeoverOpenResponse]:
    """Open Takeover

     Open a takeover of the account's computer for one session's page.

    409 if a takeover is already in progress (the one-open-per-account
    invariant); 503 if the computer is unavailable.

    Args:
        authorization (None | str | Unset):
        body (TakeoverOpenRequest): Open a takeover of the account's computer for one agent
            session's page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TakeoverOpenResponse]
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
    body: TakeoverOpenRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TakeoverOpenResponse | None:
    """Open Takeover

     Open a takeover of the account's computer for one session's page.

    409 if a takeover is already in progress (the one-open-per-account
    invariant); 503 if the computer is unavailable.

    Args:
        authorization (None | str | Unset):
        body (TakeoverOpenRequest): Open a takeover of the account's computer for one agent
            session's page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TakeoverOpenResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TakeoverOpenRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | TakeoverOpenResponse]:
    """Open Takeover

     Open a takeover of the account's computer for one session's page.

    409 if a takeover is already in progress (the one-open-per-account
    invariant); 503 if the computer is unavailable.

    Args:
        authorization (None | str | Unset):
        body (TakeoverOpenRequest): Open a takeover of the account's computer for one agent
            session's page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | TakeoverOpenResponse]
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
    body: TakeoverOpenRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | TakeoverOpenResponse | None:
    """Open Takeover

     Open a takeover of the account's computer for one session's page.

    409 if a takeover is already in progress (the one-open-per-account
    invariant); 503 if the computer is unavailable.

    Args:
        authorization (None | str | Unset):
        body (TakeoverOpenRequest): Open a takeover of the account's computer for one agent
            session's page.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | TakeoverOpenResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
