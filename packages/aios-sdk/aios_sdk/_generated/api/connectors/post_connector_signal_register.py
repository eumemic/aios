from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.signal_register_request import SignalRegisterRequest
from ...models.signal_register_response import SignalRegisterResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SignalRegisterRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/signal/register",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SignalRegisterResponse | None:
    if response.status_code == 200:
        response_200 = SignalRegisterResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | SignalRegisterResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalRegisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SignalRegisterResponse]:
    """Post Signal Register

     Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.

    Args:
        authorization (None | str | Unset):
        body (SignalRegisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalRegisterResponse]
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
    body: SignalRegisterRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SignalRegisterResponse | None:
    """Post Signal Register

     Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.

    Args:
        authorization (None | str | Unset):
        body (SignalRegisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalRegisterResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalRegisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SignalRegisterResponse]:
    """Post Signal Register

     Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.

    Args:
        authorization (None | str | Unset):
        body (SignalRegisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalRegisterResponse]
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
    body: SignalRegisterRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SignalRegisterResponse | None:
    """Post Signal Register

     Initiate signal-cli ``register`` for ``account``.

    On captcha-required (common), returns 200 with the URL — solve in a
    browser, repost with ``captcha=<token>``.  On success: SMS (or voice
    call with ``voice=true``) carrying a 6-digit code for ``verify``.

    Args:
        authorization (None | str | Unset):
        body (SignalRegisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalRegisterResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
