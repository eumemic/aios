from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.signal_verify_request import SignalVerifyRequest
from ...models.signal_verify_response import SignalVerifyResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SignalVerifyRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/signal/verify",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SignalVerifyResponse | None:
    if response.status_code == 200:
        response_200 = SignalVerifyResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | SignalVerifyResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalVerifyRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SignalVerifyResponse]:
    """Post Signal Verify

     Submit the verification code received via SMS / voice.

    On success, signal-cli writes the new account to its on-disk
    ``accounts.json``.  The running signal connector picks it up
    without restart on the next ``verify_phone`` call (the daemon
    re-reads the file fresh; see ``daemon.py:_read_accounts_index``).

    Args:
        authorization (None | str | Unset):
        body (SignalVerifyRequest): Body for ``POST /v1/connectors/signal/verify``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalVerifyResponse]
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
    body: SignalVerifyRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SignalVerifyResponse | None:
    """Post Signal Verify

     Submit the verification code received via SMS / voice.

    On success, signal-cli writes the new account to its on-disk
    ``accounts.json``.  The running signal connector picks it up
    without restart on the next ``verify_phone`` call (the daemon
    re-reads the file fresh; see ``daemon.py:_read_accounts_index``).

    Args:
        authorization (None | str | Unset):
        body (SignalVerifyRequest): Body for ``POST /v1/connectors/signal/verify``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalVerifyResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalVerifyRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | SignalVerifyResponse]:
    """Post Signal Verify

     Submit the verification code received via SMS / voice.

    On success, signal-cli writes the new account to its on-disk
    ``accounts.json``.  The running signal connector picks it up
    without restart on the next ``verify_phone`` call (the daemon
    re-reads the file fresh; see ``daemon.py:_read_accounts_index``).

    Args:
        authorization (None | str | Unset):
        body (SignalVerifyRequest): Body for ``POST /v1/connectors/signal/verify``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalVerifyResponse]
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
    body: SignalVerifyRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | SignalVerifyResponse | None:
    """Post Signal Verify

     Submit the verification code received via SMS / voice.

    On success, signal-cli writes the new account to its on-disk
    ``accounts.json``.  The running signal connector picks it up
    without restart on the next ``verify_phone`` call (the daemon
    re-reads the file fresh; see ``daemon.py:_read_accounts_index``).

    Args:
        authorization (None | str | Unset):
        body (SignalVerifyRequest): Body for ``POST /v1/connectors/signal/verify``.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalVerifyResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
