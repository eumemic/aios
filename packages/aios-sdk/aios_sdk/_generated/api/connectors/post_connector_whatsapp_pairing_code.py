from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.whatsapp_pairing_code_request import WhatsappPairingCodeRequest
from ...models.whatsapp_pairing_code_response import WhatsappPairingCodeResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: WhatsappPairingCodeRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/whatsapp/pairing-code",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WhatsappPairingCodeResponse | None:
    if response.status_code == 200:
        response_200 = WhatsappPairingCodeResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | WhatsappPairingCodeResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WhatsappPairingCodeRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhatsappPairingCodeResponse]:
    """Post Whatsapp Pairing Code

     Return the QR code currently live for the in-flight pairing
    attempt.  whatsmeow rotates the code ~every 20 s over the ~100 s
    attempt; ``/start-pairing`` surfaces only the first.  Operators poll
    this every few seconds and re-render when ``rotation_seq`` changes so
    each rotation is scannable, not just the first window.  404s when no
    attempt is live (none started, or already terminated).

    Args:
        authorization (None | str | Unset):
        body (WhatsappPairingCodeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhatsappPairingCodeResponse]
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
    body: WhatsappPairingCodeRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhatsappPairingCodeResponse | None:
    """Post Whatsapp Pairing Code

     Return the QR code currently live for the in-flight pairing
    attempt.  whatsmeow rotates the code ~every 20 s over the ~100 s
    attempt; ``/start-pairing`` surfaces only the first.  Operators poll
    this every few seconds and re-render when ``rotation_seq`` changes so
    each rotation is scannable, not just the first window.  404s when no
    attempt is live (none started, or already terminated).

    Args:
        authorization (None | str | Unset):
        body (WhatsappPairingCodeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhatsappPairingCodeResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WhatsappPairingCodeRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhatsappPairingCodeResponse]:
    """Post Whatsapp Pairing Code

     Return the QR code currently live for the in-flight pairing
    attempt.  whatsmeow rotates the code ~every 20 s over the ~100 s
    attempt; ``/start-pairing`` surfaces only the first.  Operators poll
    this every few seconds and re-render when ``rotation_seq`` changes so
    each rotation is scannable, not just the first window.  404s when no
    attempt is live (none started, or already terminated).

    Args:
        authorization (None | str | Unset):
        body (WhatsappPairingCodeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhatsappPairingCodeResponse]
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
    body: WhatsappPairingCodeRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhatsappPairingCodeResponse | None:
    """Post Whatsapp Pairing Code

     Return the QR code currently live for the in-flight pairing
    attempt.  whatsmeow rotates the code ~every 20 s over the ~100 s
    attempt; ``/start-pairing`` surfaces only the first.  Operators poll
    this every few seconds and re-render when ``rotation_seq`` changes so
    each rotation is scannable, not just the first window.  404s when no
    attempt is live (none started, or already terminated).

    Args:
        authorization (None | str | Unset):
        body (WhatsappPairingCodeRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhatsappPairingCodeResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
