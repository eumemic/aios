from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.whatsapp_confirm_pairing_request import WhatsappConfirmPairingRequest
from ...models.whatsapp_confirm_pairing_response import WhatsappConfirmPairingResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: WhatsappConfirmPairingRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/whatsapp/confirm-pairing",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | WhatsappConfirmPairingResponse | None:
    if response.status_code == 200:
        response_200 = WhatsappConfirmPairingResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | WhatsappConfirmPairingResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WhatsappConfirmPairingRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhatsappConfirmPairingResponse]:
    """Post Whatsapp Confirm Pairing

     Block until pairing terminates.  Long-poll: up to ~180 s server-side
    so a slow scan doesn't trip the HTTP timeout before the daemon
    itself reports timeout.

    Args:
        authorization (None | str | Unset):
        body (WhatsappConfirmPairingRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhatsappConfirmPairingResponse]
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
    body: WhatsappConfirmPairingRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhatsappConfirmPairingResponse | None:
    """Post Whatsapp Confirm Pairing

     Block until pairing terminates.  Long-poll: up to ~180 s server-side
    so a slow scan doesn't trip the HTTP timeout before the daemon
    itself reports timeout.

    Args:
        authorization (None | str | Unset):
        body (WhatsappConfirmPairingRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhatsappConfirmPairingResponse
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: WhatsappConfirmPairingRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | WhatsappConfirmPairingResponse]:
    """Post Whatsapp Confirm Pairing

     Block until pairing terminates.  Long-poll: up to ~180 s server-side
    so a slow scan doesn't trip the HTTP timeout before the daemon
    itself reports timeout.

    Args:
        authorization (None | str | Unset):
        body (WhatsappConfirmPairingRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | WhatsappConfirmPairingResponse]
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
    body: WhatsappConfirmPairingRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | WhatsappConfirmPairingResponse | None:
    """Post Whatsapp Confirm Pairing

     Block until pairing terminates.  Long-poll: up to ~180 s server-side
    so a slow scan doesn't trip the HTTP timeout before the daemon
    itself reports timeout.

    Args:
        authorization (None | str | Unset):
        body (WhatsappConfirmPairingRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | WhatsappConfirmPairingResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
