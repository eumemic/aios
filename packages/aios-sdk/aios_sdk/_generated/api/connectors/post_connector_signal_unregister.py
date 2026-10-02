from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.signal_unregister_request import SignalUnregisterRequest
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: SignalUnregisterRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/signal/unregister",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

    if response.status_code == 422:
        response_422 = HTTPValidationError.from_dict(response.json())

        return response_422

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalUnregisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Signal Unregister

     Release the number from this connector's signal-cli device.

    Equivalent to ``signal-cli -a <phone> unregister`` run by the
    connector against its own config dir.  Detaching an aios connection
    only changes the aios binding — Signal keeps the number registered
    to the device until this runs, so re-registering it elsewhere fails.
    Detach (or archive) the connection first: once unregistered, the
    connector can no longer send or receive on the number.

    That precondition is ENFORCED, not just documented: unregistering is
    irreversible (the number must be re-registered and re-verified), so if
    any of the caller's non-archived signal connections for this number
    (matched on digits only) still has an active binding, the request is
    refused with 409 ``conflict`` and no management call is dispatched.
    A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
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
    body: SignalUnregisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Signal Unregister

     Release the number from this connector's signal-cli device.

    Equivalent to ``signal-cli -a <phone> unregister`` run by the
    connector against its own config dir.  Detaching an aios connection
    only changes the aios binding — Signal keeps the number registered
    to the device until this runs, so re-registering it elsewhere fails.
    Detach (or archive) the connection first: once unregistered, the
    connector can no longer send or receive on the number.

    That precondition is ENFORCED, not just documented: unregistering is
    irreversible (the number must be re-registered and re-verified), so if
    any of the caller's non-archived signal connections for this number
    (matched on digits only) still has an active binding, the request is
    refused with 409 ``conflict`` and no management call is dispatched.
    A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SignalUnregisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    """Post Signal Unregister

     Release the number from this connector's signal-cli device.

    Equivalent to ``signal-cli -a <phone> unregister`` run by the
    connector against its own config dir.  Detaching an aios connection
    only changes the aios binding — Signal keeps the number registered
    to the device until this runs, so re-registering it elsewhere fails.
    Detach (or archive) the connection first: once unregistered, the
    connector can no longer send or receive on the number.

    That precondition is ENFORCED, not just documented: unregistering is
    irreversible (the number must be re-registered and re-verified), so if
    any of the caller's non-archived signal connections for this number
    (matched on digits only) still has an active binding, the request is
    refused with 409 ``conflict`` and no management call is dispatched.
    A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
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
    body: SignalUnregisterRequest,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    """Post Signal Unregister

     Release the number from this connector's signal-cli device.

    Equivalent to ``signal-cli -a <phone> unregister`` run by the
    connector against its own config dir.  Detaching an aios connection
    only changes the aios binding — Signal keeps the number registered
    to the device until this runs, so re-registering it elsewhere fails.
    Detach (or archive) the connection first: once unregistered, the
    connector can no longer send or receive on the number.

    That precondition is ENFORCED, not just documented: unregistering is
    irreversible (the number must be re-registered and re-verified), so if
    any of the caller's non-archived signal connections for this number
    (matched on digits only) still has an active binding, the request is
    refused with 409 ``conflict`` and no management call is dispatched.
    A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
