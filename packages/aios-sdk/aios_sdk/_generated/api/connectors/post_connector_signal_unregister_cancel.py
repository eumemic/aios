from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.signal_unregister_cancel_response import SignalUnregisterCancelResponse
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
        "url": "/v1/connectors/signal/unregister/cancel",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | SignalUnregisterCancelResponse | None:
    if response.status_code == 200:
        response_200 = SignalUnregisterCancelResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | SignalUnregisterCancelResponse]:
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
) -> Response[HTTPValidationError | SignalUnregisterCancelResponse]:
    """Post Signal Unregister Cancel

     Operator escape hatch: stop a pending unregister from blocking binds.

    While an ``unregister`` call for a number is non-terminal, every path
    that would make the number routable again (attach, configure per_chat,
    reparent in, bind-chat) is refused with 409
    ``number_unregister_pending``.  That holds past the call's wall-clock
    expiry, because a connector that already received the call can still
    execute it (#2322 F3).  Normally the connector clears it by resolving
    the call, and a non-terminal unregister is redelivered on every
    connector reconnect, so a crash before execution is retried.

    Use this when that cannot happen: the connector is gone for good, or
    it ran the unregister but its result was lost.  Every still-pending
    unregister call for the number (matched on ASCII digits 0-9) is marked
    ``failed`` with code ``cancelled_by_operator``.  The connector's late
    result for a cancelled call is ignored.  CAUTION: this does not recall
    a call the connector has already received.  If the connector is still
    alive, it may still unregister the number after you re-attach.  Only
    cancel when you know the connector will not run the call.

    Returns the ids of the calls that were cancelled (empty if none were
    pending).  A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalUnregisterCancelResponse]
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
) -> HTTPValidationError | SignalUnregisterCancelResponse | None:
    """Post Signal Unregister Cancel

     Operator escape hatch: stop a pending unregister from blocking binds.

    While an ``unregister`` call for a number is non-terminal, every path
    that would make the number routable again (attach, configure per_chat,
    reparent in, bind-chat) is refused with 409
    ``number_unregister_pending``.  That holds past the call's wall-clock
    expiry, because a connector that already received the call can still
    execute it (#2322 F3).  Normally the connector clears it by resolving
    the call, and a non-terminal unregister is redelivered on every
    connector reconnect, so a crash before execution is retried.

    Use this when that cannot happen: the connector is gone for good, or
    it ran the unregister but its result was lost.  Every still-pending
    unregister call for the number (matched on ASCII digits 0-9) is marked
    ``failed`` with code ``cancelled_by_operator``.  The connector's late
    result for a cancelled call is ignored.  CAUTION: this does not recall
    a call the connector has already received.  If the connector is still
    alive, it may still unregister the number after you re-attach.  Only
    cancel when you know the connector will not run the call.

    Returns the ids of the calls that were cancelled (empty if none were
    pending).  A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalUnregisterCancelResponse
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
) -> Response[HTTPValidationError | SignalUnregisterCancelResponse]:
    """Post Signal Unregister Cancel

     Operator escape hatch: stop a pending unregister from blocking binds.

    While an ``unregister`` call for a number is non-terminal, every path
    that would make the number routable again (attach, configure per_chat,
    reparent in, bind-chat) is refused with 409
    ``number_unregister_pending``.  That holds past the call's wall-clock
    expiry, because a connector that already received the call can still
    execute it (#2322 F3).  Normally the connector clears it by resolving
    the call, and a non-terminal unregister is redelivered on every
    connector reconnect, so a crash before execution is retried.

    Use this when that cannot happen: the connector is gone for good, or
    it ran the unregister but its result was lost.  Every still-pending
    unregister call for the number (matched on ASCII digits 0-9) is marked
    ``failed`` with code ``cancelled_by_operator``.  The connector's late
    result for a cancelled call is ignored.  CAUTION: this does not recall
    a call the connector has already received.  If the connector is still
    alive, it may still unregister the number after you re-attach.  Only
    cancel when you know the connector will not run the call.

    Returns the ids of the calls that were cancelled (empty if none were
    pending).  A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | SignalUnregisterCancelResponse]
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
) -> HTTPValidationError | SignalUnregisterCancelResponse | None:
    """Post Signal Unregister Cancel

     Operator escape hatch: stop a pending unregister from blocking binds.

    While an ``unregister`` call for a number is non-terminal, every path
    that would make the number routable again (attach, configure per_chat,
    reparent in, bind-chat) is refused with 409
    ``number_unregister_pending``.  That holds past the call's wall-clock
    expiry, because a connector that already received the call can still
    execute it (#2322 F3).  Normally the connector clears it by resolving
    the call, and a non-terminal unregister is redelivered on every
    connector reconnect, so a crash before execution is retried.

    Use this when that cannot happen: the connector is gone for good, or
    it ran the unregister but its result was lost.  Every still-pending
    unregister call for the number (matched on ASCII digits 0-9) is marked
    ``failed`` with code ``cancelled_by_operator``.  The connector's late
    result for a cancelled call is ignored.  CAUTION: this does not recall
    a call the connector has already received.  If the connector is still
    alive, it may still unregister the number after you re-attach.  Only
    cancel when you know the connector will not run the call.

    Returns the ids of the calls that were cancelled (empty if none were
    pending).  A malformed number with no digits is a 422.

    Args:
        authorization (None | str | Unset):
        body (SignalUnregisterRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | SignalUnregisterCancelResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
