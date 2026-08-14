from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.get_connection_discovery_v1_connectors_connections_get_arm_type_0 import (
    GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0,
)
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    arm: GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0
    | None
    | Unset = UNSET,
    after_change_seq: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    params: dict[str, Any] = {}

    json_arm: None | str | Unset
    if isinstance(arm, Unset):
        json_arm = UNSET
    elif isinstance(arm, GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0):
        json_arm = arm.value
    else:
        json_arm = arm
    params["arm"] = json_arm

    json_after_change_seq: int | None | Unset
    if isinstance(after_change_seq, Unset):
        json_after_change_seq = UNSET
    else:
        json_after_change_seq = after_change_seq
    params["after_change_seq"] = json_after_change_seq

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connectors/connections",
        "params": params,
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = response.json()
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
    arm: GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0
    | None
    | Unset = UNSET,
    after_change_seq: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\",
         \"connection_id\": \"...\",
         \"external_account_id\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    When the bearer carries a ``connection_ids`` allowlist (#350), the
    backfill and tail both filter to that set — out-of-scope IDs are
    silently omitted (not 403'd) so the runtime container's discovery
    loop just doesn't see them.

    Args:
        arm (GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0 | None | Unset):
        after_change_seq (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        arm=arm,
        after_change_seq=after_change_seq,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    arm: GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0
    | None
    | Unset = UNSET,
    after_change_seq: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\",
         \"connection_id\": \"...\",
         \"external_account_id\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    When the bearer carries a ``connection_ids`` allowlist (#350), the
    backfill and tail both filter to that set — out-of-scope IDs are
    silently omitted (not 403'd) so the runtime container's discovery
    loop just doesn't see them.

    Args:
        arm (GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0 | None | Unset):
        after_change_seq (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        arm=arm,
        after_change_seq=after_change_seq,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    arm: GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0
    | None
    | Unset = UNSET,
    after_change_seq: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Response[Any | HTTPValidationError]:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\",
         \"connection_id\": \"...\",
         \"external_account_id\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    When the bearer carries a ``connection_ids`` allowlist (#350), the
    backfill and tail both filter to that set — out-of-scope IDs are
    silently omitted (not 403'd) so the runtime container's discovery
    loop just doesn't see them.

    Args:
        arm (GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0 | None | Unset):
        after_change_seq (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        arm=arm,
        after_change_seq=after_change_seq,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    arm: GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0
    | None
    | Unset = UNSET,
    after_change_seq: int | None | Unset = UNSET,
    authorization: None | str | Unset = UNSET,
) -> Any | HTTPValidationError | None:
    r"""Get Connection Discovery

     SSE stream of ``added`` / ``removed`` connection events for the
    runtime container's connector type (#328 PR 5).

    Backfills every active connection of the caller's ``connector``
    type at subscribe time as ``added`` events, then tails the
    ``connections_<connector>`` NOTIFY channel.  Each event is keyed
    ``connection`` with a JSON body shaped::

        {\"event\": \"added\" | \"removed\",
         \"connection_id\": \"...\",
         \"external_account_id\": \"...\"}

    The runtime container subscribes once per ``connector`` type and
    fans out to per-connection workers on ``added``; tears them down
    on ``removed``.

    When the bearer carries a ``connection_ids`` allowlist (#350), the
    backfill and tail both filter to that set — out-of-scope IDs are
    silently omitted (not 403'd) so the runtime container's discovery
    loop just doesn't see them.

    Args:
        arm (GetConnectionDiscoveryV1ConnectorsConnectionsGetArmType0 | None | Unset):
        after_change_seq (int | None | Unset):
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            arm=arm,
            after_change_seq=after_change_seq,
            authorization=authorization,
        )
    ).parsed
