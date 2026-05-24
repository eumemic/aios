from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_reparent import ConnectionReparent
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connection_id: str,
    *,
    body: ConnectionReparent,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connections/{connection_id}/reparent".format(
            connection_id=quote(str(connection_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = Connection.from_dict(response.json())

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
) -> Response[Connection | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionReparent,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Reparent

     Transfer a connection to a different account. Root operator only.

    Moves ``connection.account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    daemon state (signal-cli's ``account.dat``, whatsmeow's
    ``sqlstore.db``, telegram webhook config) carries over without
    recreation. The per-account partial unique index on
    ``(account_id, connector, external_account_id) WHERE archived_at
    IS NULL`` enforces no-collision at the destination automatically;
    a colliding destination returns 409.

    Authorization (v1): root operator only — the caller's account must
    have ``parent_account_id IS NULL``. Multi-tenant consent semantics
    (\"both source and destination owners must approve\") are deferred
    to v2; v1 is the operator-only escape hatch that unblocks the
    jarbot v2 ``ExternalIdentity`` transfer flow.

    **Daemon-cache caveat (v1)**: this is a database-only reparent.
    Connector daemons cache ``account_id`` in memory at attach time
    and do NOT receive a rebind event. Restart the connector container
    after reparent — the in-memory cache is otherwise stale until the
    next restart.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionReparent): Request body for ``POST /v1/connections/{id}/reparent``.

            Moves the connection's ``account_id`` to ``destination_account_id``
            atomically, preserving ``connection.id`` so dependent connector
            state (signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``,
            telegram webhook config) carries over without recreation. v1
            auth: root operator only.

            Length bounds match the ULID-shaped ``account_id`` format used
            elsewhere on the wire (1..64 chars covers ``acc_<ULID>``).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionReparent,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Reparent

     Transfer a connection to a different account. Root operator only.

    Moves ``connection.account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    daemon state (signal-cli's ``account.dat``, whatsmeow's
    ``sqlstore.db``, telegram webhook config) carries over without
    recreation. The per-account partial unique index on
    ``(account_id, connector, external_account_id) WHERE archived_at
    IS NULL`` enforces no-collision at the destination automatically;
    a colliding destination returns 409.

    Authorization (v1): root operator only — the caller's account must
    have ``parent_account_id IS NULL``. Multi-tenant consent semantics
    (\"both source and destination owners must approve\") are deferred
    to v2; v1 is the operator-only escape hatch that unblocks the
    jarbot v2 ``ExternalIdentity`` transfer flow.

    **Daemon-cache caveat (v1)**: this is a database-only reparent.
    Connector daemons cache ``account_id`` in memory at attach time
    and do NOT receive a rebind event. Restart the connector container
    after reparent — the in-memory cache is otherwise stale until the
    next restart.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionReparent): Request body for ``POST /v1/connections/{id}/reparent``.

            Moves the connection's ``account_id`` to ``destination_account_id``
            atomically, preserving ``connection.id`` so dependent connector
            state (signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``,
            telegram webhook config) carries over without recreation. v1
            auth: root operator only.

            Length bounds match the ULID-shaped ``account_id`` format used
            elsewhere on the wire (1..64 chars covers ``acc_<ULID>``).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return sync_detailed(
        connection_id=connection_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionReparent,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Reparent

     Transfer a connection to a different account. Root operator only.

    Moves ``connection.account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    daemon state (signal-cli's ``account.dat``, whatsmeow's
    ``sqlstore.db``, telegram webhook config) carries over without
    recreation. The per-account partial unique index on
    ``(account_id, connector, external_account_id) WHERE archived_at
    IS NULL`` enforces no-collision at the destination automatically;
    a colliding destination returns 409.

    Authorization (v1): root operator only — the caller's account must
    have ``parent_account_id IS NULL``. Multi-tenant consent semantics
    (\"both source and destination owners must approve\") are deferred
    to v2; v1 is the operator-only escape hatch that unblocks the
    jarbot v2 ``ExternalIdentity`` transfer flow.

    **Daemon-cache caveat (v1)**: this is a database-only reparent.
    Connector daemons cache ``account_id`` in memory at attach time
    and do NOT receive a rebind event. Restart the connector container
    after reparent — the in-memory cache is otherwise stale until the
    next restart.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionReparent): Request body for ``POST /v1/connections/{id}/reparent``.

            Moves the connection's ``account_id`` to ``destination_account_id``
            atomically, preserving ``connection.id`` so dependent connector
            state (signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``,
            telegram webhook config) carries over without recreation. v1
            auth: root operator only.

            Length bounds match the ULID-shaped ``account_id`` format used
            elsewhere on the wire (1..64 chars covers ``acc_<ULID>``).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        connection_id=connection_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connection_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionReparent,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Reparent

     Transfer a connection to a different account. Root operator only.

    Moves ``connection.account_id`` to ``destination_account_id``
    atomically, preserving ``connection.id`` so dependent connector
    daemon state (signal-cli's ``account.dat``, whatsmeow's
    ``sqlstore.db``, telegram webhook config) carries over without
    recreation. The per-account partial unique index on
    ``(account_id, connector, external_account_id) WHERE archived_at
    IS NULL`` enforces no-collision at the destination automatically;
    a colliding destination returns 409.

    Authorization (v1): root operator only — the caller's account must
    have ``parent_account_id IS NULL``. Multi-tenant consent semantics
    (\"both source and destination owners must approve\") are deferred
    to v2; v1 is the operator-only escape hatch that unblocks the
    jarbot v2 ``ExternalIdentity`` transfer flow.

    **Daemon-cache caveat (v1)**: this is a database-only reparent.
    Connector daemons cache ``account_id`` in memory at attach time
    and do NOT receive a rebind event. Restart the connector container
    after reparent — the in-memory cache is otherwise stale until the
    next restart.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionReparent): Request body for ``POST /v1/connections/{id}/reparent``.

            Moves the connection's ``account_id`` to ``destination_account_id``
            atomically, preserving ``connection.id`` so dependent connector
            state (signal-cli's ``account.dat``, whatsmeow's ``sqlstore.db``,
            telegram webhook config) carries over without recreation. v1
            auth: root operator only.

            Length bounds match the ULID-shaped ``account_id`` format used
            elsewhere on the wire (1..64 chars covers ``acc_<ULID>``).

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            connection_id=connection_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
