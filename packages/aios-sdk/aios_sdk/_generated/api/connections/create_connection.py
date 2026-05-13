from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_create import ConnectionCreate
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: ConnectionCreate,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connections",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Connection | HTTPValidationError | None:
    if response.status_code == 201:
        response_201 = Connection.from_dict(response.json())

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
) -> Response[Connection | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    """Create

     Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the connector-scoped
    ``GET /v1/connectors/secrets`` route — operator-facing reads return
    ``secrets_set: bool`` instead of values.

    Args:
        authorization (None | str | Unset):
        body (ConnectionCreate): Request body for ``POST /v1/connections``.

            Created in detached mode — neither ``session_id`` nor
            ``session_template_id`` is set.  Use ``POST .../attach`` or
            ``POST .../configure-per-chat`` afterward to bind a routing mode.

            ``connector`` and ``account`` may not contain ``/`` — they're used
            in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
            and a ``/`` would create ambiguous segment boundaries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
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
    body: ConnectionCreate,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    """Create

     Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the connector-scoped
    ``GET /v1/connectors/secrets`` route — operator-facing reads return
    ``secrets_set: bool`` instead of values.

    Args:
        authorization (None | str | Unset):
        body (ConnectionCreate): Request body for ``POST /v1/connections``.

            Created in detached mode — neither ``session_id`` nor
            ``session_template_id`` is set.  Use ``POST .../attach`` or
            ``POST .../configure-per-chat`` afterward to bind a routing mode.

            ``connector`` and ``account`` may not contain ``/`` — they're used
            in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
            and a ``/`` would create ambiguous segment boundaries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ConnectionCreate,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    """Create

     Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the connector-scoped
    ``GET /v1/connectors/secrets`` route — operator-facing reads return
    ``secrets_set: bool`` instead of values.

    Args:
        authorization (None | str | Unset):
        body (ConnectionCreate): Request body for ``POST /v1/connections``.

            Created in detached mode — neither ``session_id`` nor
            ``session_template_id`` is set.  Use ``POST .../attach`` or
            ``POST .../configure-per-chat`` afterward to bind a routing mode.

            ``connector`` and ``account`` may not contain ``/`` — they're used
            in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
            and a ``/`` would create ambiguous segment boundaries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Connection | HTTPValidationError]
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
    body: ConnectionCreate,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    """Create

     Create a detached connection, **idempotent on ``(connector, account)``**.

    Per plan decision #5, this endpoint and the supervisor's
    auto-create-on-first-inbound path race-safely converge on a single row:
    posting twice with the same ``(connector, account)`` returns 201 with the
    existing row rather than 409.  The ``id`` may differ from a freshly-allocated
    one if a concurrent writer landed first; the response always reflects the
    canonical active row.

    Optional ``secrets`` carry platform credentials (e.g. Telegram
    ``bot_token``).  They are encrypted at rest via ``AIOS_VAULT_KEY``
    and only ever read back through the connector-scoped
    ``GET /v1/connectors/secrets`` route — operator-facing reads return
    ``secrets_set: bool`` instead of values.

    Args:
        authorization (None | str | Unset):
        body (ConnectionCreate): Request body for ``POST /v1/connections``.

            Created in detached mode — neither ``session_id`` nor
            ``session_template_id`` is set.  Use ``POST .../attach`` or
            ``POST .../configure-per-chat`` afterward to bind a routing mode.

            ``connector`` and ``account`` may not contain ``/`` — they're used
            in the focal-channel address scheme ``{connector}/{account}/{chat_id}``
            and a ``/`` would create ambiguous segment boundaries.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Connection | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
