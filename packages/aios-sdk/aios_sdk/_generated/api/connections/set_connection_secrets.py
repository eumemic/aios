from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connection import Connection
from ...models.connection_set_secrets import ConnectionSetSecrets
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    connection_id: str,
    *,
    body: ConnectionSetSecrets,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/connections/{connection_id}/secrets".format(
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
    body: ConnectionSetSecrets,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Set Secrets

     Replace the connection's encrypted secrets dict, wholesale.

    The request body fully replaces the stored blob.  Pass
    ``{\"secrets\": {}}`` to clear secrets entirely.  Operator-facing reads
    only ever expose ``secrets_set: bool``; the decrypted values are
    exclusively available via the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route to a connector container
    holding a runtime token for this connector type.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionSetSecrets): Request body for ``PUT /v1/connections/{id}/secrets``.

            Replaces the connection's secrets dict wholesale.  Encrypted at
            rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
            them back.

            Pass an empty dict to clear secrets.

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
    body: ConnectionSetSecrets,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Set Secrets

     Replace the connection's encrypted secrets dict, wholesale.

    The request body fully replaces the stored blob.  Pass
    ``{\"secrets\": {}}`` to clear secrets entirely.  Operator-facing reads
    only ever expose ``secrets_set: bool``; the decrypted values are
    exclusively available via the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route to a connector container
    holding a runtime token for this connector type.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionSetSecrets): Request body for ``PUT /v1/connections/{id}/secrets``.

            Replaces the connection's secrets dict wholesale.  Encrypted at
            rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
            them back.

            Pass an empty dict to clear secrets.

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
    body: ConnectionSetSecrets,
    authorization: None | str | Unset = UNSET,
) -> Response[Connection | HTTPValidationError]:
    r"""Set Secrets

     Replace the connection's encrypted secrets dict, wholesale.

    The request body fully replaces the stored blob.  Pass
    ``{\"secrets\": {}}`` to clear secrets entirely.  Operator-facing reads
    only ever expose ``secrets_set: bool``; the decrypted values are
    exclusively available via the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route to a connector container
    holding a runtime token for this connector type.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionSetSecrets): Request body for ``PUT /v1/connections/{id}/secrets``.

            Replaces the connection's secrets dict wholesale.  Encrypted at
            rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
            them back.

            Pass an empty dict to clear secrets.

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
    body: ConnectionSetSecrets,
    authorization: None | str | Unset = UNSET,
) -> Connection | HTTPValidationError | None:
    r"""Set Secrets

     Replace the connection's encrypted secrets dict, wholesale.

    The request body fully replaces the stored blob.  Pass
    ``{\"secrets\": {}}`` to clear secrets entirely.  Operator-facing reads
    only ever expose ``secrets_set: bool``; the decrypted values are
    exclusively available via the runtime-scoped
    ``GET /v1/connectors/runtime/secrets`` route to a connector container
    holding a runtime token for this connector type.

    Args:
        connection_id (str):
        authorization (None | str | Unset):
        body (ConnectionSetSecrets): Request body for ``PUT /v1/connections/{id}/secrets``.

            Replaces the connection's secrets dict wholesale.  Encrypted at
            rest server-side via ``AIOS_VAULT_KEY``; the operator never reads
            them back.

            Pass an empty dict to clear secrets.

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
