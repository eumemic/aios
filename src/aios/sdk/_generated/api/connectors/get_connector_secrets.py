from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_secrets import ConnectorSecrets
from ...models.http_validation_error import HTTPValidationError
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/connectors/secrets",
    }

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorSecrets | HTTPValidationError | None:
    if response.status_code == 200:
        response_200 = ConnectorSecrets.from_dict(response.json())

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
) -> Response[ConnectorSecrets | HTTPValidationError]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorSecrets | HTTPValidationError]:
    r"""Get Secrets

     Decrypted secrets for the caller's connection.

    The bearer token resolves server-side to one ``connection_id``;
    operators set secrets on that connection via
    ``POST /v1/connections`` or ``PUT /v1/connections/{id}/secrets`` and
    never read them back through the operator surface.  This is the only
    decryption path.

    Returns ``{\"secrets\": {}}`` when no secrets are configured — the
    connector author decides whether that's acceptable (most need at
    least one credential and should fail loudly).

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorSecrets | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> ConnectorSecrets | HTTPValidationError | None:
    r"""Get Secrets

     Decrypted secrets for the caller's connection.

    The bearer token resolves server-side to one ``connection_id``;
    operators set secrets on that connection via
    ``POST /v1/connections`` or ``PUT /v1/connections/{id}/secrets`` and
    never read them back through the operator surface.  This is the only
    decryption path.

    Returns ``{\"secrets\": {}}`` when no secrets are configured — the
    connector author decides whether that's acceptable (most need at
    least one credential and should fail loudly).

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorSecrets | HTTPValidationError
    """

    return sync_detailed(
        client=client,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> Response[ConnectorSecrets | HTTPValidationError]:
    r"""Get Secrets

     Decrypted secrets for the caller's connection.

    The bearer token resolves server-side to one ``connection_id``;
    operators set secrets on that connection via
    ``POST /v1/connections`` or ``PUT /v1/connections/{id}/secrets`` and
    never read them back through the operator surface.  This is the only
    decryption path.

    Returns ``{\"secrets\": {}}`` when no secrets are configured — the
    connector author decides whether that's acceptable (most need at
    least one credential and should fail loudly).

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorSecrets | HTTPValidationError]
    """

    kwargs = _get_kwargs(
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    authorization: None | str | Unset = UNSET,
) -> ConnectorSecrets | HTTPValidationError | None:
    r"""Get Secrets

     Decrypted secrets for the caller's connection.

    The bearer token resolves server-side to one ``connection_id``;
    operators set secrets on that connection via
    ``POST /v1/connections`` or ``PUT /v1/connections/{id}/secrets`` and
    never read them back through the operator surface.  This is the only
    decryption path.

    Returns ``{\"secrets\": {}}`` when no secrets are configured — the
    connector author decides whether that's acceptable (most need at
    least one credential and should fail loudly).

    Args:
        authorization (None | str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorSecrets | HTTPValidationError
    """

    return (
        await asyncio_detailed(
            client=client,
            authorization=authorization,
        )
    ).parsed
