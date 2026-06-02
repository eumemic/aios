from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.o_auth_start_request import OAuthStartRequest
from ...models.o_auth_start_response import OAuthStartResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    *,
    body: OAuthStartRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/vaults/{vault_id}/credentials/oauth/start".format(
            vault_id=quote(str(vault_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | OAuthStartResponse | None:
    if response.status_code == 200:
        response_200 = OAuthStartResponse.from_dict(response.json())

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
) -> Response[HTTPValidationError | OAuthStartResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: OAuthStartRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OAuthStartResponse]:
    r"""Start Credential Oauth

     Begin an interactive OAuth \"Connect\" flow for an MCP server.

    Discovers the target's OAuth metadata, registers a client (RFC 7591
    Dynamic Client Registration) or uses a caller-supplied ``client_id`` /
    ``client_secret`` for servers without DCR, generates PKCE + a CSRF
    ``state``, and returns the provider ``authorization_url`` to redirect the
    user to. The token fields are obtained from the provider — the caller does
    not supply them. Complete the flow with the returned ``state`` + the
    authorization ``code`` via ``complete_vault_credential_oauth``.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthStartRequest): Begin an interactive OAuth authorization-code flow for an MCP
            server.

            With the token fields left blank, the server discovers the target's OAuth
            metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
            the supplied ``client_id``/``client_secret``, and returns an
            ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
            console's callback and is reused verbatim on completion.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OAuthStartResponse]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        body=body,
        authorization=authorization,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: OAuthStartRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OAuthStartResponse | None:
    r"""Start Credential Oauth

     Begin an interactive OAuth \"Connect\" flow for an MCP server.

    Discovers the target's OAuth metadata, registers a client (RFC 7591
    Dynamic Client Registration) or uses a caller-supplied ``client_id`` /
    ``client_secret`` for servers without DCR, generates PKCE + a CSRF
    ``state``, and returns the provider ``authorization_url`` to redirect the
    user to. The token fields are obtained from the provider — the caller does
    not supply them. Complete the flow with the returned ``state`` + the
    authorization ``code`` via ``complete_vault_credential_oauth``.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthStartRequest): Begin an interactive OAuth authorization-code flow for an MCP
            server.

            With the token fields left blank, the server discovers the target's OAuth
            metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
            the supplied ``client_id``/``client_secret``, and returns an
            ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
            console's callback and is reused verbatim on completion.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OAuthStartResponse
    """

    return sync_detailed(
        vault_id=vault_id,
        client=client,
        body=body,
        authorization=authorization,
    ).parsed


async def asyncio_detailed(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: OAuthStartRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | OAuthStartResponse]:
    r"""Start Credential Oauth

     Begin an interactive OAuth \"Connect\" flow for an MCP server.

    Discovers the target's OAuth metadata, registers a client (RFC 7591
    Dynamic Client Registration) or uses a caller-supplied ``client_id`` /
    ``client_secret`` for servers without DCR, generates PKCE + a CSRF
    ``state``, and returns the provider ``authorization_url`` to redirect the
    user to. The token fields are obtained from the provider — the caller does
    not supply them. Complete the flow with the returned ``state`` + the
    authorization ``code`` via ``complete_vault_credential_oauth``.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthStartRequest): Begin an interactive OAuth authorization-code flow for an MCP
            server.

            With the token fields left blank, the server discovers the target's OAuth
            metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
            the supplied ``client_id``/``client_secret``, and returns an
            ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
            console's callback and is reused verbatim on completion.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | OAuthStartResponse]
    """

    kwargs = _get_kwargs(
        vault_id=vault_id,
        body=body,
        authorization=authorization,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    vault_id: str,
    *,
    client: AuthenticatedClient | Client,
    body: OAuthStartRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | OAuthStartResponse | None:
    r"""Start Credential Oauth

     Begin an interactive OAuth \"Connect\" flow for an MCP server.

    Discovers the target's OAuth metadata, registers a client (RFC 7591
    Dynamic Client Registration) or uses a caller-supplied ``client_id`` /
    ``client_secret`` for servers without DCR, generates PKCE + a CSRF
    ``state``, and returns the provider ``authorization_url`` to redirect the
    user to. The token fields are obtained from the provider — the caller does
    not supply them. Complete the flow with the returned ``state`` + the
    authorization ``code`` via ``complete_vault_credential_oauth``.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthStartRequest): Begin an interactive OAuth authorization-code flow for an MCP
            server.

            With the token fields left blank, the server discovers the target's OAuth
            metadata, registers a client (RFC 7591 Dynamic Client Registration) or uses
            the supplied ``client_id``/``client_secret``, and returns an
            ``authorization_url`` to redirect the user to. The ``redirect_uri`` is the
            console's callback and is reused verbatim on completion.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | OAuthStartResponse
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
