from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.http_validation_error import HTTPValidationError
from ...models.o_auth_complete_request import OAuthCompleteRequest
from ...models.vault_credential import VaultCredential
from ...types import UNSET, Response, Unset


def _get_kwargs(
    vault_id: str,
    *,
    body: OAuthCompleteRequest,
    authorization: None | str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(authorization, Unset):
        headers["Authorization"] = authorization

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/vaults/{vault_id}/credentials/oauth/complete".format(
            vault_id=quote(str(vault_id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> HTTPValidationError | VaultCredential | None:
    if response.status_code == 201:
        response_201 = VaultCredential.from_dict(response.json())

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
) -> Response[HTTPValidationError | VaultCredential]:
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
    body: OAuthCompleteRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Complete Credential Oauth

     Finish an interactive OAuth flow: exchange the code and store the credential.

    Validates the ``state`` against the in-progress flow, exchanges the
    authorization ``code`` for tokens, and stores them as an ``oauth2_refresh``
    credential (creating a new one, or rotating an existing credential for the
    same ``target_url``). Secrets are encrypted at rest and never returned.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthCompleteRequest): Finish an interactive OAuth flow: exchange the returned code
            for tokens.

            The ``state`` correlates back to the in-progress flow (and guards CSRF);
            ``code`` is the authorization code the provider returned to the callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
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
    body: OAuthCompleteRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Complete Credential Oauth

     Finish an interactive OAuth flow: exchange the code and store the credential.

    Validates the ``state`` against the in-progress flow, exchanges the
    authorization ``code`` for tokens, and stores them as an ``oauth2_refresh``
    credential (creating a new one, or rotating an existing credential for the
    same ``target_url``). Secrets are encrypted at rest and never returned.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthCompleteRequest): Finish an interactive OAuth flow: exchange the returned code
            for tokens.

            The ``state`` correlates back to the in-progress flow (and guards CSRF);
            ``code`` is the authorization code the provider returned to the callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
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
    body: OAuthCompleteRequest,
    authorization: None | str | Unset = UNSET,
) -> Response[HTTPValidationError | VaultCredential]:
    """Complete Credential Oauth

     Finish an interactive OAuth flow: exchange the code and store the credential.

    Validates the ``state`` against the in-progress flow, exchanges the
    authorization ``code`` for tokens, and stores them as an ``oauth2_refresh``
    credential (creating a new one, or rotating an existing credential for the
    same ``target_url``). Secrets are encrypted at rest and never returned.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthCompleteRequest): Finish an interactive OAuth flow: exchange the returned code
            for tokens.

            The ``state`` correlates back to the in-progress flow (and guards CSRF);
            ``code`` is the authorization code the provider returned to the callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[HTTPValidationError | VaultCredential]
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
    body: OAuthCompleteRequest,
    authorization: None | str | Unset = UNSET,
) -> HTTPValidationError | VaultCredential | None:
    """Complete Credential Oauth

     Finish an interactive OAuth flow: exchange the code and store the credential.

    Validates the ``state`` against the in-progress flow, exchanges the
    authorization ``code`` for tokens, and stores them as an ``oauth2_refresh``
    credential (creating a new one, or rotating an existing credential for the
    same ``target_url``). Secrets are encrypted at rest and never returned.

    Args:
        vault_id (str):
        authorization (None | str | Unset):
        body (OAuthCompleteRequest): Finish an interactive OAuth flow: exchange the returned code
            for tokens.

            The ``state`` correlates back to the in-progress flow (and guards CSRF);
            ``code`` is the authorization code the provider returned to the callback.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        HTTPValidationError | VaultCredential
    """

    return (
        await asyncio_detailed(
            vault_id=vault_id,
            client=client,
            body=body,
            authorization=authorization,
        )
    ).parsed
